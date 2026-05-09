# online_influence_gnn.py
#
# Online approach:
# - Build a known base influence graph from 511 labeled posts
# - Add one new social media post to the graph
# - Classify only that new post as True/Fake
# - Report ROC AUC, accuracy, F1, total time, and average time per post
#
# Labels:
#   0 = True
#   1 = Fake
#
# Install:
#   pip install pandas numpy scikit-learn torch sentence-transformers torch-geometric

import random
import time
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.utils import resample

import torch
import torch.nn as nn
import torch.nn.functional as F

from sentence_transformers import SentenceTransformer

from torch_geometric.data import Data
from torch_geometric.nn import GCNConv


# ============================================================
# Reproducibility
# ============================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# Config
# ============================================================
@dataclass
class Config:
    true_csv: str = "True.csv"
    fake_csv: str = "Fake.csv"

    text_model_name: str = "all-MiniLM-L6-v2"
    batch_size_embed: int = 256

    base_graph_size: int = 1000
    influence_threshold: float = 0.323

    max_edges_per_node: int = 5
    add_self_loops: bool = True
    ensure_graph_connectivity: bool = True

    hidden_dim: int = 128
    dropout: float = 0.30
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 20

    train_frac: float = 0.75
    val_frac: float = 0.125
    test_frac: float = 0.125

    results_csv: str = "online_single_post_results.csv"
    predictions_csv: str = "online_single_post_predictions.csv"


CFG = Config()


# ============================================================
# CSV loading
# ============================================================
def read_news_csv(path: str) -> pd.DataFrame:
    required = {"title", "text", "subject", "date"}

    df = pd.read_csv(path)
    df.columns = [str(c).strip().lower() for c in df.columns]

    unnamed = [c for c in df.columns if c.startswith("unnamed")]
    if unnamed:
        df = df.drop(columns=unnamed)

    if required.issubset(df.columns):
        return df

    df2 = pd.read_csv(path, index_col=0)
    df2.columns = [str(c).strip().lower() for c in df2.columns]

    unnamed2 = [c for c in df2.columns if c.startswith("unnamed")]
    if unnamed2:
        df2 = df2.drop(columns=unnamed2)

    if required.issubset(df2.columns):
        return df2

    raise ValueError(
        f"{path} does not have the required columns.\n"
        f"Normal read columns: {df.columns.tolist()}\n"
        f"index_col=0 read columns: {df2.columns.tolist()}"
    )


def safe_parse_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series.astype(str).str.strip(), errors="coerce").dt.normalize()


def load_data(true_csv: str, fake_csv: str) -> pd.DataFrame:
    true_df = read_news_csv(true_csv)
    fake_df = read_news_csv(fake_csv)

    true_df = true_df.copy()
    fake_df = fake_df.copy()

    true_df["label"] = 0
    fake_df["label"] = 1

    print("Raw rows:")
    print("  True:", len(true_df))
    print("  Fake:", len(fake_df))

    true_df["date"] = safe_parse_date(true_df["date"])
    fake_df["date"] = safe_parse_date(fake_df["date"])

    true_df = true_df.dropna(subset=["date"]).reset_index(drop=True)
    fake_df = fake_df.dropna(subset=["date"]).reset_index(drop=True)

    print("After date cleaning:")
    print("  True:", len(true_df))
    print("  Fake:", len(fake_df))

    if len(true_df) == 0 or len(fake_df) == 0:
        raise ValueError("One class is empty after date parsing.")

    df = pd.concat([true_df, fake_df], ignore_index=True)
    df = df[["title", "text", "subject", "date", "label"]].copy()
    df = df.sort_values("date").reset_index(drop=True)

    print("Final label counts:")
    print(df["label"].value_counts().sort_index())

    return df


# ============================================================
# Split and balance
# ============================================================
def chronological_split_by_class(
    df: pd.DataFrame,
    train_frac: float,
    val_frac: float,
    test_frac: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    split_parts = []

    for label in [0, 1]:
        d = df[df["label"] == label].sort_values("date").reset_index(drop=True)
        n = len(d)

        train_end = int(n * train_frac)
        val_end = train_end + int(n * val_frac)

        train_part = d.iloc[:train_end].copy()
        val_part = d.iloc[train_end:val_end].copy()
        test_part = d.iloc[val_end:].copy()

        split_parts.append((train_part, val_part, test_part))

    train_df = pd.concat([split_parts[0][0], split_parts[1][0]], ignore_index=True)
    val_df = pd.concat([split_parts[0][1], split_parts[1][1]], ignore_index=True)
    test_df = pd.concat([split_parts[0][2], split_parts[1][2]], ignore_index=True)

    train_df = train_df.sort_values("date").reset_index(drop=True)
    val_df = val_df.sort_values("date").reset_index(drop=True)
    test_df = test_df.sort_values("date").reset_index(drop=True)

    return train_df, val_df, test_df


def balance_training_set(train_df: pd.DataFrame) -> pd.DataFrame:
    true_part = train_df[train_df["label"] == 0]
    fake_part = train_df[train_df["label"] == 1]

    n_min = min(len(true_part), len(fake_part))

    if n_min == 0:
        raise ValueError("Training set lost one class.")

    true_bal = resample(true_part, replace=False, n_samples=n_min, random_state=SEED)
    fake_bal = resample(fake_part, replace=False, n_samples=n_min, random_state=SEED)

    out = pd.concat([true_bal, fake_bal], ignore_index=True)
    out = out.sort_values("date").reset_index(drop=True)

    return out


# ============================================================
# Embeddings and graph construction
# ============================================================
def combine_title_text(df: pd.DataFrame) -> List[str]:
    title = df["title"].fillna("").astype(str)
    text = df["text"].fillna("").astype(str)
    return (title + " [SEP] " + text).tolist()


def embed_dataframe_texts(
    df: pd.DataFrame,
    encoder: SentenceTransformer,
    batch_size: int,
) -> np.ndarray:
    texts = combine_title_text(df)

    embeddings = encoder.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )

    return embeddings.astype(np.float32)


def cosine_similarity_matrix(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    x_norm = x / norms
    return x_norm @ x_norm.T


def normalize_edge_weight(sim: float, threshold: float) -> float:
    if sim <= threshold:
        return 0.0

    denom = max(1.0 - threshold, 1e-8)
    return (sim - threshold) / denom


def build_influence_graph(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    threshold: float,
    max_edges_per_node: int,
    add_self_loops: bool = True,
    ensure_graph_connectivity: bool = True,
) -> Data:

    n = len(df)

    if n != len(embeddings):
        raise ValueError("DataFrame and embedding length mismatch.")

    if n == 1:
        x = torch.tensor(embeddings, dtype=torch.float)
        y = torch.tensor(df["label"].values, dtype=torch.long)

        if add_self_loops:
            edge_index = torch.tensor([[0], [0]], dtype=torch.long)
            edge_weight = torch.tensor([1.0], dtype=torch.float)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_weight = torch.empty((0,), dtype=torch.float)

        return Data(x=x, edge_index=edge_index, edge_weight=edge_weight, y=y)

    sims = cosine_similarity_matrix(embeddings)
    dates = df["date"].tolist()

    src_list = []
    dst_list = []
    wt_list = []

    for i in range(n):
        candidates = []

        for j in range(n):
            if i == j:
                continue

            # Older posts point to newer posts only
            if dates[i] >= dates[j]:
                continue

            # Same-day posts have no connection
            if dates[i] == dates[j]:
                continue

            sim = float(sims[i, j])

            if sim > threshold:
                candidates.append((j, sim))

        candidates.sort(key=lambda t: t[1], reverse=True)
        candidates = candidates[:max_edges_per_node]

        for j, sim in candidates:
            src_list.append(i)
            dst_list.append(j)
            wt_list.append(normalize_edge_weight(sim, threshold))

    # Fallback if threshold removes all edges
    if ensure_graph_connectivity and len(src_list) == 0:
        for i in range(n):
            future_candidates = []

            for j in range(n):
                if i == j:
                    continue

                if dates[i] < dates[j] and dates[i] != dates[j]:
                    future_candidates.append((j, float(sims[i, j])))

            if future_candidates:
                j_best, sim_best = max(future_candidates, key=lambda t: t[1])
                src_list.append(i)
                dst_list.append(j_best)
                wt_list.append(max(sim_best, 1e-4))

    if add_self_loops:
        for i in range(n):
            src_list.append(i)
            dst_list.append(i)
            wt_list.append(1.0)

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    edge_weight = torch.tensor(wt_list, dtype=torch.float)
    x = torch.tensor(embeddings, dtype=torch.float)
    y = torch.tensor(df["label"].values, dtype=torch.long)

    return Data(x=x, edge_index=edge_index, edge_weight=edge_weight, y=y)


# ============================================================
# Model
# ============================================================
class InfluenceGCN(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, dropout: float):
        super().__init__()

        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.lin = nn.Linear(hidden_dim, 1)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight):
        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        logits = self.lin(x).squeeze(-1)

        return logits


# ============================================================
# Base model training
# ============================================================
def train_base_model(
    base_df: pd.DataFrame,
    base_embeddings: np.ndarray,
) -> InfluenceGCN:

    base_graph = build_influence_graph(
        df=base_df,
        embeddings=base_embeddings,
        threshold=CFG.influence_threshold,
        max_edges_per_node=CFG.max_edges_per_node,
        add_self_loops=CFG.add_self_loops,
        ensure_graph_connectivity=CFG.ensure_graph_connectivity,
    ).to(DEVICE)

    model = InfluenceGCN(
        in_dim=base_graph.x.shape[1],
        hidden_dim=CFG.hidden_dim,
        dropout=CFG.dropout,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=CFG.lr,
        weight_decay=CFG.weight_decay,
    )

    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(1, CFG.epochs + 1):
        model.train()
        optimizer.zero_grad()

        logits = model(
            base_graph.x,
            base_graph.edge_index,
            base_graph.edge_weight,
        )

        loss = criterion(logits, base_graph.y.float())
        loss.backward()
        optimizer.step()

        print(f"Base Training Epoch {epoch:02d} | Loss: {loss.item():.4f}")

    return model


# ============================================================
# Online one-post evaluation
# ============================================================
def classify_single_online_post(
    model: InfluenceGCN,
    base_df: pd.DataFrame,
    base_embeddings: np.ndarray,
    new_post_df: pd.DataFrame,
    encoder: SentenceTransformer,
) -> Tuple[int, float, int, float]:

    start = time.time()

    new_embedding = embed_dataframe_texts(
        new_post_df,
        encoder,
        batch_size=1,
    )

    online_df = pd.concat([base_df, new_post_df], ignore_index=True)
    online_embeddings = np.vstack([base_embeddings, new_embedding])

    online_df = online_df.sort_values("date").reset_index(drop=True)

    # Reorder embeddings to match sorted online_df
    # Use a temporary ID so the new post can be tracked correctly
    base_temp = base_df.copy()
    base_temp["_row_id"] = np.arange(len(base_temp))

    new_temp = new_post_df.copy()
    new_temp["_row_id"] = len(base_temp)

    combined_temp = pd.concat([base_temp, new_temp], ignore_index=True)
    combined_temp = combined_temp.sort_values("date").reset_index(drop=True)

    reorder_indices = combined_temp["_row_id"].values
    online_embeddings = online_embeddings[reorder_indices]

    new_node_idx = int(np.where(reorder_indices == len(base_temp))[0][0])

    online_graph = build_influence_graph(
        df=online_df,
        embeddings=online_embeddings,
        threshold=CFG.influence_threshold,
        max_edges_per_node=CFG.max_edges_per_node,
        add_self_loops=CFG.add_self_loops,
        ensure_graph_connectivity=CFG.ensure_graph_connectivity,
    ).to(DEVICE)

    model.eval()
    with torch.no_grad():
        logits = model(
            online_graph.x,
            online_graph.edge_index,
            online_graph.edge_weight,
        )

        new_logit = logits[new_node_idx]
        prob_fake = torch.sigmoid(new_logit).item()
        pred_label = 1 if prob_fake >= 0.5 else 0

    true_label = int(new_post_df.iloc[0]["label"])
    elapsed = time.time() - start

    return true_label, prob_fake, pred_label, elapsed


def online_evaluation(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    encoder: SentenceTransformer,
):

    if len(train_df) < CFG.base_graph_size:
        raise ValueError(
            f"Need at least {CFG.base_graph_size} training posts, "
            f"but only found {len(train_df)}."
        )

    base_df = train_df.sort_values("date").iloc[-CFG.base_graph_size:].reset_index(drop=True)

    print("\n" + "=" * 80)
    print("ONLINE INFLUENCE-GNN EXPERIMENT")
    print("=" * 80)
    print(f"Base graph size: {CFG.base_graph_size}")
    print(f"Influence threshold: {CFG.influence_threshold}")
    print(f"Online test posts: {len(test_df)}")

    print("\nEmbedding base graph...")
    base_embeddings = embed_dataframe_texts(
        base_df,
        encoder,
        batch_size=CFG.batch_size_embed,
    )

    print("\nTraining base GNN...")
    model = train_base_model(
        base_df=base_df,
        base_embeddings=base_embeddings,
    )

    y_true = []
    y_prob = []
    y_pred = []
    times = []

    total_start = time.time()

    for i in range(len(test_df)):
        new_post_df = test_df.iloc[[i]].copy()

        true_label, prob_fake, pred_label, elapsed = classify_single_online_post(
            model=model,
            base_df=base_df,
            base_embeddings=base_embeddings,
            new_post_df=new_post_df,
            encoder=encoder,
        )

        y_true.append(true_label)
        y_prob.append(prob_fake)
        y_pred.append(pred_label)
        times.append(elapsed)

        if (i + 1) % 100 == 0:
            print(
                f"Processed {i + 1}/{len(test_df)} posts | "
                f"Last post time: {elapsed:.4f}s"
            )

    total_time = time.time() - total_start

    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    y_pred = np.array(y_pred)

    roc_auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan")
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    results = {
        "base_graph_size": CFG.base_graph_size,
        "influence_threshold": CFG.influence_threshold,
        "num_online_posts": len(test_df),
        "roc_auc": roc_auc,
        "accuracy": accuracy,
        "f1": f1,
        "total_processing_time_sec": total_time,
        "avg_time_per_post_sec": float(np.mean(times)),
        "median_time_per_post_sec": float(np.median(times)),
    }

    predictions = pd.DataFrame({
        "true_label": y_true,
        "prob_fake": y_prob,
        "pred_label": y_pred,
        "time_sec": times,
    })

    pd.DataFrame([results]).to_csv(CFG.results_csv, index=False)
    predictions.to_csv(CFG.predictions_csv, index=False)

    print("\n" + "=" * 80)
    print("ONLINE RESULTS")
    print("=" * 80)
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Total Processing Time: {total_time:.4f} sec")
    print(f"Average Time per Post: {np.mean(times):.4f} sec")
    print(f"Median Time per Post: {np.median(times):.4f} sec")

    print("\nSaved:")
    print(f"  {CFG.results_csv}")
    print(f"  {CFG.predictions_csv}")

    return results


# ============================================================
# Main
# ============================================================
def main():
    print(f"Using device: {DEVICE}")

    df = load_data(CFG.true_csv, CFG.fake_csv)

    train_df_raw, val_df, test_df = chronological_split_by_class(
        df,
        train_frac=CFG.train_frac,
        val_frac=CFG.val_frac,
        test_frac=CFG.test_frac,
    )

    train_df = balance_training_set(train_df_raw)

    print("\nSplit sizes:")
    print("Train:", len(train_df), train_df["label"].value_counts().to_dict())
    print("Val:", len(val_df), val_df["label"].value_counts().to_dict())
    print("Test:", len(test_df), test_df["label"].value_counts().to_dict())

    encoder = SentenceTransformer(CFG.text_model_name, device=str(DEVICE))

    online_evaluation(
        train_df=train_df,
        test_df=test_df,
        encoder=encoder,
    )


if __name__ == "__main__":
    main()
