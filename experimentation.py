# experimentation.py
#
# Full corrected script for:
# - Loading True.csv and Fake.csv robustly
# - Fixing CSV/index-column issues
# - Parsing dates safely
# - Building weekly influence graphs
# - Training a GNN for node classification (true vs fake)
# - Using chronological-by-class splits
# - Balancing only the training split to 50/50
# - Measuring ROC AUC and processing time across:
#       1) influence network input size
#       2) influence threshold
#
# Labels:
#   0 = True
#   1 = Fake
#
# Install:
#   pip install pandas numpy scikit-learn torch sentence-transformers
#   pip install torch-geometric
#
# If sentence-transformers / huggingface versions conflict, use compatible versions.

import math
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.utils import resample

import torch
import torch.nn as nn
import torch.nn.functional as F

from sentence_transformers import SentenceTransformer

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
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

    # Real-world window assumption:
    # classify posts collected over about one week
    window_days: int = 7
    stride_days: int = 7

    # Independent variables
    input_sizes: Tuple[int, ...] = (32, 64, 128, 256, 512)
    influence_thresholds: Tuple[float, ...] = (0.3, 0.4, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70)

    # Graph construction
    max_edges_per_node: int = 5
    ensure_graph_connectivity: bool = True
    add_self_loops: bool = True

    # Model
    hidden_dim: int = 128
    dropout: float = 0.30

    # Training
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 20
    loader_batch_size: int = 8

    # Splits
    train_frac: float = 0.75
    val_frac: float = 0.125
    test_frac: float = 0.125

    # Output
    results_csv: str = "gnn_experiment_results.csv"


CFG = Config()


# ============================================================
# CSV reading / preprocessing
# ============================================================
def read_news_csv(path: str) -> pd.DataFrame:
    """
    Reads CSV robustly.
    Handles:
    - normal CSVs with columns: title, text, subject, date
    - CSVs with an extra index column at the front
    """
    required = {"title", "text", "subject", "date"}

    # Try normal read first
    df = pd.read_csv(path)
    df.columns = [str(c).strip().lower() for c in df.columns]

    unnamed = [c for c in df.columns if c.startswith("unnamed")]
    if unnamed:
        df = df.drop(columns=unnamed)

    if required.issubset(df.columns):
        return df

    # Try first-column-as-index read
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
    return pd.to_datetime(
        series.astype(str).str.strip(),
        errors="coerce"
    ).dt.normalize()


def load_data(true_csv: str, fake_csv: str) -> pd.DataFrame:
    true_df = read_news_csv(true_csv)
    fake_df = read_news_csv(fake_csv)

    required_cols = {"title", "text", "subject", "date"}
    if not required_cols.issubset(set(true_df.columns)):
        raise ValueError(f"{true_csv} must contain columns: {required_cols}")
    if not required_cols.issubset(set(fake_df.columns)):
        raise ValueError(f"{fake_csv} must contain columns: {required_cols}")

    true_df = true_df.copy()
    fake_df = fake_df.copy()

    true_df["label"] = 0
    fake_df["label"] = 1

    print("Raw rows:")
    print("  True:", len(true_df))
    print("  Fake:", len(fake_df))

    true_df["date"] = safe_parse_date(true_df["date"])
    fake_df["date"] = safe_parse_date(fake_df["date"])

    print("Valid dates:")
    print("  True:", true_df["date"].notna().sum())
    print("  Fake:", fake_df["date"].notna().sum())

    true_df = true_df.dropna(subset=["date"]).reset_index(drop=True)
    fake_df = fake_df.dropna(subset=["date"]).reset_index(drop=True)

    print("After date cleaning:")
    print("  True:", len(true_df))
    print("  Fake:", len(fake_df))

    if len(true_df) == 0 or len(fake_df) == 0:
        raise ValueError(
            f"After date parsing, one class is empty. "
            f"True rows remaining: {len(true_df)}, Fake rows remaining: {len(fake_df)}"
        )

    df = pd.concat([true_df, fake_df], ignore_index=True)
    df = df[["title", "text", "subject", "date", "label"]].copy()
    df = df.sort_values("date").reset_index(drop=True)

    print("Final label counts:")
    print(df["label"].value_counts())

    return df


def combine_title_text(df: pd.DataFrame) -> List[str]:
    title = df["title"].fillna("").astype(str)
    text = df["text"].fillna("").astype(str)
    return (title + " [SEP] " + text).tolist()


# ============================================================
# Splitting
# ============================================================
def chronological_split_by_class(
    df: pd.DataFrame,
    train_frac: float,
    val_frac: float,
    test_frac: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Chronological split within each class, then recombine.
    This preserves time ordering while keeping both classes
    present in train/val/test.
    """
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-8

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
    """
    Enforce 50/50 class balance in training set only.
    """
    true_part = train_df[train_df["label"] == 0]
    fake_part = train_df[train_df["label"] == 1]

    print("Training class counts before balancing:")
    print("  True:", len(true_part))
    print("  Fake:", len(fake_part))

    n_min = min(len(true_part), len(fake_part))
    if n_min == 0:
        raise ValueError(
            "Training split lost one class entirely. "
            f"True={len(true_part)}, Fake={len(fake_part)}."
        )

    true_bal = resample(true_part, replace=False, n_samples=n_min, random_state=SEED)
    fake_bal = resample(fake_part, replace=False, n_samples=n_min, random_state=SEED)

    out = pd.concat([true_bal, fake_bal], ignore_index=True)
    out = out.sort_values("date").reset_index(drop=True)
    return out


# ============================================================
# Time windows
# ============================================================
def make_time_windows(df: pd.DataFrame, window_days: int, stride_days: int) -> List[pd.DataFrame]:
    if df.empty:
        return []

    start_date = df["date"].min()
    end_date = df["date"].max()

    windows = []
    current_start = start_date

    while current_start <= end_date:
        current_end = current_start + pd.Timedelta(days=window_days - 1)
        window_df = df[(df["date"] >= current_start) & (df["date"] <= current_end)].copy()

        if len(window_df) > 0:
            windows.append(window_df.reset_index(drop=True))

        current_start = current_start + pd.Timedelta(days=stride_days)

    return windows


# ============================================================
# Embeddings / similarity
# ============================================================
def embed_dataframe_texts(df: pd.DataFrame, model: SentenceTransformer, batch_size: int) -> np.ndarray:
    texts = combine_title_text(df)
    embeddings = model.encode(
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


# ============================================================
# Graph construction
# ============================================================
def build_influence_graph(
    window_df: pd.DataFrame,
    embeddings: np.ndarray,
    threshold: float,
    max_edges_per_node: int,
    max_nodes: int,
    ensure_graph_connectivity: bool = True,
    add_self_loops: bool = True,
) -> Data:
    """
    Directed graph:
    - older posts -> newer posts only
    - same-date posts are not connected
    - edges if semantic similarity > threshold
    - fallback edges added if threshold creates no usable graph
    """

    # Limit graph input size
    if len(window_df) > max_nodes:
        window_df = window_df.sort_values("date").iloc[-max_nodes:].reset_index(drop=True)
        embeddings = embeddings[-max_nodes:]

    n = len(window_df)

    if n == 1:
        x = torch.tensor(embeddings, dtype=torch.float)
        y = torch.tensor(window_df["label"].values, dtype=torch.long)

        if add_self_loops:
            edge_index = torch.tensor([[0], [0]], dtype=torch.long)
            edge_weight = torch.tensor([1.0], dtype=torch.float)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_weight = torch.empty((0,), dtype=torch.float)

        return Data(x=x, edge_index=edge_index, edge_weight=edge_weight, y=y)

    sims = cosine_similarity_matrix(embeddings)
    dates = window_df["date"].tolist()

    src_list = []
    dst_list = []
    wt_list = []

    for i in range(n):
        candidates = []

        for j in range(n):
            if i == j:
                continue

            # Older -> newer only
            if dates[i] >= dates[j]:
                continue

            # Same date => no connection
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

    # Fallback graph rule so the graph is still usable
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
    y = torch.tensor(window_df["label"].values, dtype=torch.long)

    return Data(x=x, edge_index=edge_index, edge_weight=edge_weight, y=y)


def build_graph_dataset(
    df: pd.DataFrame,
    encoder: SentenceTransformer,
    threshold: float,
    max_nodes: int,
    window_days: int,
    stride_days: int,
    max_edges_per_node: int,
    ensure_graph_connectivity: bool,
    add_self_loops: bool,
    batch_size_embed: int,
) -> Tuple[List[Data], float]:
    start = time.time()

    windows = make_time_windows(df, window_days=window_days, stride_days=stride_days)
    graphs = []

    for wdf in windows:
        emb = embed_dataframe_texts(wdf, encoder, batch_size=batch_size_embed)
        graph = build_influence_graph(
            window_df=wdf,
            embeddings=emb,
            threshold=threshold,
            max_edges_per_node=max_edges_per_node,
            max_nodes=max_nodes,
            ensure_graph_connectivity=ensure_graph_connectivity,
            add_self_loops=add_self_loops,
        )
        graphs.append(graph)

    build_time = time.time() - start
    return graphs, build_time


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
# Evaluation
# ============================================================
def collect_predictions(model: nn.Module, loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_probs = []
    all_true = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            logits = model(batch.x, batch.edge_index, batch.edge_weight)
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            labels = batch.y.detach().cpu().numpy()

            all_probs.append(probs)
            all_true.append(labels)

    all_probs = np.concatenate(all_probs)
    all_true = np.concatenate(all_true)
    return all_true, all_probs


def evaluate_predictions(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    y_pred = (y_prob >= 0.5).astype(int)

    if len(np.unique(y_true)) < 2:
        auc = float("nan")
    else:
        auc = roc_auc_score(y_true, y_prob)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    return {
        "roc_auc": auc,
        "accuracy": acc,
        "f1": f1,
    }


# ============================================================
# Training
# ============================================================
def train_one_experiment(
    train_graphs: List[Data],
    val_graphs: List[Data],
    test_graphs: List[Data],
    hidden_dim: int,
    dropout: float,
    lr: float,
    weight_decay: float,
    epochs: int,
    loader_batch_size: int,
) -> Dict[str, float]:
    if len(train_graphs) == 0 or len(val_graphs) == 0 or len(test_graphs) == 0:
        raise ValueError("One of the graph splits is empty. Adjust the window settings or data split.")

    in_dim = train_graphs[0].x.shape[1]

    train_loader = DataLoader(train_graphs, batch_size=loader_batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=loader_batch_size, shuffle=False)
    test_loader = DataLoader(test_graphs, batch_size=loader_batch_size, shuffle=False)

    model = InfluenceGCN(in_dim=in_dim, hidden_dim=hidden_dim, dropout=dropout).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    best_val_auc = -float("inf")
    best_state = None

    train_start = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()

            logits = model(batch.x, batch.edge_index, batch.edge_weight)
            labels = batch.y.float()

            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        y_val_true, y_val_prob = collect_predictions(model, val_loader)
        val_metrics = evaluate_predictions(y_val_true, y_val_prob)
        val_auc = val_metrics["roc_auc"]

        if not math.isnan(val_auc) and val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        print(
            f"Epoch {epoch:02d} | "
            f"Loss: {epoch_loss / max(n_batches, 1):.4f} | "
            f"Val ROC AUC: {val_auc:.4f}"
        )

    training_time = time.time() - train_start

    if best_state is not None:
        model.load_state_dict(best_state)

    inference_start = time.time()
    y_train_true, y_train_prob = collect_predictions(model, train_loader)
    y_val_true, y_val_prob = collect_predictions(model, val_loader)
    y_test_true, y_test_prob = collect_predictions(model, test_loader)
    inference_time = time.time() - inference_start

    train_metrics = evaluate_predictions(y_train_true, y_train_prob)
    val_metrics = evaluate_predictions(y_val_true, y_val_prob)
    test_metrics = evaluate_predictions(y_test_true, y_test_prob)

    return {
        "train_roc_auc": train_metrics["roc_auc"],
        "val_roc_auc": val_metrics["roc_auc"],
        "test_roc_auc": test_metrics["roc_auc"],
        "train_accuracy": train_metrics["accuracy"],
        "val_accuracy": val_metrics["accuracy"],
        "test_accuracy": test_metrics["accuracy"],
        "train_f1": train_metrics["f1"],
        "val_f1": val_metrics["f1"],
        "test_f1": test_metrics["f1"],
        "training_time_sec": training_time,
        "inference_time_sec": inference_time,
    }


# ============================================================
# Main
# ============================================================
def main():
    print(f"Using device: {DEVICE}")

    df = load_data(CFG.true_csv, CFG.fake_csv)
    print("Loaded rows:", len(df))
    print("Label counts:\n", df["label"].value_counts().sort_index())

    # Chronological split within each class
    train_df_raw, val_df, test_df = chronological_split_by_class(
        df,
        train_frac=CFG.train_frac,
        val_frac=CFG.val_frac,
        test_frac=CFG.test_frac,
    )

    # Training only: force 50/50 balance
    train_df = balance_training_set(train_df_raw)

    print("\nSplit sizes")
    print("Train (balanced):", len(train_df), "| class counts:", train_df["label"].value_counts().to_dict())
    print("Val:", len(val_df), "| class counts:", val_df["label"].value_counts().to_dict())
    print("Test:", len(test_df), "| class counts:", test_df["label"].value_counts().to_dict())

    encoder = SentenceTransformer(CFG.text_model_name, device=str(DEVICE))

    all_results = []

    for max_nodes in CFG.input_sizes:
        for threshold in CFG.influence_thresholds:
            print("\n" + "=" * 90)
            print(f"Experiment | input_size={max_nodes} | influence_threshold={threshold:.2f}")

            train_graphs, train_build_time = build_graph_dataset(
                df=train_df,
                encoder=encoder,
                threshold=threshold,
                max_nodes=max_nodes,
                window_days=CFG.window_days,
                stride_days=CFG.stride_days,
                max_edges_per_node=CFG.max_edges_per_node,
                ensure_graph_connectivity=CFG.ensure_graph_connectivity,
                add_self_loops=CFG.add_self_loops,
                batch_size_embed=CFG.batch_size_embed,
            )

            val_graphs, val_build_time = build_graph_dataset(
                df=val_df,
                encoder=encoder,
                threshold=threshold,
                max_nodes=max_nodes,
                window_days=CFG.window_days,
                stride_days=CFG.stride_days,
                max_edges_per_node=CFG.max_edges_per_node,
                ensure_graph_connectivity=CFG.ensure_graph_connectivity,
                add_self_loops=CFG.add_self_loops,
                batch_size_embed=CFG.batch_size_embed,
            )

            test_graphs, test_build_time = build_graph_dataset(
                df=test_df,
                encoder=encoder,
                threshold=threshold,
                max_nodes=max_nodes,
                window_days=CFG.window_days,
                stride_days=CFG.stride_days,
                max_edges_per_node=CFG.max_edges_per_node,
                ensure_graph_connectivity=CFG.ensure_graph_connectivity,
                add_self_loops=CFG.add_self_loops,
                batch_size_embed=CFG.batch_size_embed,
            )

            total_graph_build_time = train_build_time + val_build_time + test_build_time

            print(f"Graphs built: train={len(train_graphs)}, val={len(val_graphs)}, test={len(test_graphs)}")
            print(f"Graph construction time: {total_graph_build_time:.2f} sec")

            metrics = train_one_experiment(
                train_graphs=train_graphs,
                val_graphs=val_graphs,
                test_graphs=test_graphs,
                hidden_dim=CFG.hidden_dim,
                dropout=CFG.dropout,
                lr=CFG.lr,
                weight_decay=CFG.weight_decay,
                epochs=CFG.epochs,
                loader_batch_size=CFG.loader_batch_size,
            )

            result_row = {
                "input_size": max_nodes,
                "influence_threshold": threshold,
                "window_days": CFG.window_days,
                "stride_days": CFG.stride_days,
                "train_graphs": len(train_graphs),
                "val_graphs": len(val_graphs),
                "test_graphs": len(test_graphs),
                "graph_build_time_sec": total_graph_build_time,
                "total_processing_time_sec": (
                    total_graph_build_time
                    + metrics["training_time_sec"]
                    + metrics["inference_time_sec"]
                ),
                **metrics,
            }

            all_results.append(result_row)

            print("Train ROC AUC:", result_row["train_roc_auc"])
            print("Val ROC AUC:", result_row["val_roc_auc"])
            print("Test ROC AUC:", result_row["test_roc_auc"])
            print("Total processing time (sec):", result_row["total_processing_time_sec"])

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(CFG.results_csv, index=False)

    print("\nSaved experiment results to:", CFG.results_csv)
    print("\nTop experiments by test ROC AUC:")
    print(results_df.sort_values("test_roc_auc", ascending=False).head(10))


if __name__ == "__main__":
    main()
