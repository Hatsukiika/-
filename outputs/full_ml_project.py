from __future__ import annotations

import csv
import html
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "work" / "第3章 数据集"
CSV_DIR = DATA_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
VIS_DIR = OUTPUT_DIR / "visualizations"

RAW_COLUMNS = [
    "file_path",
    "label_raw",
    "title",
    "author_keywords",
    "abstract",
    "title_tokens",
    "title_pos",
    "abstract_tokens",
    "abstract_pos",
    "auto_keywords_a",
    "auto_keywords_b",
    "auto_keywords_c",
]
TEXT_COLUMNS = ["title", "author_keywords", "abstract", "title_tokens", "abstract_tokens"]
TOKEN_COLUMNS = ["title_tokens", "author_keywords", "abstract_tokens"]


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    VIS_DIR.mkdir(exist_ok=True)


def tokenize(value: object) -> list[str]:
    if pd.isna(value):
        return []
    return [token for token in str(value).strip().split() if token]


def load_raw_data() -> pd.DataFrame:
    frames = []
    for split in ["train", "val", "test"]:
        path = CSV_DIR / f"{split}.csv"
        df = pd.read_csv(path, header=None, names=RAW_COLUMNS, encoding="utf-8-sig")
        df["split"] = split
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def read_optional_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines()
        if line.strip()
    ]


def label_to_level2(value: object) -> str | None:
    label = str(value).strip().upper()
    match = re.match(r"^(F\d)", label)
    return match.group(1) if match else None


def clean_data(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    report: dict[str, object] = {}
    report["raw_rows"] = int(len(raw_df))
    report["raw_columns"] = RAW_COLUMNS + ["split"]
    report["missing_before"] = raw_df.isna().sum().astype(int).to_dict()

    df = raw_df.copy()
    for column in TEXT_COLUMNS:
        df[column] = df[column].fillna("").astype(str).str.strip()
    df["file_path"] = df["file_path"].fillna("unknown").astype(str).str.strip()
    df["label_raw"] = df["label_raw"].astype(str).str.strip().str.upper()
    df["label"] = df["label_raw"].map(label_to_level2)

    before_label_drop = len(df)
    df = df.dropna(subset=["label"]).copy()
    report["invalid_label_rows_removed"] = int(before_label_drop - len(df))

    before_dedup = len(df)
    df = df.drop_duplicates(
        subset=["label_raw", "title", "author_keywords", "abstract"],
        keep="first",
    ).copy()
    report["duplicate_rows_removed"] = int(before_dedup - len(df))

    df["title_len"] = df["title_tokens"].map(lambda x: len(tokenize(x)))
    df["keyword_len"] = df["author_keywords"].map(lambda x: len(tokenize(x)))
    df["abstract_len"] = df["abstract_tokens"].map(lambda x: len(tokenize(x)))
    df["combined_len"] = df["title_len"] + df["keyword_len"] + df["abstract_len"]

    before_empty_drop = len(df)
    df = df[df["combined_len"] > 0].copy()
    report["empty_text_rows_removed"] = int(before_empty_drop - len(df))

    q1 = float(df["combined_len"].quantile(0.25))
    q3 = float(df["combined_len"].quantile(0.75))
    iqr = q3 - q1
    lower = max(1.0, q1 - 1.5 * iqr)
    upper = q3 + 1.5 * iqr
    outlier_mask = (df["combined_len"] < lower) | (df["combined_len"] > upper)
    report["outlier_rule"] = {
        "field": "combined_len",
        "method": "IQR",
        "lower": round(lower, 3),
        "upper": round(upper, 3),
    }
    report["outlier_rows_removed"] = int(outlier_mask.sum())
    df = df[~outlier_mask].copy()

    report["missing_after"] = df.isna().sum().astype(int).to_dict()
    report["clean_rows"] = int(len(df))
    report["label_counts"] = df["label"].value_counts().sort_index().astype(int).to_dict()
    report["split_counts_after_cleaning"] = df["split"].value_counts().astype(int).to_dict()
    return df.reset_index(drop=True), report


def build_docs(df: pd.DataFrame) -> list[list[str]]:
    docs: list[list[str]] = []
    for _, row in df.iterrows():
        tokens: list[str] = []
        tokens.extend(tokenize(row["title_tokens"]))
        tokens.extend(tokenize(row["author_keywords"]))
        tokens.extend(tokenize(row["abstract_tokens"]))
        docs.append(tokens)
    return docs


def chi2_select_vocabulary(
    docs: list[list[str]],
    labels: list[str],
    min_df: int = 2,
    max_features: int = 12000,
) -> tuple[list[str], pd.DataFrame]:
    n_docs = len(docs)
    label_counts = Counter(labels)
    term_doc_counts: Counter[str] = Counter()
    term_class_counts: dict[str, Counter[str]] = defaultdict(Counter)

    for tokens, label in zip(docs, labels):
        for term in set(tokens):
            term_doc_counts[term] += 1
            term_class_counts[term][label] += 1

    rows = []
    for term, a_plus_c in term_doc_counts.items():
        if a_plus_c < min_df:
            continue
        best_score = 0.0
        best_label = ""
        for label, class_total in label_counts.items():
            a = term_class_counts[term][label]
            b = class_total - a
            c = a_plus_c - a
            d = n_docs - a - b - c
            denom = (a + c) * (b + d) * (a + b) * (c + d)
            score = (n_docs * (a * d - b * c) ** 2 / denom) if denom else 0.0
            if score > best_score:
                best_score = score
                best_label = label
        rows.append((term, best_score, best_label, a_plus_c))

    rows.sort(key=lambda item: (-item[1], -item[3], item[0]))
    selected = rows[:max_features]
    feature_df = pd.DataFrame(
        selected,
        columns=["feature", "chi2_score", "related_label", "document_frequency"],
    )
    return [row[0] for row in selected], feature_df


class TfidfVectorizer:
    def __init__(self, vocabulary: list[str]) -> None:
        self.terms = vocabulary
        self.vocabulary = {term: idx for idx, term in enumerate(vocabulary)}
        self.idf: np.ndarray | None = None

    def fit(self, docs: list[list[str]]) -> "TfidfVectorizer":
        df = np.zeros(len(self.terms), dtype=float)
        for tokens in docs:
            seen = set()
            for token in tokens:
                idx = self.vocabulary.get(token)
                if idx is not None:
                    seen.add(idx)
            for idx in seen:
                df[idx] += 1
        self.idf = np.log((1 + len(docs)) / (1 + df)) + 1.0
        return self

    def transform(self, docs: list[list[str]]) -> list[dict[int, float]]:
        if self.idf is None:
            raise RuntimeError("Vectorizer must be fitted before transform.")
        rows: list[dict[int, float]] = []
        for tokens in docs:
            counts: dict[int, int] = defaultdict(int)
            for token in tokens:
                idx = self.vocabulary.get(token)
                if idx is not None:
                    counts[idx] += 1
            weighted: dict[int, float] = {}
            norm_sq = 0.0
            for idx, count in counts.items():
                value = count * float(self.idf[idx])
                weighted[idx] = value
                norm_sq += value * value
            if norm_sq:
                norm = math.sqrt(norm_sq)
                weighted = {idx: value / norm for idx, value in weighted.items()}
            rows.append(weighted)
        return rows


def sparse_dot(weights: dict[int, float], row: dict[int, float]) -> float:
    if len(weights) < len(row):
        return sum(value * row.get(idx, 0.0) for idx, value in weights.items())
    return sum(value * weights.get(idx, 0.0) for idx, value in row.items())


class MultinomialNB:
    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha
        self.classes: list[str] = []
        self.class_log_prior: dict[str, float] = {}
        self.feature_log_prob: dict[str, dict[int, float]] = {}
        self.default_log_prob: dict[str, float] = {}

    def fit(self, rows: list[dict[int, float]], y: list[str], n_features: int) -> "MultinomialNB":
        class_counts = Counter(y)
        self.classes = sorted(class_counts)
        feature_sum: dict[str, Counter[int]] = {label: Counter() for label in self.classes}
        total_sum: dict[str, float] = {label: 0.0 for label in self.classes}
        for row, label in zip(rows, y):
            for idx, value in row.items():
                feature_sum[label][idx] += value
                total_sum[label] += value
        for label in self.classes:
            self.class_log_prior[label] = math.log(class_counts[label] / len(y))
            denom = total_sum[label] + self.alpha * n_features
            self.default_log_prob[label] = math.log(self.alpha / denom)
            self.feature_log_prob[label] = {
                idx: math.log((value + self.alpha) / denom)
                for idx, value in feature_sum[label].items()
            }
        return self

    def scores_one(self, row: dict[int, float]) -> dict[str, float]:
        scores = {}
        for label in self.classes:
            score = self.class_log_prior[label]
            probs = self.feature_log_prob[label]
            default = self.default_log_prob[label]
            for idx, value in row.items():
                score += value * probs.get(idx, default)
            scores[label] = score
        return scores

    def predict(self, rows: list[dict[int, float]]) -> list[str]:
        return [max(self.scores_one(row).items(), key=lambda item: item[1])[0] for row in rows]

    def predict_scores(self, rows: list[dict[int, float]]) -> list[dict[str, float]]:
        return [self.scores_one(row) for row in rows]


class RocchioClassifier:
    def __init__(self, top_k: int = 0) -> None:
        self.top_k = top_k
        self.classes: list[str] = []
        self.centroids: dict[str, dict[int, float]] = {}

    def fit(self, rows: list[dict[int, float]], y: list[str], n_features: int) -> "RocchioClassifier":
        del n_features
        self.classes = sorted(set(y))
        sums: dict[str, Counter[int]] = {label: Counter() for label in self.classes}
        counts = Counter(y)
        for row, label in zip(rows, y):
            for idx, value in row.items():
                sums[label][idx] += value
        for label in self.classes:
            centroid = {idx: value / counts[label] for idx, value in sums[label].items()}
            if self.top_k > 0 and len(centroid) > self.top_k:
                keep = {
                    idx
                    for idx, _ in sorted(
                        centroid.items(),
                        key=lambda item: abs(item[1]),
                        reverse=True,
                    )[: self.top_k]
                }
                centroid = {idx: value for idx, value in centroid.items() if idx in keep}
            norm = math.sqrt(sum(value * value for value in centroid.values()))
            self.centroids[label] = (
                {idx: value / norm for idx, value in centroid.items()} if norm else centroid
            )
        return self

    def scores_one(self, row: dict[int, float]) -> dict[str, float]:
        return {label: sparse_dot(centroid, row) for label, centroid in self.centroids.items()}

    def predict(self, rows: list[dict[int, float]]) -> list[str]:
        return [max(self.scores_one(row).items(), key=lambda item: item[1])[0] for row in rows]

    def predict_scores(self, rows: list[dict[int, float]]) -> list[dict[str, float]]:
        return [self.scores_one(row) for row in rows]


class SparsePerceptron:
    def __init__(self, epochs: int = 5, learning_rate: float = 1.0) -> None:
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.classes: list[str] = []
        self.weights: dict[str, Counter[int]] = {}

    def fit(self, rows: list[dict[int, float]], y: list[str], n_features: int) -> "SparsePerceptron":
        del n_features
        self.classes = sorted(set(y))
        self.weights = {label: Counter() for label in self.classes}
        order = list(range(len(rows)))
        for epoch in range(self.epochs):
            for i in order:
                row = rows[i]
                truth = y[i]
                pred = self.predict_one(row)
                if pred != truth:
                    for idx, value in row.items():
                        update = self.learning_rate * value
                        self.weights[truth][idx] += update
                        self.weights[pred][idx] -= update
        return self

    def scores_one(self, row: dict[int, float]) -> dict[str, float]:
        return {label: sparse_dot(weights, row) for label, weights in self.weights.items()}

    def predict_one(self, row: dict[int, float]) -> str:
        return max(self.scores_one(row).items(), key=lambda item: item[1])[0]

    def predict(self, rows: list[dict[int, float]]) -> list[str]:
        return [self.predict_one(row) for row in rows]

    def predict_scores(self, rows: list[dict[int, float]]) -> list[dict[str, float]]:
        return [self.scores_one(row) for row in rows]


def confusion_matrix(y_true: list[str], y_pred: list[str], labels: list[str]) -> np.ndarray:
    idx = {label: i for i, label in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)), dtype=int)
    for truth, pred in zip(y_true, y_pred):
        matrix[idx[truth], idx[pred]] += 1
    return matrix


def classification_metrics(
    y_true: list[str],
    y_pred: list[str],
    score_rows: list[dict[str, float]],
    labels: list[str],
) -> dict[str, float]:
    matrix = confusion_matrix(y_true, y_pred, labels)
    precision_scores = []
    recall_scores = []
    f1_scores = []
    for i, _ in enumerate(labels):
        tp = matrix[i, i]
        fp = matrix[:, i].sum() - tp
        fn = matrix[i, :].sum() - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)
    return {
        "accuracy": float(np.trace(matrix) / matrix.sum()) if matrix.sum() else 0.0,
        "macro_precision": float(mean(precision_scores)) if precision_scores else 0.0,
        "macro_recall": float(mean(recall_scores)) if recall_scores else 0.0,
        "macro_f1": float(mean(f1_scores)) if f1_scores else 0.0,
        "macro_auc": multiclass_auc(y_true, score_rows, labels),
    }


def binary_roc_auc(y_binary: list[int], scores: list[float]) -> tuple[float, list[tuple[float, float]]]:
    pairs = sorted(zip(scores, y_binary), key=lambda item: item[0], reverse=True)
    positives = sum(y_binary)
    negatives = len(y_binary) - positives
    if positives == 0 or negatives == 0:
        return 0.0, [(0.0, 0.0), (1.0, 1.0)]
    tp = fp = 0
    points = [(0.0, 0.0)]
    for _, label in pairs:
        if label:
            tp += 1
        else:
            fp += 1
        points.append((fp / negatives, tp / positives))
    auc = 0.0
    last_x, last_y = points[0]
    for x, y in points[1:]:
        auc += (x - last_x) * (y + last_y) / 2
        last_x, last_y = x, y
    return float(auc), points


def multiclass_auc(y_true: list[str], score_rows: list[dict[str, float]], labels: list[str]) -> float:
    aucs = []
    for label in labels:
        y_binary = [1 if truth == label else 0 for truth in y_true]
        scores = [row.get(label, 0.0) for row in score_rows]
        auc, _ = binary_roc_auc(y_binary, scores)
        aucs.append(auc)
    return float(mean(aucs)) if aucs else 0.0


def get_model(name: str, params: dict[str, float | int]):
    if name == "MultinomialNB":
        return MultinomialNB(alpha=float(params["alpha"]))
    if name == "Rocchio":
        return RocchioClassifier(top_k=int(params.get("top_k", 0)))
    if name == "SparsePerceptron":
        return SparsePerceptron(
            epochs=int(params["epochs"]),
            learning_rate=float(params["learning_rate"]),
        )
    raise ValueError(f"Unknown model: {name}")


@dataclass
class RunResult:
    model_name: str
    params: dict[str, float | int]
    val_metrics: dict[str, float]
    test_metrics: dict[str, float]
    val_pred: list[str]
    test_pred: list[str]
    val_scores: list[dict[str, float]]
    test_scores: list[dict[str, float]]
    model: object


def train_and_evaluate(
    model_name: str,
    params: dict[str, float | int],
    x_train: list[dict[int, float]],
    y_train: list[str],
    x_val: list[dict[int, float]],
    y_val: list[str],
    x_test: list[dict[int, float]],
    y_test: list[str],
    labels: list[str],
    n_features: int,
) -> RunResult:
    model = get_model(model_name, params)
    model.fit(x_train, y_train, n_features)
    val_pred = model.predict(x_val)
    test_pred = model.predict(x_test)
    val_scores = model.predict_scores(x_val)
    test_scores = model.predict_scores(x_test)
    return RunResult(
        model_name=model_name,
        params=params,
        val_metrics=classification_metrics(y_val, val_pred, val_scores, labels),
        test_metrics=classification_metrics(y_test, test_pred, test_scores, labels),
        val_pred=val_pred,
        test_pred=test_pred,
        val_scores=val_scores,
        test_scores=test_scores,
        model=model,
    )


def save_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def color_hex(value: float, max_value: float) -> str:
    ratio = 0 if max_value <= 0 else max(0.0, min(1.0, value / max_value))
    r = int(245 - 165 * ratio)
    g = int(247 - 90 * ratio)
    b = int(250 - 60 * ratio)
    return f"#{r:02x}{g:02x}{b:02x}"


def write_svg(path: Path, content: str, width: int = 980, height: int = 620) -> None:
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">'
        '<rect width="100%" height="100%" fill="#ffffff"/>'
        '<style>text{font-family:Arial,"Microsoft YaHei",sans-serif;fill:#1f2937}'
        '.small{font-size:12px}.label{font-size:14px}.title{font-size:22px;font-weight:700}'
        '</style>'
        f"{content}</svg>"
    )
    path.write_text(svg, encoding="utf-8")


def svg_bar_chart(path: Path, title: str, data: dict[str, float], ylabel: str = "数量") -> None:
    width, height = 980, 560
    left, right, top, bottom = 80, 40, 70, 90
    chart_w = width - left - right
    chart_h = height - top - bottom
    labels = list(data.keys())
    values = list(data.values())
    max_value = max(values) if values else 1
    gap = 12
    bar_w = max(12, (chart_w - gap * (len(labels) - 1)) / max(1, len(labels)))
    parts = [f'<text class="title" x="{left}" y="36">{html.escape(title)}</text>']
    parts.append(f'<line x1="{left}" y1="{top+chart_h}" x2="{left+chart_w}" y2="{top+chart_h}" stroke="#374151"/>')
    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+chart_h}" stroke="#374151"/>')
    parts.append(f'<text class="small" x="18" y="{top+10}">{html.escape(ylabel)}</text>')
    for i, (label, value) in enumerate(zip(labels, values)):
        x = left + i * (bar_w + gap)
        h = chart_h * value / max_value if max_value else 0
        y = top + chart_h - h
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" fill="#2563eb"/>')
        parts.append(f'<text class="small" x="{x + bar_w/2:.1f}" y="{y-6:.1f}" text-anchor="middle">{value:.0f}</text>')
        parts.append(f'<text class="small" x="{x + bar_w/2:.1f}" y="{top+chart_h+24}" text-anchor="middle">{html.escape(label)}</text>')
    write_svg(path, "".join(parts), width, height)


def svg_heatmap(path: Path, title: str, matrix: np.ndarray, xlabels: list[str], ylabels: list[str]) -> None:
    width, height = 900, 760
    left, top = 130, 80
    cell = min(56, int((width - left - 80) / max(1, len(xlabels))))
    max_value = float(matrix.max()) if matrix.size else 1.0
    parts = [f'<text class="title" x="{left}" y="36">{html.escape(title)}</text>']
    for i, label in enumerate(xlabels):
        x = left + i * cell + cell / 2
        parts.append(f'<text class="small" x="{x}" y="{top-12}" text-anchor="middle">{html.escape(label)}</text>')
    for i, label in enumerate(ylabels):
        y = top + i * cell + cell / 2 + 4
        parts.append(f'<text class="small" x="{left-12}" y="{y}" text-anchor="end">{html.escape(label)}</text>')
    for r in range(matrix.shape[0]):
        for c in range(matrix.shape[1]):
            value = float(matrix[r, c])
            x = left + c * cell
            y = top + r * cell
            fill = color_hex(value, max_value)
            parts.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="{fill}" stroke="#ffffff"/>')
            parts.append(f'<text class="small" x="{x+cell/2}" y="{y+cell/2+4}" text-anchor="middle">{int(value)}</text>')
    write_svg(path, "".join(parts), width, height)


def svg_correlation_heatmap(path: Path, df: pd.DataFrame) -> None:
    cols = ["title_len", "keyword_len", "abstract_len", "combined_len"]
    corr = df[cols].corr().fillna(0.0).to_numpy()
    width, height = 760, 680
    left, top, cell = 170, 90, 95
    parts = [f'<text class="title" x="{left}" y="38">文本长度相关性热力图</text>']
    for i, label in enumerate(cols):
        parts.append(f'<text class="small" x="{left+i*cell+cell/2}" y="{top-16}" text-anchor="middle">{label}</text>')
        parts.append(f'<text class="small" x="{left-12}" y="{top+i*cell+cell/2+4}" text-anchor="end">{label}</text>')
    for r in range(len(cols)):
        for c in range(len(cols)):
            value = corr[r, c]
            ratio = (value + 1) / 2
            red = int(248 - 120 * ratio)
            blue = int(248 - 170 * (1 - ratio))
            fill = f"#{red:02x}d4{blue:02x}"
            x = left + c * cell
            y = top + r * cell
            parts.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="{fill}" stroke="#ffffff"/>')
            parts.append(f'<text class="label" x="{x+cell/2}" y="{y+cell/2+5}" text-anchor="middle">{value:.2f}</text>')
    write_svg(path, "".join(parts), width, height)


def svg_model_comparison(path: Path, rows: list[dict[str, object]]) -> None:
    data = {str(row["model"]): float(row["test_macro_f1"]) for row in rows}
    svg_bar_chart(path, "模型性能对比（Test Macro-F1）", data, "Macro-F1")


def svg_roc(path: Path, title: str, y_true: list[str], scores: list[dict[str, float]], labels: list[str]) -> None:
    width, height = 760, 620
    left, top, size = 90, 70, 480
    colors = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0891b2", "#4f46e5", "#be123c", "#65a30d"]
    parts = [f'<text class="title" x="{left}" y="36">{html.escape(title)}</text>']
    parts.append(f'<rect x="{left}" y="{top}" width="{size}" height="{size}" fill="#f8fafc" stroke="#374151"/>')
    parts.append(f'<line x1="{left}" y1="{top+size}" x2="{left+size}" y2="{top}" stroke="#9ca3af" stroke-dasharray="4 4"/>')
    for i, label in enumerate(labels[:9]):
        y_binary = [1 if truth == label else 0 for truth in y_true]
        label_scores = [row.get(label, 0.0) for row in scores]
        auc, points = binary_roc_auc(y_binary, label_scores)
        sampled = points[:: max(1, len(points) // 120)]
        if points[-1] not in sampled:
            sampled.append(points[-1])
        coords = " ".join(
            f"{left + fpr * size:.1f},{top + size - tpr * size:.1f}"
            for fpr, tpr in sampled
        )
        color = colors[i % len(colors)]
        parts.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2"/>')
        parts.append(f'<rect x="{left+size+40}" y="{top+i*26}" width="14" height="14" fill="{color}"/>')
        parts.append(f'<text class="small" x="{left+size+62}" y="{top+i*26+12}">{html.escape(label)} AUC={auc:.3f}</text>')
    parts.append(f'<text class="small" x="{left+size/2}" y="{top+size+40}" text-anchor="middle">False Positive Rate</text>')
    parts.append(f'<text class="small" x="{left-48}" y="{top+size/2}" transform="rotate(-90 {left-48},{top+size/2})" text-anchor="middle">True Positive Rate</text>')
    write_svg(path, "".join(parts), width, height)


def extract_feature_importance(best: RunResult, terms: list[str], top_n: int = 20) -> pd.DataFrame:
    scores: Counter[int] = Counter()
    model = best.model
    if isinstance(model, MultinomialNB):
        for idx in range(len(terms)):
            values = [
                model.feature_log_prob[label].get(idx, model.default_log_prob[label])
                for label in model.classes
            ]
            scores[idx] = max(values) - min(values)
    elif isinstance(model, RocchioClassifier):
        for centroid in model.centroids.values():
            for idx, value in centroid.items():
                scores[idx] += abs(value)
    elif isinstance(model, SparsePerceptron):
        for weights in model.weights.values():
            for idx, value in weights.items():
                scores[idx] += abs(value)
    rows = [
        {"feature": terms[idx], "importance": float(score)}
        for idx, score in scores.most_common(top_n)
    ]
    return pd.DataFrame(rows)


def svg_feature_importance(path: Path, feature_df: pd.DataFrame) -> None:
    data = {
        str(row["feature"]): float(row["importance"])
        for _, row in feature_df.head(15).iloc[::-1].iterrows()
    }
    svg_bar_chart(path, "最优模型特征重要性 Top 15", data, "重要性")


def write_report(
    cleaning_report: dict[str, object],
    model_rows: list[dict[str, object]],
    tuning_rows: list[dict[str, object]],
    best_row: dict[str, object],
) -> None:
    lines = [
        "# 机器学习实验课程：图书类目自动标引完整实验",
        "",
        "## 1. 数据处理",
        "",
        f"- 原始数据量：{cleaning_report['raw_rows']} 条。",
        f"- 清洗后数据量：{cleaning_report['clean_rows']} 条。",
        f"- 去重删除：{cleaning_report['duplicate_rows_removed']} 条。",
        f"- 无效标签删除：{cleaning_report['invalid_label_rows_removed']} 条。",
        f"- 空文本删除：{cleaning_report['empty_text_rows_removed']} 条。",
        f"- 异常值处理：按文本总长度 IQR 规则删除 {cleaning_report['outlier_rows_removed']} 条。",
        "- 缺失值处理：文本字段统一填充为空字符串，文件路径缺失填充为 `unknown`，标签缺失或格式异常的记录删除。",
        "- 格式统一：原始中图分类号如 `F270`、`F832` 统一转换为二级类目 `F2`、`F8`。",
        "",
        "## 2. 特征工程与特征选择",
        "",
        "- 特征来源：标题分词、作者关键词、摘要分词。",
        "- 特征工程：构造 TF-IDF 稀疏文本向量，并统计标题长度、关键词长度、摘要长度等 EDA 辅助特征。",
        "- 特征选择：在训练集上使用卡方统计量筛选文本特征，避免测试集信息泄漏。",
        "",
        "## 3. 候选算法与初始超参数",
        "",
        "| 模型 | 初始超参数 | 选择依据 |",
        "| --- | --- | --- |",
        "| MultinomialNB | alpha=1.0 | 适合词频/TF-IDF 文本分类，训练速度快，可解释性强 |",
        "| Rocchio | 无需主要超参数 | 基于类中心的文本分类基线，能衡量类别主题中心相似度 |",
        "| SparsePerceptron | epochs=5, learning_rate=1.0 | 线性判别模型，适合高维稀疏文本特征 |",
        "",
        "## 4. 评估指标",
        "",
        "- Accuracy：衡量整体分类正确率，便于和教材案例表格对照。",
        "- Macro-Precision / Macro-Recall / Macro-F1：各类别等权平均，适合本数据集中类别不均衡的情况。",
        "- Macro-AUC：一对多方式计算多分类 AUC，衡量模型排序区分能力。",
        "",
        "## 5. 模型对比结果",
        "",
        "| 模型 | Val Acc | Val Macro-F1 | Test Acc | Test Macro-F1 | Test Macro-AUC |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in model_rows:
        lines.append(
            f"| {row['model']} | {row['val_accuracy']:.4f} | {row['val_macro_f1']:.4f} | "
            f"{row['test_accuracy']:.4f} | {row['test_macro_f1']:.4f} | {row['test_macro_auc']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## 6. 最优模型调参",
            "",
            f"- 初始验证集 Macro-F1 最优模型：{best_row['model']}。",
            "- 调参策略：围绕最优模型的关键超参数进行网格搜索，以验证集 Macro-F1 选择最优配置。",
            "",
            "| 模型 | 参数 | Val Macro-F1 | Test Macro-F1 |",
            "| --- | --- | ---: | ---: |",
        ]
    )
    for row in tuning_rows:
        lines.append(
            f"| {row['model']} | {row['params']} | {row['val_macro_f1']:.4f} | {row['test_macro_f1']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## 7. 差异分析",
            "",
            "朴素贝叶斯对文本词项的条件概率建模较直接，在关键词较强的分类任务中通常有稳定表现；Rocchio 使用类别中心表示主题，训练很快，但对边界复杂的类别区分能力有限；感知机是线性判别模型，在高维稀疏特征上有较强表达能力，但对类别不均衡和训练轮数较敏感。",
            "",
            "可视化图表与 Streamlit 展示脚本已生成在 `outputs` 文件夹。",
            "",
        ]
    )
    (OUTPUT_DIR / "final_project_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    stopwords = read_optional_lines(DATA_DIR / "stop.txt")
    keyword_list = read_optional_lines(DATA_DIR / "keyword_list.txt")
    raw_df = load_raw_data()
    clean_df, cleaning_report = clean_data(raw_df)
    cleaning_report["stopwords_count"] = len(stopwords)
    cleaning_report["keyword_list_count"] = len(keyword_list)

    clean_df.to_csv(OUTPUT_DIR / "cleaned_dataset.csv", index=False, encoding="utf-8-sig")
    (OUTPUT_DIR / "data_cleaning_report.json").write_text(
        json.dumps(cleaning_report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    train_df = clean_df[clean_df["split"] == "train"].reset_index(drop=True)
    val_df = clean_df[clean_df["split"] == "val"].reset_index(drop=True)
    test_df = clean_df[clean_df["split"] == "test"].reset_index(drop=True)
    y_train = train_df["label"].astype(str).tolist()
    y_val = val_df["label"].astype(str).tolist()
    y_test = test_df["label"].astype(str).tolist()
    labels = sorted(set(y_train) | set(y_val) | set(y_test))

    train_docs = build_docs(train_df)
    val_docs = build_docs(val_df)
    test_docs = build_docs(test_df)
    vocabulary, feature_selection_df = chi2_select_vocabulary(
        train_docs,
        y_train,
        min_df=2,
        max_features=12000,
    )
    feature_selection_df.to_csv(
        OUTPUT_DIR / "feature_selection_chi2.csv",
        index=False,
        encoding="utf-8-sig",
    )

    vectorizer = TfidfVectorizer(vocabulary).fit(train_docs)
    x_train = vectorizer.transform(train_docs)
    x_val = vectorizer.transform(val_docs)
    x_test = vectorizer.transform(test_docs)
    n_features = len(vocabulary)

    initial_configs = [
        ("MultinomialNB", {"alpha": 1.0}),
        ("Rocchio", {"top_k": 0}),
        ("SparsePerceptron", {"epochs": 5, "learning_rate": 1.0}),
    ]
    initial_results = [
        train_and_evaluate(
            name,
            params,
            x_train,
            y_train,
            x_val,
            y_val,
            x_test,
            y_test,
            labels,
            n_features,
        )
        for name, params in initial_configs
    ]
    model_rows = []
    for result in initial_results:
        model_rows.append(
            {
                "model": result.model_name,
                "params": json.dumps(result.params, ensure_ascii=False),
                "val_accuracy": result.val_metrics["accuracy"],
                "val_macro_precision": result.val_metrics["macro_precision"],
                "val_macro_recall": result.val_metrics["macro_recall"],
                "val_macro_f1": result.val_metrics["macro_f1"],
                "val_macro_auc": result.val_metrics["macro_auc"],
                "test_accuracy": result.test_metrics["accuracy"],
                "test_macro_precision": result.test_metrics["macro_precision"],
                "test_macro_recall": result.test_metrics["macro_recall"],
                "test_macro_f1": result.test_metrics["macro_f1"],
                "test_macro_auc": result.test_metrics["macro_auc"],
            }
        )
    save_csv(OUTPUT_DIR / "model_comparison_results.csv", model_rows)

    best_initial = max(initial_results, key=lambda item: item.val_metrics["macro_f1"])
    if best_initial.model_name == "MultinomialNB":
        tuning_configs = [
            ("MultinomialNB", {"alpha": alpha})
            for alpha in [0.05, 0.1, 0.3, 0.5, 1.0, 2.0]
        ]
    elif best_initial.model_name == "SparsePerceptron":
        tuning_configs = [
            ("SparsePerceptron", {"epochs": epochs, "learning_rate": lr})
            for epochs in [3, 5, 8]
            for lr in [0.5, 1.0]
        ]
    else:
        tuning_configs = [
            ("Rocchio", {"top_k": top_k})
            for top_k in [0, 100, 300, 500, 1000, 2000]
        ]

    tuning_results = [
        train_and_evaluate(
            name,
            params,
            x_train,
            y_train,
            x_val,
            y_val,
            x_test,
            y_test,
            labels,
            n_features,
        )
        for name, params in tuning_configs
    ]
    tuning_rows = []
    for result in tuning_results:
        tuning_rows.append(
            {
                "model": result.model_name,
                "params": json.dumps(result.params, ensure_ascii=False),
                "val_accuracy": result.val_metrics["accuracy"],
                "val_macro_f1": result.val_metrics["macro_f1"],
                "val_macro_auc": result.val_metrics["macro_auc"],
                "test_accuracy": result.test_metrics["accuracy"],
                "test_macro_f1": result.test_metrics["macro_f1"],
                "test_macro_auc": result.test_metrics["macro_auc"],
            }
        )
    save_csv(OUTPUT_DIR / "hyperparameter_tuning_results.csv", tuning_rows)

    best_final = max(tuning_results, key=lambda item: item.val_metrics["macro_f1"])
    best_summary = {
        "model": best_final.model_name,
        "params": best_final.params,
        "val_metrics": best_final.val_metrics,
        "test_metrics": best_final.test_metrics,
        "labels": labels,
        "n_features": n_features,
    }
    (OUTPUT_DIR / "best_model_summary.json").write_text(
        json.dumps(best_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    matrix = confusion_matrix(y_test, best_final.test_pred, labels)
    pd.DataFrame(matrix, index=labels, columns=labels).to_csv(
        OUTPUT_DIR / "confusion_matrix_best.csv",
        encoding="utf-8-sig",
    )
    feature_importance_df = extract_feature_importance(best_final, vocabulary, top_n=50)
    feature_importance_df.to_csv(
        OUTPUT_DIR / "feature_importance_best.csv",
        index=False,
        encoding="utf-8-sig",
    )

    svg_bar_chart(
        VIS_DIR / "eda_label_distribution.svg",
        "二级类目分布",
        clean_df["label"].value_counts().sort_index().astype(float).to_dict(),
    )
    missing_before = {
        key: value
        for key, value in cleaning_report["missing_before"].items()
        if int(value) > 0
    }
    svg_bar_chart(VIS_DIR / "eda_missing_values.svg", "缺失值分布（清洗前）", missing_before or {"无缺失": 0})
    svg_correlation_heatmap(VIS_DIR / "eda_correlation_heatmap.svg", clean_df)
    svg_model_comparison(VIS_DIR / "model_comparison.svg", model_rows)
    svg_heatmap(VIS_DIR / "confusion_matrix_best.svg", "最优模型混淆矩阵", matrix, labels, labels)
    svg_roc(VIS_DIR / "roc_curve_best.svg", "最优模型 ROC 曲线（一对多）", y_test, best_final.test_scores, labels)
    svg_feature_importance(VIS_DIR / "feature_importance_best.svg", feature_importance_df)

    best_row = max(model_rows, key=lambda row: row["val_macro_f1"])
    write_report(cleaning_report, model_rows, tuning_rows, best_row)

    print("完整实验已完成")
    print(f"清洗后样本数: {len(clean_df)}")
    print(f"特征数量: {n_features}")
    for row in model_rows:
        print(
            f"{row['model']}: Val F1={row['val_macro_f1']:.4f}, "
            f"Test F1={row['test_macro_f1']:.4f}, Test Acc={row['test_accuracy']:.4f}"
        )
    print(
        f"调参后最优: {best_final.model_name} {best_final.params}, "
        f"Val F1={best_final.val_metrics['macro_f1']:.4f}, "
        f"Test F1={best_final.test_metrics['macro_f1']:.4f}"
    )


if __name__ == "__main__":
    main()
