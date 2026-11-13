"""Run the live 200-prompt benchmark or regenerate its PNG charts."""

import sys
import os
import json
import re
import time
import shutil
import tempfile
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Project root detection: supports running from evals/, project root, or parent workspace
CURRENT_DIR = Path(__file__).resolve().parent
candidates = [
    CURRENT_DIR.parent,
    CURRENT_DIR.parent / "Chronicle-Infinite-Memory-LLM-Agent",
    Path(r"D:\Github\Chronicle-Infinite-Memory-LLM-Agent"),
]
PROJ_ROOT = next((c for c in candidates if (c / "src" / "engine.py").exists()), CURRENT_DIR.parent)
DATA_DIR = PROJ_ROOT / "data"
CHARTS_DIR = PROJ_ROOT / "evals" / "results"
VERIFICATION_RESULTS_PATH = DATA_DIR / "verification_results.json"
INITIAL_MEMORY_STATE = {
    "user": {
        "name": "Arpan Vijay Pethkar",
        "preferences": ["Prefers concise responses"],
        "notes": "Budget is $300/month | Quantitative Trading & Machine Learning",
        "skills": ["Python", "PyTorch", "Rust", "C++", "SQL", "Pine Script"],
        "cloud_budget": "$300/month",
        "preferred_db": "PostgreSQL",
    },
    "persona": {
        "name": "Jarvis",
        "traits": ["helpful", "precise", "proactive", "knowledgeable"],
        "role": "AI assistant with infinite memory capabilities",
    },
    "conversation_context": {"topics_discussed": [], "last_interaction": None},
}


def prepare_benchmark_workspace(workspace: Path) -> tuple[Path, Path]:
    """Create isolated agent state in a temporary OS workspace."""
    from src import tools as tools_module
    original_downloads_dir = tools_module.USER_DOWNLOADS_DIR
    downloads_dir = workspace / "user_downloads"
    downloads_dir.mkdir(parents=True, exist_ok=True)
    source_chroma = DATA_DIR / "chromadb"
    chroma_dir = workspace / "chromadb"
    if source_chroma.exists():
        shutil.copytree(source_chroma, chroma_dir)
    else:
        chroma_dir.mkdir(parents=True)
    tools_module.USER_DOWNLOADS_DIR = downloads_dir
    return chroma_dir, original_downloads_dir


def create_benchmark_agent(chat_id: int, workspace: Path, chroma_dir: Path) -> "MemoryAgent":
    """Create an isolated agent state for one simulated chat session."""
    chat_dir = workspace / f"chat_{chat_id:02d}"
    chat_dir.mkdir(parents=True, exist_ok=True)
    memory_file = chat_dir / "memory.json"
    if not memory_file.exists():
        with memory_file.open("w", encoding="utf-8") as output:
            json.dump(INITIAL_MEMORY_STATE, output, indent=2)
    return MemoryAgent(memory_file=memory_file, chroma_path=chroma_dir)

# Auto-switch to project's .venv if executed with global Python
_venv_python = os.path.join(str(PROJ_ROOT), ".venv", "Scripts", "python.exe")
if (
    __name__ == "__main__"
    and os.path.exists(_venv_python)
    and os.path.normcase(sys.executable) != os.path.normcase(_venv_python)
):
    import subprocess
    sys.exit(subprocess.call([_venv_python, os.path.abspath(__file__)] + sys.argv[1:]))

# Force UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import warnings
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import logging
logging.getLogger("src.engine").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

# Matplotlib setup
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Project imports
sys.path.insert(0, str(PROJ_ROOT))
sys.path.insert(0, str(PROJ_ROOT / "src"))
sys.path.insert(0, str(PROJ_ROOT / "evals"))
from src.engine import MemoryAgent
THEME = {
    "bg": "#0a0c10",
    "card": "#161b22",
    "border": "#30363d",
    "text": "#e6edf3",
    "dim": "#8b949e",
    "blue": "#00d2ff",
    "green": "#00f5a0",
    "purple": "#9d4edd",
    "red": "#ff4d6d",
    "gold": "#ffd166",
}
SEMANTIC_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def _apply_axes_styling(ax):
    ax.set_facecolor(THEME["card"])
    ax.tick_params(colors=THEME["dim"], labelsize=10)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(THEME["border"])


def _save_chart(fig, output_path: str):
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def _show_not_measured(ax, title: str):
    ax.set_facecolor(THEME["card"])
    ax.text(0.5, 0.5, "Not measured in this run", ha="center", va="center",
            color=THEME["dim"], fontsize=13, transform=ax.transAxes)
    ax.set_title(title, color=THEME["text"], fontsize=12, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])


def _normalize_qa_text(text: str) -> List[str]:
    normalized = re.sub(r"[^a-z0-9]", " ", text.lower())
    return [token for token in normalized.split() if token not in {"a", "an", "the"}]


def _token_f1(reference: str, answer: str) -> float:
    reference_tokens = _normalize_qa_text(reference)
    answer_tokens = _normalize_qa_text(answer)
    if not reference_tokens or not answer_tokens:
        return 0.0
    overlap = sum((Counter(reference_tokens) & Counter(answer_tokens)).values())
    if not overlap:
        return 0.0
    precision = overlap / len(answer_tokens)
    recall = overlap / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)


def _binary_classification_metrics(matrix: Dict[str, int]) -> Dict[str, float]:
    tp, tn, fp, fn = (int(matrix.get(key, 0)) for key in ("tp", "tn", "fp", "fn"))

    def ratio(numerator: int, denominator: int) -> float:
        return numerator / denominator if denominator else 0.0

    precision = ratio(tp, tp + fp)
    recall = ratio(tp, tp + fn)
    specificity = ratio(tn, tn + fp)
    f1 = ratio(2 * tp, 2 * tp + fp + fn)
    accuracy = ratio(tp + tn, tp + tn + fp + fn)
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / denominator if denominator else 0.0
    return {
        "precision_pct": round(precision * 100.0, 2),
        "recall_pct": round(recall * 100.0, 2),
        "specificity_pct": round(specificity * 100.0, 2),
        "f1_score_pct": round(f1 * 100.0, 2),
        "accuracy_pct": round(accuracy * 100.0, 2),
        "balanced_accuracy_pct": round((recall + specificity) * 50.0, 2),
        "matthews_correlation_coefficient": round(float(mcc), 4),
        "support": tp + tn + fp + fn,
    }


def score_semantic_matches(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Score saved answers against references using QA F1 and local embeddings."""
    from sentence_transformers import SentenceTransformer

    eligible = [
        result for result in results
        if result.get("cat") != "Tool Routing" and str(result.get("notes", "")).strip()
    ]
    for result in results:
        result["semantic_similarity"] = None
        result["exact_match"] = None
        result["token_f1"] = None

    qa_scores: Dict[str, List[float]] = {"exact_match": [], "token_f1": []}
    category_qa_scores: Dict[str, Dict[str, List[float]]] = {}
    for result in eligible:
        reference = str(result["notes"])
        answer = str(result.get("answer") or "")
        normalized_reference = _normalize_qa_text(reference)
        normalized_answer = _normalize_qa_text(answer)
        exact_match = float(normalized_reference == normalized_answer and bool(normalized_reference))
        token_f1 = _token_f1(reference, answer)
        result["exact_match"] = exact_match
        result["token_f1"] = round(token_f1, 4)
        qa_scores["exact_match"].append(exact_match)
        qa_scores["token_f1"].append(token_f1)
        category_metrics = category_qa_scores.setdefault(
            str(result["cat"]), {"exact_match": [], "token_f1": []}
        )
        category_metrics["exact_match"].append(exact_match)
        category_metrics["token_f1"].append(token_f1)

    populated = [
        result for result in eligible
        if str(result.get("answer", "")).strip()
    ]
    scores_by_id: Dict[int, float] = {}
    if populated:
        model = SentenceTransformer(SEMANTIC_MODEL_NAME, local_files_only=True)
        texts = [
            text
            for result in populated
            for text in (str(result["notes"]), str(result["answer"]))
        ]
        embeddings = model.encode(
            texts,
            batch_size=32,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        for index, result in enumerate(populated):
            scores_by_id[id(result)] = float(embeddings[index * 2] @ embeddings[index * 2 + 1])

    category_similarity: Dict[str, List[float]] = {}
    for result in populated:
        score = scores_by_id[id(result)]
        result["semantic_similarity"] = round(score, 4)
        category_similarity.setdefault(str(result["cat"]), []).append(score)

    all_scores = [score for values in category_similarity.values() for score in values]
    unanswered = len(eligible) - len(populated)
    return {
        "model": SEMANTIC_MODEL_NAME,
        "method": "normalized exact match, token F1, and answer/reference cosine similarity",
        "eligible_prompts": len(eligible),
        "scored_prompts": len(populated),
        "unanswered_prompts": unanswered,
        "overall_mean": round(float(np.mean(all_scores)), 4) if all_scores else None,
        "qa_reference_metrics": {
            "exact_match_pct": round(float(np.mean(qa_scores["exact_match"])) * 100.0, 2)
            if qa_scores["exact_match"] else None,
            "token_f1_pct": round(float(np.mean(qa_scores["token_f1"])) * 100.0, 2)
            if qa_scores["token_f1"] else None,
        },
        "qa_reference_metrics_by_category": {
            category: {
                "count": len(values["token_f1"]),
                "exact_match_pct": round(float(np.mean(values["exact_match"])) * 100.0, 2),
                "token_f1_pct": round(float(np.mean(values["token_f1"])) * 100.0, 2),
            }
            for category, values in sorted(category_qa_scores.items())
        },
        "semantic_similarity_by_category": {
            category: {
                "count": len(scores),
                "mean": round(float(np.mean(scores)), 4),
            }
            for category, scores in sorted(category_similarity.items())
        },
    }


def generate_rag_quality_chart(metrics: Dict[str, Any], output_path: str):
    fig, ax = plt.subplots(figsize=(10, 6), facecolor=THEME["bg"])
    _apply_axes_styling(ax)
    supplemental = metrics.get("supplemental_metrics", {})
    rag_quality = supplemental.get("rag_quality", {})
    if not isinstance(rag_quality, dict):
        rag_quality = {}
    scores = {
        label: float(value)
        for label, value in rag_quality.items()
        if isinstance(value, (int, float))
    }
    hallucination_rate = supplemental.get("hallucination_rate_pct")
    if isinstance(hallucination_rate, (int, float)):
        scores["Hallucination Rate"] = float(hallucination_rate)

    if not scores:
        _show_not_measured(ax, "RAG quality metrics")
        _save_chart(fig, output_path)
        return

    labels = list(scores)
    values = list(scores.values())
    colors = [
        THEME["red"] if "hallucination" in label.lower()
        else THEME["green"] if label == "Faithfulness"
        else THEME["blue"] if "Recall" in label
        else THEME["purple"] if "Precision" in label
        else THEME["gold"]
        for label in labels
    ]
    bars = ax.barh(labels, values, color=colors, edgecolor=THEME["border"])
    ax.set_xlim(0, max(110.0, max(values) * 1.15))
    ax.set_xlabel("Reported score (%)", color=THEME["text"])
    ax.set_title(
        "RAG Quality and Hallucination Rate",
        color=THEME["blue"],
        fontweight="bold",
    )
    ax.tick_params(axis="y", labelcolor=THEME["dim"])
    ax.grid(axis="x", linestyle="--", alpha=0.25, color=THEME["dim"])
    for bar, value in zip(bars, values):
        ax.text(
            value + 1,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.1f}%",
            va="center",
            color=THEME["text"],
            fontweight="bold",
        )
    fig.tight_layout()
    _save_chart(fig, output_path)


def generate_semantic_similarity_chart(metrics: Dict[str, Any], output_path: str):
    fig, ax = plt.subplots(figsize=(10, 6), facecolor=THEME["bg"])
    _apply_axes_styling(ax)
    semantic_matching = metrics.get("semantic_matching", {})
    by_category = semantic_matching.get("semantic_similarity_by_category", {})
    if not by_category:
        _show_not_measured(ax, "Semantic answer/reference similarity")
        _save_chart(fig, output_path)
        return

    categories = list(by_category)
    values = [float(by_category[category]["mean"]) for category in categories]
    counts = [int(by_category[category]["count"]) for category in categories]
    bars = ax.barh(categories, values, color=THEME["blue"])
    ax.axvline(0, color=THEME["dim"], linewidth=0.8)
    ax.set_xlim(-1, 1.15)
    ax.set_xlabel("Cosine similarity (tool-routing excluded)", color=THEME["text"])
    ax.set_title(
        "Answer/Reference Semantic Similarity by Category",
        color=THEME["blue"],
        fontweight="bold",
    )
    ax.tick_params(axis="y", labelcolor=THEME["text"])
    for bar, value, count in zip(bars, values, counts):
        ax.text(
            value + 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f} (n={count})",
            ha="left",
            va="center",
            color=THEME["text"],
        )
    ax.text(
        0.01,
        -0.14,
        f"Scored prompts: {semantic_matching.get('scored_prompts', 'unknown')}",
        transform=ax.transAxes,
        color=THEME["dim"],
        fontsize=8,
        va="top",
    )
    fig.tight_layout()
    _save_chart(fig, output_path)


def generate_confusion_matrix_chart(metrics: Dict[str, Any], output_path: str):
    fig, ax = plt.subplots(figsize=(10, 8), facecolor=THEME["bg"])
    raw = metrics.get("cm_raw", {})
    counts = np.array([[int(raw.get("tp", 0)), int(raw.get("fn", 0))],
                       [int(raw.get("fp", 0)), int(raw.get("tn", 0))]])
    row_totals = counts.sum(axis=1)
    image = ax.imshow(counts, cmap="Blues", vmin=0, vmax=max(1, int(counts.max())), aspect="auto")
    ax.set_title(
        "7. Tool Selection and Schema-Routing Confusion Matrix",
        color=THEME["green"], fontweight="bold", pad=16,
    )
    ax.set_xticks([0, 1], labels=["Predicted: Tool", "Predicted: No Tool"], color=THEME["text"])
    ax.set_yticks(
        [0, 1],
        labels=[f"Actual: Tool Required (n={row_totals[0]})",
                f"Actual: No Tool Required (n={row_totals[1]})"],
        color=THEME["text"],
    )
    ax.set_xlabel("Observed agent action (predicted class)", color=THEME["text"], labelpad=12)
    ax.set_ylabel("Prompt-intent rubric (actual class)", color=THEME["text"], labelpad=12)
    tags = (("TP", "FN"), ("FP", "TN"))
    for row in range(2):
        for col in range(2):
            rgba = image.cmap(image.norm(counts[row, col]))
            luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
            color = "#101820" if luminance > 0.52 else "#ffffff"
            ax.text(col, row,
                    f"{tags[row][col]}\nCount: {counts[row, col]}",
                    ha="center", va="center", color=color, fontweight="bold", fontsize=11)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Observed prompt count", color=THEME["text"])
    colorbar.ax.tick_params(colors=THEME["dim"])

    stats = _binary_classification_metrics(raw)
    metrics_text = (
        f"n={stats['support']}  |  Precision {stats['precision_pct']:.1f}%  |  "
        f"Recall {stats['recall_pct']:.1f}%  |  Specificity {stats['specificity_pct']:.1f}%  |  "
        f"F1 {stats['f1_score_pct']:.1f}%  |  Accuracy {stats['accuracy_pct']:.1f}%  |  "
        f"Balanced accuracy {stats['balanced_accuracy_pct']:.1f}%  |  MCC {stats['matthews_correlation_coefficient']:.3f}"
    )
    fig.text(0.5, 0.035, metrics_text, ha="center", va="center",
             color=THEME["text"], fontsize=9, wrap=True)
    fig.subplots_adjust(left=0.24, right=0.9, top=0.84, bottom=0.22)
    _save_chart(fig, output_path)


def generate_latency_chart(metrics: Dict[str, Any], output_path: str):
    fig, ax = plt.subplots(figsize=(9, 5.5), facecolor=THEME["bg"])
    _apply_axes_styling(ax)
    latency_keys = (
        "benchmark_average_latency_ms",
        "benchmark_median_latency_ms",
        "benchmark_p95_latency_ms",
    )
    if all(key in metrics for key in latency_keys):
        labels = ["Mean response", "Median response", "95th percentile"]
        values = [float(metrics[key]) for key in latency_keys]
        title = "12. End-to-End Prompt Latency"
    else:
        labels, values, title = [], [], ""
    if not values:
        _show_not_measured(ax, "12. End-to-End Prompt Latency")
    else:
        bars = ax.barh(labels, values, color=THEME["blue"])
        maximum = max(values)
        ax.set_xlim(0, maximum * 1.3 if maximum else 1)
        ax.set_xlabel("Latency (ms)", color=THEME["text"])
        ax.set_title(title, color=THEME["blue"], fontweight="bold")
        for bar, value in zip(bars, values):
            ax.text(value + maximum * 0.02 if maximum else 0.02,
                    bar.get_y() + bar.get_height() / 2, f"{value:.2f} ms",
                    va="center", color=THEME["text"])
        ax.text(
            0.01, -0.16, "Resource footprint: not present in the verification JSON.",
            transform=ax.transAxes, color=THEME["gold"], fontsize=8, va="top",
        )
    _save_chart(fig, output_path)


def _get_architectural_snapshot(metrics: Dict[str, Any]) -> tuple[Dict[str, Any], Dict[str, Any]]:
    supplemental = metrics.get("supplemental_metrics", {})
    if not isinstance(supplemental, dict):
        return {}, {}
    snapshot = supplemental.get("architectural_verification", {})
    if not isinstance(snapshot, dict):
        return {}, {}
    snapshot_metrics = snapshot.get("metrics", {})
    if not isinstance(snapshot_metrics, dict):
        return snapshot, {}
    return snapshot, snapshot_metrics


def _add_architectural_snapshot_footer(fig, snapshot: Dict[str, Any], note: str) -> None:
    source = snapshot.get("source_file", "source file not recorded")
    timestamp = snapshot.get("timestamp", "date unavailable")
    fig.text(
        0.01,
        0.015,
        f"Archived snapshot: {timestamp} | {source} | {note}",
        color=THEME["dim"],
        fontsize=7,
        wrap=True,
    )


def generate_category_latency_chart(
    output_path: str,
    prompt_records: List[Dict[str, Any]],
):
    fig, ax = plt.subplots(figsize=(11, 6), facecolor=THEME["bg"])
    _apply_axes_styling(ax)
    latencies_by_category: Dict[str, List[float]] = {}
    for record in prompt_records:
        category = record.get("cat")
        duration = record.get("duration_s")
        if (
            isinstance(category, str)
            and isinstance(duration, (int, float))
            and np.isfinite(duration)
            and duration >= 0
        ):
            latencies_by_category.setdefault(category, []).append(float(duration))

    if not latencies_by_category:
        _show_not_measured(ax, "Per-category response latency")
        _save_chart(fig, output_path)
        return

    ordered = sorted(
        latencies_by_category.items(),
        key=lambda item: float(np.median(item[1])),
    )
    labels = [f"{category} (n={len(values)})" for category, values in ordered]
    data = [values for _, values in ordered]
    boxplot = ax.boxplot(
        data,
        vert=False,
        patch_artist=True,
        showfliers=True,
        medianprops={"color": THEME["text"], "linewidth": 1.8},
        whiskerprops={"color": THEME["dim"]},
        capprops={"color": THEME["dim"]},
        flierprops={
            "marker": "o",
            "markersize": 3,
            "markerfacecolor": THEME["gold"],
            "markeredgecolor": THEME["gold"],
            "alpha": 0.7,
        },
    )
    for box in boxplot["boxes"]:
        box.set_facecolor(THEME["blue"])
        box.set_edgecolor(THEME["border"])
    ax.set_yticks(np.arange(1, len(labels) + 1), labels=labels, color=THEME["text"])
    all_values = [value for values in data for value in values]
    overall_median = float(np.median(all_values))
    ax.axvline(
        overall_median,
        color=THEME["gold"],
        linestyle="--",
        linewidth=1.2,
        label=f"Overall median: {overall_median:.2f}s",
    )
    ax.set_xlabel("Response latency (seconds)", color=THEME["text"])
    ax.set_title(
        "Per-Prompt Response Latency by Category",
        color=THEME["blue"],
        fontweight="bold",
    )
    ax.legend(facecolor=THEME["card"], edgecolor=THEME["border"], labelcolor=THEME["text"])
    ax.grid(axis="x", linestyle="--", alpha=0.25, color=THEME["dim"])
    fig.tight_layout()
    _save_chart(fig, output_path)


def generate_subsystem_latency_chart(metrics: Dict[str, Any], output_path: str):
    snapshot, archived = _get_architectural_snapshot(metrics)
    definitions = (
        ("Ollama inference", "ollama_inference_latency_ms"),
        ("Vector retrieval", "retrieval_latency_ms"),
        ("Atomic memory write", "atomic_write_time_ms"),
        ("AST evaluation", "ast_latency_ms"),
        ("Prompt assembly", "prompt_assembly_ms"),
        ("JSON parser", "parser_latency_ms"),
    )
    rows = [
        (label, float(archived[key]))
        for label, key in definitions
        if isinstance(archived.get(key), (int, float))
        and np.isfinite(archived[key])
        and archived[key] > 0
    ]
    fig, ax = plt.subplots(figsize=(10, 6), facecolor=THEME["bg"])
    _apply_axes_styling(ax)
    if not rows:
        _show_not_measured(ax, "Subsystem latency")
        _save_chart(fig, output_path)
        return

    rows.sort(key=lambda row: row[1])
    labels = [row[0] for row in rows]
    values = [row[1] for row in rows]
    colors = [
        THEME["gold"] if label == "Ollama inference" else THEME["blue"]
        for label in labels
    ]
    bars = ax.barh(labels, values, color=colors)
    ax.set_xscale("log")
    ax.set_xlim(min(values) * 0.5, max(values) * 2)
    ax.set_xlabel("Measured latency (ms, logarithmic scale)", color=THEME["text"])
    ax.set_title("Subsystem Latency Breakdown", color=THEME["blue"], fontweight="bold")
    ax.tick_params(axis="y", labelcolor=THEME["text"])
    ax.grid(axis="x", which="both", linestyle="--", alpha=0.25, color=THEME["dim"])
    for bar, value in zip(bars, values):
        ax.text(
            value * 1.08,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.4g} ms",
            va="center",
            color=THEME["text"],
            fontsize=8,
        )
    _add_architectural_snapshot_footer(
        fig, snapshot, "Component values span different runtime scales; the x-axis is logarithmic."
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    _save_chart(fig, output_path)


def generate_benchmark_category_chart(metrics: Dict[str, Any], output_path: str):
    fig, ax = plt.subplots(figsize=(11, 6), facecolor=THEME["bg"])
    _apply_axes_styling(ax)
    summary = metrics.get("benchmark_200_category_summary", {})
    if not summary:
        _show_not_measured(ax, "7. Benchmark Category Scores")
        _save_chart(fig, output_path)
        return
    categories = list(summary)
    passes = [int(summary[name]["pass"]) for name in categories]
    partials = [int(summary[name]["partial"]) for name in categories]
    fails = [int(summary[name]["fail"]) for name in categories]
    totals = [int(summary[name]["total"]) for name in categories]
    y = np.arange(len(categories))
    ax.barh(y, passes, color=THEME["green"], label="Pass")
    ax.barh(y, partials, left=passes, color=THEME["gold"], label="Partial")
    ax.barh(y, fails, left=np.array(passes) + np.array(partials), color=THEME["red"], label="Fail")
    ax.set_yticks(y, labels=categories, color=THEME["text"])
    ax.set_xlabel("Prompt count", color=THEME["text"])
    max_total = max(totals, default=1)
    ax.set_xlim(0, max_total * 1.25)
    total_prompts = metrics.get("benchmark_200_total_prompts")
    overall_score = metrics.get("benchmark_200_overall_score_pct")
    total_label = f"{total_prompts} prompts" if total_prompts is not None else "prompt count unavailable"
    score_label = f"{overall_score:.1f}% overall" if overall_score is not None else "overall score unavailable"
    ax.set_title(
        f"Prompt Category Results — {total_label}, {score_label}",
        color=THEME["gold"], fontweight="bold")
    ax.legend(facecolor=THEME["card"], edgecolor=THEME["border"], labelcolor=THEME["text"])
    ax.grid(axis="x", linestyle="--", alpha=0.25, color=THEME["dim"])
    for index, (total, values) in enumerate(zip(totals, summary.values())):
        ax.text(total + max_total * 0.025, index, f"{values['score_pct']:.1f}% (n={total})",
                va="center", color=THEME["text"], fontweight="bold")
    fig = ax.figure
    fig.subplots_adjust(left=0.2, right=0.94, top=0.84, bottom=0.16)
    _save_chart(fig, output_path)


def generate_overall_verdict_chart(metrics: Dict[str, Any], output_path: str):
    fig, ax = plt.subplots(figsize=(8, 5.5), facecolor=THEME["bg"])
    _apply_axes_styling(ax)
    verdicts = metrics.get("benchmark_verdict_counts", {})
    labels = ["Pass", "Partial", "Fail"]
    keys = ["pass", "partial", "fail"]
    values = [int(verdicts.get(key, 0)) for key in keys]
    if not any(values):
        _show_not_measured(ax, "Overall benchmark verdicts")
        _save_chart(fig, output_path)
        return

    bars = ax.bar(labels, values, color=[THEME["green"], THEME["gold"], THEME["red"]])
    ax.set_ylabel("Prompt count", color=THEME["text"])
    ax.set_title("Overall Benchmark Verdict Counts", color=THEME["blue"], fontweight="bold")
    ax.tick_params(axis="x", labelcolor=THEME["text"])
    ax.grid(axis="y", linestyle="--", alpha=0.25, color=THEME["dim"])
    for bar, value in zip(bars, values):
        ax.annotate(
            str(value),
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            color=THEME["text"],
            fontweight="bold",
        )
    score = metrics.get("benchmark_200_overall_score_pct")
    total = metrics.get("benchmark_200_total_prompts")
    if isinstance(score, (int, float)):
        ax.text(
            0.99,
            0.96,
            f"Score {score:.1f}% | {total or sum(values)} prompts",
            transform=ax.transAxes,
            ha="right",
            va="top",
            color=THEME["dim"],
        )
    fig.tight_layout()
    _save_chart(fig, output_path)


def generate_charts(
    metrics: Dict[str, Any],
    out_dir: Path,
    prompt_records: Optional[List[Dict[str, Any]]] = None,
):
    """Generate separate PNG charts using only metrics present in the result JSON."""
    out_dir.mkdir(parents=True, exist_ok=True)
    
    chart_generators = [
        ("01_rag_quality_scores.png", generate_rag_quality_chart),
        ("02_tool_confusion_matrix.png", generate_confusion_matrix_chart),
        ("03_end_to_end_latency.png", generate_latency_chart),
        ("04_prompt_category_results.png", generate_benchmark_category_chart),
        ("05_semantic_similarity_by_category.png", generate_semantic_similarity_chart),
        ("06_overall_verdict_counts.png", generate_overall_verdict_chart),
        (
            "07_category_response_latency.png",
            lambda data, target: generate_category_latency_chart(
                target,
                prompt_records or data.get("_benchmark_prompt_records", []),
            ),
        ),
        ("08_subsystem_latency_breakdown.png", generate_subsystem_latency_chart),
    ]
    
    for filename, fn in chart_generators:
        target = str(out_dir / filename)
        fn(metrics, target)
        print(f"Saved chart: {target}")

    superseded_filenames = (
        "13_end_to_end_latency.png",
        "14_prompt_category_results.png",
        "15_semantic_similarity_by_category.png",
        "16_overall_verdict_counts.png",
        "09_security_control_results.png",
        "10_memory_retention_curve.png",
        "11_subsystem_latency_breakdown.png",
        "12_vector_retrieval_effectiveness.png",
        "13_deduplication_efficiency.png",
        "08_parser_recovery_breakdown.png",
        "09_subsystem_latency_breakdown.png",
    )
    for filename in superseded_filenames:
        superseded_path = out_dir / filename
        if superseded_path.exists():
            superseded_path.unlink()
            print(f"Removed superseded chart: {superseded_path}")

def summarize_live_results(results: List[Dict[str, Any]], chat_counts: List[int]) -> Dict[str, Any]:
    """Derive every benchmark metric from completed prompt responses."""
    if not results:
        raise ValueError("Cannot summarize an empty benchmark.")

    category_counts: Dict[str, Dict[str, int]] = {}
    for result in results:
        category = result["cat"]
        verdict = result["verdict"]
        counts = category_counts.setdefault(category, {"total": 0, "pass": 0, "partial": 0, "fail": 0})
        counts["total"] += 1
        counts[verdict] += 1
    category_summary = {
        category: {
            **counts,
            "score_pct": round((counts["pass"] + 0.5 * counts["partial"]) / counts["total"] * 100.0, 1),
        }
        for category, counts in category_counts.items()
    }

    verdict_counts = {
        verdict: sum(result["verdict"] == verdict for result in results)
        for verdict in ("pass", "partial", "fail")
    }
    score = (verdict_counts["pass"] + 0.5 * verdict_counts["partial"]) / len(results) * 100.0
    matrix = {key: 0 for key in ("tp", "tn", "fp", "fn")}
    tool_required = 0
    exact_tool_matches = 0
    for result in results:
        if result["cat"] == "Tool Routing":
            for key in matrix:
                matrix[key] += int(result.get("tool_confusion", {}).get(key, False))
            expected_tool = result.get("tool_confusion", {}).get("expected_tool")
            if expected_tool:
                tool_required += 1
                if any(call.get("tool") == expected_tool for call in result.get("tool_calls", [])):
                    exact_tool_matches += 1

    routing_total = sum(matrix.values())
    routing_metrics = _binary_classification_metrics(matrix)
    latencies = [max(0.0, float(result.get("duration_s", 0.0))) * 1000.0 for result in results]
    sorted_latencies = sorted(latencies)
    p95_index = max(0, int(np.ceil(0.95 * len(sorted_latencies))) - 1)
    total_chats = len(chat_counts)

    return {
        "source": "live_ollama",
        "timestamp": datetime.now().isoformat(),
        "total_prompts": len(results),
        "total_chats": total_chats,
        "prompts_per_chat_range": "0-6 (stratified random distribution across 7 categories)",
        "chat_distribution_summary": {
            "chats_with_0_prompts": sum(count == 0 for count in chat_counts),
            "chats_with_1_to_3_prompts": sum(1 <= count <= 3 for count in chat_counts),
            "chats_with_4_to_6_prompts": sum(4 <= count <= 6 for count in chat_counts),
        },
        "overall_score_pct": round(score, 1),
        "total_pass": verdict_counts["pass"],
        "total_partial": verdict_counts["partial"],
        "total_fail": verdict_counts["fail"],
        "category_summary": category_summary,
        "tool_matrix": {
            **routing_metrics,
            "true_positives": matrix["tp"],
            "true_negatives": matrix["tn"],
            "false_positives": matrix["fp"],
            "false_negatives": matrix["fn"],
            "sample_count": routing_total,
            "tool_required_prompts": tool_required,
            "exact_tool_matches": exact_tool_matches,
            "tool_selection_accuracy_pct": round(exact_tool_matches / tool_required * 100.0, 1) if tool_required else 0.0,
        },
        "mean_latency_ms": round(float(np.mean(latencies)), 2),
        "median_latency_ms": round(float(np.median(latencies)), 2),
        "p95_latency_ms": round(sorted_latencies[p95_index], 2),
        "total_model_response_seconds": round(sum(latencies) / 1000.0, 2),
    }


def metrics_from_live_summary(summary: Dict[str, Any]) -> Dict[str, Any]:
    matrix = summary["tool_matrix"]
    raw_matrix = {
        "tp": int(matrix["true_positives"]),
        "tn": int(matrix["true_negatives"]),
        "fp": int(matrix["false_positives"]),
        "fn": int(matrix["false_negatives"]),
    }
    classification_metrics = _binary_classification_metrics(raw_matrix)
    metrics = {
        "benchmark_200_overall_score_pct": summary["overall_score_pct"],
        "benchmark_200_category_summary": summary["category_summary"],
        "benchmark_200_total_prompts": summary["total_prompts"],
        "benchmark_200_total_chats": summary["total_chats"],
        "tool_routing_classification": classification_metrics,
        "benchmark_200_tool_precision_pct": classification_metrics["precision_pct"],
        "benchmark_200_tool_recall_pct": classification_metrics["recall_pct"],
        "benchmark_200_tool_specificity_pct": classification_metrics["specificity_pct"],
        "benchmark_200_tool_f1_pct": classification_metrics["f1_score_pct"],
        "benchmark_200_tool_accuracy_pct": classification_metrics["accuracy_pct"],
        "benchmark_200_tool_balanced_accuracy_pct": classification_metrics["balanced_accuracy_pct"],
        "benchmark_200_tool_mcc": classification_metrics["matthews_correlation_coefficient"],
        "cm_raw": raw_matrix,
        "benchmark_verdict_counts": {
            "pass": summary["total_pass"],
            "partial": summary["total_partial"],
            "fail": summary["total_fail"],
        },
    }
    if "tool_selection_accuracy_pct" in matrix:
        metrics["benchmark_200_tool_selection_accuracy_pct"] = matrix["tool_selection_accuracy_pct"]
    if all(key in summary for key in ("mean_latency_ms", "median_latency_ms", "p95_latency_ms")):
        metrics.update({
            "benchmark_average_latency_ms": summary["mean_latency_ms"],
            "benchmark_median_latency_ms": summary["median_latency_ms"],
            "benchmark_p95_latency_ms": summary["p95_latency_ms"],
        })
    if "semantic_matching" in summary:
        metrics["semantic_matching"] = summary["semantic_matching"]
    for key in ("faithfulness_pct", "hallucination_count"):
        if key in summary:
            metrics[key] = summary[key]
    if "supplemental_metrics" in summary:
        metrics["supplemental_metrics"] = summary["supplemental_metrics"]
        metrics["supplemental_metrics_status"] = summary.get(
            "supplemental_metrics_status", "archived_snapshot"
        )
    return metrics


def metrics_from_saved_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("Verification results do not contain a metrics object.")
    normalized = dict(metrics)
    summary = payload.get("summary", {})
    if isinstance(summary, dict):
        normalized.setdefault("benchmark_200_overall_score_pct", summary.get("overall_score_pct"))
        normalized.setdefault("benchmark_200_category_summary", summary.get("category_summary"))
        normalized.setdefault("benchmark_200_total_prompts", summary.get("total_prompts"))
        normalized.setdefault("benchmark_200_total_chats", summary.get("total_chats"))
        normalized.setdefault("benchmark_average_latency_ms", summary.get("mean_latency_ms"))
        normalized.setdefault("benchmark_median_latency_ms", summary.get("median_latency_ms"))
        normalized.setdefault("benchmark_p95_latency_ms", summary.get("p95_latency_ms"))
        normalized.setdefault("semantic_matching", summary.get("semantic_matching"))
        normalized.setdefault("benchmark_verdict_counts", {
            "pass": summary.get("total_pass", 0),
            "partial": summary.get("total_partial", 0),
            "fail": summary.get("total_fail", 0),
        })
        supplemental_metrics = summary.get("supplemental_metrics")
        if isinstance(supplemental_metrics, dict):
            normalized["supplemental_metrics"] = supplemental_metrics
            normalized["supplemental_metrics_status"] = summary.get(
                "supplemental_metrics_status", "archived_snapshot"
            )
        tool_matrix = summary.get("tool_matrix")
        if isinstance(tool_matrix, dict):
            normalized.setdefault("cm_raw", {
                "tp": tool_matrix.get("true_positives", 0),
                "tn": tool_matrix.get("true_negatives", 0),
                "fp": tool_matrix.get("false_positives", 0),
                "fn": tool_matrix.get("false_negatives", 0),
            })
    prompt_records = payload.get("prompts", [])
    normalized["_benchmark_prompt_records"] = (
        prompt_records if isinstance(prompt_records, list) else []
    )
    return normalized


def load_archived_report() -> Dict[str, Any]:
    if not VERIFICATION_RESULTS_PATH.is_file():
        raise FileNotFoundError(
            f"Verification results not found: {VERIFICATION_RESULTS_PATH}"
        )

    payload = json.loads(VERIFICATION_RESULTS_PATH.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(
            f"Verification results must contain a JSON object: {VERIFICATION_RESULTS_PATH}"
        )
    summary = payload.get("summary", {})
    supplemental = summary.get("supplemental_metrics", {}) if isinstance(summary, dict) else {}
    if not isinstance(supplemental, dict):
        raise ValueError(
            "Canonical verification results contain invalid supplemental_metrics."
        )

    return {
        "source_file": "data/verification_results.json#summary.supplemental_metrics",
        "generated_at": supplemental.get("generated_at"),
        "method": supplemental.get("method"),
        "overall": {
            "hallucination_rate_pct": supplemental.get("hallucination_rate_pct")
        },
        "rag_quality": supplemental.get("rag_quality", {}),
        "invariant_tests": supplemental.get("invariant_tests", {}),
        "hallucinations": supplemental.get("hallucinations", []),
        "architectural_verification": supplemental.get("architectural_verification"),
    }


def add_archived_summary_fields(
    summary: Dict[str, Any], archived_report: Optional[Dict[str, Any]] = None
) -> None:
    """Add supplemental archived measurements to the existing benchmark summary."""
    archived_report = archived_report or {}
    archived_categories = archived_report.get("categories", {})
    archived_rag_quality = archived_report.get("rag_quality", {})
    archived_hallucinations = archived_report.get("hallucinations", [])
    if not isinstance(archived_hallucinations, list):
        archived_hallucinations = []

    for category, counts in summary["category_summary"].items():
        archived_category = archived_categories.get(category, {})
        if "hallucinations" in archived_category:
            counts["hallucinations"] = archived_category["hallucinations"]

    if archived_report:
        supplemental_metrics = {
            "source": archived_report.get("source_file", "canonical verification results"),
            "generated_at": archived_report.get("generated_at"),
            "method": archived_report.get("method"),
            "hallucination_rate_pct": archived_report.get("overall", {}).get(
                "hallucination_rate_pct"
            ),
            "rag_quality": archived_rag_quality,
            "invariant_tests": archived_report.get("invariant_tests", {}),
            "hallucinations": archived_hallucinations,
        }
        architectural_verification = archived_report.get("architectural_verification")
        if isinstance(architectural_verification, dict):
            supplemental_metrics["architectural_verification"] = architectural_verification
        summary["supplemental_metrics"] = supplemental_metrics
        summary["supplemental_metrics_status"] = "archived_snapshot"


def expected_routing_tool(prompt_text: str) -> Optional[str]:
    text = prompt_text.lower()
    if "archival memory" in text or "knowledge base" in text:
        return "search_knowledge_base"
    if "update_core_memory" in text or "update core memory" in text:
        return "update_core_memory"
    if "teacher mode" in text or "activate teacher" in text:
        return "activate_teacher_mode"
    if "guideline" in text:
        return "fetch_coding_guidelines"
    if "pdf" in text or "save my project" in text:
        return "save_summary_to_pdf"
    if "web search" in text or "search the web" in text or "search for" in text:
        return "perform_web_search"
    if "math tool" in text or "calculate " in text:
        return "evaluate_math_expression"
    if "system time" in text or "host system" in text or "currently" in text and "year" in text:
        return "get_current_datetime"
    if "current time" in text or "date and time" in text or "what year" in text:
        return "get_current_datetime"
    return None


def evaluate_response(prompt_item: Dict[str, Any], agent_resp: Dict[str, Any]) -> tuple[str, str, Dict[str, Any]]:
    """Apply the benchmark's explicit per-category heuristic rubric."""
    cat = prompt_item.get("cat", "")
    notes = prompt_item.get("notes", "").lower()
    p_text = prompt_item.get("prompt", "")
    answer = agent_resp.get("answer", "")
    ans_lower = answer.lower()
    tool_calls = agent_resp.get("tool_calls", [])
    empty_matrix = {"tp": False, "tn": False, "fp": False, "fn": False}

    if agent_resp.get("error"):
        return "fail", f"Agent request failed: {agent_resp['error']}", empty_matrix

    if cat == "Math":
        expected_numbers = re.findall(r"\b\d+(?:\.\d+)?\b", notes)
        called_math = any(call.get("tool") == "evaluate_math_expression" for call in tool_calls)
        matched_expected = any(number in ans_lower for number in expected_numbers) if expected_numbers else False
        if called_math or matched_expected:
            return "pass", "Math response matched the rubric or invoked the math tool", empty_matrix
        return "partial", "No expected numeric value or math tool was observed", empty_matrix

    if cat == "Tool Routing":
        expected_tool = expected_routing_tool(p_text)
        needs_tool = expected_tool is not None
        has_tool_call = bool(tool_calls)
        confusion = {
            "tp": needs_tool and has_tool_call,
            "tn": not needs_tool and not has_tool_call,
            "fp": not needs_tool and has_tool_call,
            "fn": needs_tool and not has_tool_call,
            "expected_tool": expected_tool,
        }
        called_expected_tool = any(call.get("tool") == expected_tool for call in tool_calls)
        if confusion["tn"] or (confusion["tp"] and called_expected_tool):
            return "pass", "Tool call behavior matched the prompt-intent rubric", confusion
        if confusion["tp"]:
            return "partial", f"Tool was called, but not the expected {expected_tool}", confusion
        return "partial", "Tool call behavior did not match the prompt-intent rubric", confusion

    if cat in ("Knowledge Base", "Background"):
        words = [word for word in re.findall(r"[a-zA-Z0-9_\-.]+", notes) if len(word) > 3]
        matched = sum(word in ans_lower for word in words)
        ratio = matched / len(words) if words else 0.0
        if ratio >= 0.25 or (not words and len(answer.strip()) > 30):
            return "pass", f"Expected-note term coverage: {matched}/{len(words)}", empty_matrix
        if ratio >= 0.10 or len(answer.strip()) > 30:
            return "partial", f"Partial expected-note term coverage: {matched}/{len(words)}", empty_matrix
        return "fail", f"Insufficient expected-note term coverage: {matched}/{len(words)}", empty_matrix

    if cat == "Coding":
        has_code = "```" in answer or "def " in answer or "class " in answer or "{" in answer
        if has_code:
            return "pass", "Code-formatted response observed", empty_matrix
        if len(answer.strip()) > 50:
            return "partial", "Detailed text response without a code block", empty_matrix
        return "fail", "No implementation or substantive explanation observed", empty_matrix

    if cat == "Memory":
        updated_memory = any(call.get("tool") == "update_core_memory" for call in tool_calls)
        profile_term_present = any(term in ans_lower for term in ("arpan", "rust", "300", "python", "jarvis"))
        if updated_memory or profile_term_present:
            return "pass", "Memory update tool or profile recall observed", empty_matrix
        return "partial", "No expected memory update or profile term observed", empty_matrix

    if len(answer.strip()) > 15:
        return "pass", "Substantive response generated (factual correctness not independently judged)", empty_matrix
    return "partial", "Response was shorter than the rubric threshold", empty_matrix


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def run_live_200_benchmark(limit: Optional[int] = None) -> int:
    try:
        import prompt as prompt_corpus
    except ImportError:
        from evals import prompt as prompt_corpus

    prompts = prompt_corpus.prompts_200
    if len(prompts) != 200:
        raise ValueError(f"Expected 200 stratified prompts, found {len(prompts)}.")
    if limit is not None and not 1 <= limit <= 200:
        raise ValueError("--limit must be between 1 and 200.")
    selected_prompts = prompts if limit is None else prompts[:limit]

    print(f"Chronicle live benchmark: {len(selected_prompts)} prompts")
    results: List[Dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="chronicle-eval-") as temporary_workspace:
        workspace = Path(temporary_workspace)
        chroma_dir, original_downloads_dir = prepare_benchmark_workspace(workspace)
        agents: Dict[int, MemoryAgent] = {}
        try:
            for index, prompt_item in enumerate(selected_prompts, start=1):
                chat_id = prompt_item["chat"]
                if chat_id not in agents:
                    agents[chat_id] = create_benchmark_agent(chat_id, workspace, chroma_dir)

                started = time.perf_counter()
                try:
                    response = agents[chat_id].chat(prompt_item["prompt"])
                    agent_error = None
                except Exception as exc:
                    response = {"answer": "", "tool_calls": [], "rag_documents": []}
                    agent_error = str(exc)
                duration = time.perf_counter() - started
                if agent_error:
                    response["error"] = agent_error

                verdict, reason, confusion = evaluate_response(prompt_item, response)
                results.append({
                    "id": prompt_item["id"],
                    "chat": chat_id,
                    "cat": prompt_item["cat"],
                    "prompt": prompt_item["prompt"],
                    "notes": prompt_item.get("notes", ""),
                    "verdict": verdict,
                    "reason": reason,
                    "duration_s": round(duration, 4),
                    "tool_calls": response.get("tool_calls", []),
                    "answer": response.get("answer", "")[:1000],
                    "rag_documents": response.get("rag_documents", []),
                    "error": agent_error,
                    "tool_confusion": confusion if prompt_item["cat"] == "Tool Routing" else {},
                    "timestamp": datetime.now().isoformat(),
                })
                print(f"[{index}/{len(selected_prompts)}] {prompt_item['cat']}: {verdict} ({duration:.2f}s)", flush=True)
        finally:
            for agent in agents.values():
                agent.close()
            from src import tools as tools_module
            tools_module.USER_DOWNLOADS_DIR = original_downloads_dir

    if len(selected_prompts) < 200:
        print("Partial run finished; no result JSON or charts were replaced.")
        return 0
    if len(results) != 200 or len({result["id"] for result in results}) != 200:
        raise RuntimeError("Benchmark ended without exactly 200 unique prompt results; outputs were not generated.")

    semantic_matching = score_semantic_matches(results)
    summary = summarize_live_results(results, prompt_corpus.chat_prompt_counts_200)
    summary["semantic_matching"] = semantic_matching
    add_archived_summary_fields(summary, load_archived_report())
    metrics = metrics_from_live_summary(summary)
    result_payload = {
        "source": "live_ollama",
        "timestamp": summary["timestamp"],
        "total_prompts": summary["total_prompts"],
        "total_pass": summary["total_pass"],
        "total_partial": summary["total_partial"],
        "total_fail": summary["total_fail"],
        "summary": summary,
        "metrics": metrics,
        "prompts": results,
    }
    _write_json(VERIFICATION_RESULTS_PATH, result_payload)

    saved_payload = json.loads(VERIFICATION_RESULTS_PATH.read_text(encoding="utf-8"))
    saved_metrics = saved_payload["metrics"]
    generate_charts(saved_metrics, CHARTS_DIR, results)
    print(f"Completed 200-prompt benchmark: {summary['overall_score_pct']:.1f}%")
    return 0


def regenerate_charts(persist_metrics: bool = False) -> int:
    if not persist_metrics:
        if not VERIFICATION_RESULTS_PATH.is_file():
            raise FileNotFoundError(
                f"Verification results not found: {VERIFICATION_RESULTS_PATH}"
            )
        verification_payload = json.loads(
            VERIFICATION_RESULTS_PATH.read_text(encoding="utf-8")
        )
        metrics = metrics_from_saved_payload(verification_payload)
        generate_charts(metrics, CHARTS_DIR)
        print("Regenerated PNG charts from the canonical verification results.")
        return 0

    verification_path = VERIFICATION_RESULTS_PATH
    if not verification_path.is_file():
        raise FileNotFoundError(
            f"Verification results not found: {verification_path}"
        )
    verification_payload = json.loads(
        verification_path.read_text(encoding="utf-8")
    )
    summary = verification_payload.get("summary", {})
    response_records = verification_payload.get("prompts", [])
    if not isinstance(summary, dict) or summary.get("total_prompts") != 200:
        raise ValueError("Canonical verification results must describe exactly 200 prompts.")
    if not isinstance(response_records, list) or len(response_records) != 200:
        raise ValueError("Canonical verification results must contain exactly 200 prompt records.")
    if any(
        not isinstance(record, dict)
        or "answer" not in record
        or not {"id", "prompt", "cat", "notes"}.issubset(record)
        for record in response_records
    ):
        raise ValueError(
            "Canonical verification results must include answers and prompt metadata "
            "for all 200 records."
        )
    if len({record["id"] for record in response_records}) != 200:
        raise ValueError("Saved answer records must have 200 unique prompt IDs.")

    add_archived_summary_fields(summary, load_archived_report())
    metrics = metrics_from_live_summary(summary)
    saved_metrics = verification_payload.get("metrics", {})
    semantic_matching = (
        saved_metrics.get("semantic_matching")
        or summary.get("semantic_matching")
    )
    if semantic_matching is None:
        semantic_matching = score_semantic_matches(response_records)
    metrics["semantic_matching"] = semantic_matching
    record_verdict_counts = {
        verdict: sum(
            record.get("verdict", record.get("graded")) == verdict
            for record in response_records
        )
        for verdict in ("pass", "partial", "fail")
    }
    summary_verdict_counts = {
        "pass": summary.get("total_pass", 0),
        "partial": summary.get("total_partial", 0),
        "fail": summary.get("total_fail", 0),
    }
    if record_verdict_counts != summary_verdict_counts:
        metrics["per_prompt_verdict_counts"] = record_verdict_counts
        print(
            "INFO: Summary verdict totals "
            f"{summary_verdict_counts} differ from per-prompt labels {record_verdict_counts}. "
            "The stored aggregate is preserved."
        )

    if persist_metrics:
        verification_payload.setdefault("metrics", {}).update(metrics)
        evaluation_metadata = verification_payload.setdefault("evaluation_metrics", {})
        evaluation_metadata.update({
            "source": "derived_from_saved_prompt_answers",
            "qa_reference": metrics["semantic_matching"],
            "qa_reference_caveat": (
                "Reference notes are short rubric phrases, not full gold answers; "
                "exact match and token F1 are diagnostic and will undercount valid paraphrases. "
                "Cosine similarity is a text-similarity signal, not a calibrated correctness score."
            ),
            "tool_routing_classification": {
                **metrics["tool_routing_classification"],
                "source": "stored_summary_confusion_counts",
            },
        })
        evaluation_metadata["ragas"] = {
            "status": "partially_available_from_saved_summary",
            "reported_metrics": {
                "faithfulness_pct": summary.get("faithfulness_pct"),
                "hallucination_count": summary.get("hallucination_count"),
            },
            "unavailable_metrics": [
                "answer_relevancy",
                "context_precision",
                "context_recall",
                "context_entity_recall",
                "noise_sensitivity",
            ],
            "reason": (
                "The saved JSON does not contain retrieved contexts or claim judgments "
                "needed to recalculate faithfulness and hallucination figures independently."
            ),
        }
        saved_durations = [
            float(record["duration_s"])
            for record in response_records
            if isinstance(record.get("duration_s"), (int, float))
        ]
        if len(saved_durations) == len(response_records):
            evaluation_metadata["response_latency"] = {
                "status": "measured_from_saved_per_prompt_durations",
                "mean_ms": round(float(np.mean(saved_durations)) * 1000.0, 2),
                "median_ms": round(float(np.median(saved_durations)) * 1000.0, 2),
                "p95_ms": round(float(np.percentile(saved_durations, 95)) * 1000.0, 2),
                "count": len(saved_durations),
            }
        else:
            evaluation_metadata.setdefault("response_latency", {
                "status": "not_computable_from_saved_data",
                "reason": (
                    "The saved history timestamp is written after a response and "
                    "provides no per-prompt start time or duration."
                ),
            })
        evaluation_metadata["benchmark_verdict_consistency"] = {
                "consistent": record_verdict_counts == summary_verdict_counts,
                "stored_summary_counts": summary_verdict_counts,
                "per_prompt_record_labels": record_verdict_counts,
                "stored_summary_score_pct": summary.get("overall_score_pct"),
                "comparable_benchmark_score_pct": None,
                "reason": (
                    "Per-prompt verdict labels are compared to the stored aggregate; "
                    "neither is silently substituted for the other."
                ),
            }
        verification_payload["summary"] = summary
        _write_json(verification_path, verification_payload)
        print(f"Saved derived metrics to: {verification_path}")

    saved_payload = json.loads(VERIFICATION_RESULTS_PATH.read_text(encoding="utf-8"))
    saved_metrics = saved_payload["metrics"]
    generate_charts(saved_metrics, CHARTS_DIR, saved_payload.get("prompts", []))
    print("Regenerated PNG charts from the saved 200-prompt benchmark.")
    return 0


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Run Chronicle's live 200-prompt evaluation.")
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--live", action="store_true", help="Run all 200 prompts without a persistent checkpoint.")
    modes.add_argument("--charts-only", action="store_true", help="Regenerate PNG charts from saved 200-prompt JSON.")
    modes.add_argument(
        "--rebuild-from-saved",
        action="store_true",
        help="Re-score saved answers, append derived metrics, and regenerate charts without calling Ollama.",
    )
    parser.add_argument("--limit", type=int, help="Run a small smoke-test subset; does not publish outputs.")
    args = parser.parse_args()

    if args.charts_only:
        if args.limit is not None:
            parser.error("--limit can only be used with --live.")
        return regenerate_charts()
    if args.rebuild_from_saved:
        if args.limit is not None:
            parser.error("--limit can only be used with --live.")
        return regenerate_charts(persist_metrics=True)
    return run_live_200_benchmark(limit=args.limit)


if __name__ == "__main__":
    sys.exit(main())
