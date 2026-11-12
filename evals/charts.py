#!/usr/bin/env python3
"""Generate charts from the canonical saved verification results."""

import json
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Dict


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT_CANDIDATES = (
    CURRENT_DIR.parent,
    CURRENT_DIR.parent / "Chronicle-Infinite-Memory-LLM-Agent",
)
PROJECT_ROOT = next(
    (path for path in PROJECT_ROOT_CANDIDATES if (path / "src" / "engine.py").exists()),
    CURRENT_DIR.parent,
)
VERIFICATION_RESULTS_PATH = PROJECT_ROOT / "data" / "verification_results.json"
CHART_OUTPUT_DIR = PROJECT_ROOT / "evals" / "results"


def _load_evaluator() -> ModuleType:
    sys.path.insert(0, str(PROJECT_ROOT))
    evaluator_path = CURRENT_DIR / "evaluate.py"
    spec = importlib.util.spec_from_file_location("chronicle_evaluate", evaluator_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load chart generators from {evaluator_path}")
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    return evaluator


def load_metrics(evaluator: ModuleType) -> Dict[str, Any]:
    """Load and normalize chart inputs through the canonical evaluation helper."""
    if not VERIFICATION_RESULTS_PATH.is_file():
        raise FileNotFoundError(
            f"Verification results not found: {VERIFICATION_RESULTS_PATH}"
        )

    payload = json.loads(VERIFICATION_RESULTS_PATH.read_text(encoding="utf-8"))
    return evaluator.metrics_from_saved_payload(payload)


def main() -> None:
    evaluator = _load_evaluator()
    evaluator.generate_charts(load_metrics(evaluator), CHART_OUTPUT_DIR)


if __name__ == "__main__":
    main()
