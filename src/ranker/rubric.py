"""Load and provide typed access to the frozen rubric.yaml.

The rubric is the single source of truth for the JD-understanding layer. We load it
once and pass the dict around; helper accessors keep feature code readable and make
missing-key bugs loud rather than silent.
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, Dict, List

import yaml

DEFAULT_RUBRIC_PATH = os.environ.get(
    "RUBRIC_PATH",
    os.path.join(os.path.dirname(__file__), "..", "..", "rubric.yaml"),
)


@lru_cache(maxsize=4)
def load_rubric(path: str = DEFAULT_RUBRIC_PATH) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        rub = yaml.safe_load(f)
    _validate(rub)
    return rub


def _validate(rub: Dict[str, Any]) -> None:
    required = [
        "domain", "experience", "must_haves", "disqualifiers",
        "location", "product_company", "behavioral", "consistency", "composition",
    ]
    missing = [k for k in required if k not in rub]
    if missing:
        raise ValueError(f"rubric.yaml missing top-level sections: {missing}")


def lower_list(rub_section: Any) -> List[str]:
    """Return a section that is expected to be a list of strings, lowercased."""
    if rub_section is None:
        return []
    return [str(x).lower() for x in rub_section]
