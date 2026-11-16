"""Dataset access and the data splits used in the paper (Appendix B.1).

ProSec     : Hugging Face `prosecalign/prosec-mixed-clm7b-inst` (train split).  For each
             language we keep the non-benign vulnerability-inducing instructions and split
             them 40% / 60% with `train_test_split(train_size=0.4, seed=20, shuffle=True)`.
             The 40% "adaptation" part is used for contrast pairs (and, held-out inside it,
             for development); the 60% part is the ProSec test set reported in the paper.
CyberSecEval: `datasets/instruct/instruct-v2.json` of Meta's CybersecurityBenchmarks (PurpleLlama),
             restricted to the (language, CWE) pairs of Appendix Table 5
             (`scripts/prepare_cyberseceval.py`; 388 test cases).
MBPP       : `Muennighoff/mbpp` (full), training tasks 601-974, used only to collect benign
             activation statistics.
"""

from __future__ import annotations

from datasets import load_dataset

from .io import read_json
from .languages import get_language_config

PROSEC_DATASET = "prosecalign/prosec-mixed-clm7b-inst"
MBPP_DATASET = "Muennighoff/mbpp"
PROSEC_SEED = 20
PROSEC_ADAPT_FRACTION = 0.4


def load_prosec(lang: str, split: str = "test", seed: int = PROSEC_SEED,
                adapt_fraction: float = PROSEC_ADAPT_FRACTION):
    """Return the ProSec rows of one language.  `split` is `adapt` (40%) or `test` (60%).
    Each row has `original_instruction` (the task) and `cwe`."""
    cfg = get_language_config(lang)
    ds = load_dataset(PROSEC_DATASET)["train"]
    ds = ds.filter(lambda x: x["lang"] == cfg.prosec_lang and x["benign"] is False)
    parts = ds.train_test_split(train_size=adapt_fraction, seed=seed, shuffle=True)
    if split not in ("adapt", "test"):
        raise ValueError("split must be 'adapt' or 'test'")
    return parts["train"] if split == "adapt" else parts["test"]


def prosec_tasks(rows) -> list[dict]:
    return [{"prompt": r["original_instruction"], "cwe": r.get("cwe")} for r in rows]


def load_cyberseceval(path: str, lang: str) -> list[dict]:
    """Rows of the (filtered) CyberSecEval instruct file for one language.
    Each row has `test_case_prompt`, `language` and `cwe_identifier`."""
    cfg = get_language_config(lang)
    rows = read_json(path)
    return [{"prompt": r["test_case_prompt"], "cwe": r.get("cwe_identifier")}
            for r in rows if r.get("language") == cfg.cyberseceval_lang]


def load_mbpp_train(task_id_start: int = 601, task_id_end: int = 974):
    ds = load_dataset(MBPP_DATASET, "full")
    ds = ds["train"] if "train" in ds else ds[list(ds.keys())[0]]
    return ds.filter(lambda x: task_id_start <= int(x["task_id"]) <= task_id_end)
