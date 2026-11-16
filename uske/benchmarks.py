"""Benchmark loaders producing (rows, prompts, stop_strings) for `scripts/07_generate.py`.

Security benchmarks are prompted with the chat template (baseline or secure system/user
prompts); utility benchmarks are completion-style (the raw prompt of the task, cut at the
usual stop strings), scored by bigcode-evaluation-harness (HumanEval / MultiPL-E) and
mxeval (MBXP)."""

from __future__ import annotations

from .data import load_cyberseceval, load_prosec, prosec_tasks
from .languages import get_language_config
from .prompting import secure_code_prompt

MULTIPLE_REVISION = "28441b6024e71d4a1c1c0f6bf171c935cd5a43f2"   # nuprl/MultiPL-E snapshot used in the paper
HUMANEVAL_STOP = ["\nclass", "\ndef", "\n#", "\nif", "\nprint", "\n```"]
MBXP_STOP = ["<file_sep>", "```", "\n\n\n"]
MBXP_SPLIT = {"python": "python", "c": "cpp", "cpp": "cpp", "java": "java", "js": "javascript"}
SECURITY_BUDGET = 2048
UTILITY_BUDGET = 512


def load_security_benchmark(name: str, lang: str, tokenizer, template_mode: str = "baseline",
                            cyberseceval_path: str = "data/cyberseceval_filtered.json"):
    cfg = get_language_config(lang)
    if name == "prosec_test":
        rows = prosec_tasks(load_prosec(cfg.canonical, split="test"))
    elif name == "cyberseceval":
        rows = load_cyberseceval(cyberseceval_path, cfg.canonical)
    else:
        raise ValueError(name)
    for r in rows:
        r["lang"] = cfg.detector_lang
    prompts = [secure_code_prompt(tokenizer, r["prompt"], cfg.canonical, template_mode) for r in rows]
    return rows, prompts, None


def load_utility_benchmark(name: str, lang: str):
    from datasets import load_dataset

    cfg = get_language_config(lang)
    if name == "humaneval":
        if cfg.canonical == "python":
            ds = load_dataset("openai_humaneval")["test"]
            stop = list(HUMANEVAL_STOP)
        else:
            multiple_lang = cfg.utility_task.replace("multiple-", "")        # cpp / java / js
            ds = load_dataset("nuprl/MultiPL-E", f"humaneval-{multiple_lang}", revision=MULTIPLE_REVISION)["test"]
            stop = (list(ds[0]["stop_tokens"]) if "stop_tokens" in ds[0] else []) + ["<file_sep>"]
        rows = [dict(d) for d in ds]
    elif name == "mbxp":
        ds = load_dataset("AmazonScience/mxeval", "mbxp", split=MBXP_SPLIT[cfg.canonical], trust_remote_code=True)
        rows, stop = [dict(d) for d in ds], list(MBXP_STOP)
    else:
        raise ValueError(name)
    prompts = [r["prompt"].strip() for r in rows]
    return rows, prompts, stop
