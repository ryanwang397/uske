#!/usr/bin/env python
"""Step 2 / evaluation: run the Insecure Code Detector (ICD) of Meta's
CybersecurityBenchmarks (PurpleLlama) on generated code and summarize the
vulnerable-code ratio.

Input : JSONL rows with `prompt`, `cwe`, `lang` (detector language) and `response`
        (as written by scripts/01_generate_contrast_responses.py and scripts/07_generate.py).
Output: JSONL rows with the extracted `code` and the ICD `detection_results`, plus a
        summary JSON.  Set PURPLELLAMA_DIR to a clone of https://github.com/meta-llama/PurpleLlama
        (its CybersecurityBenchmarks needs `weggli` and `semgrep` installed)."""

import argparse
import asyncio
import json
import os
import sys
from multiprocessing import Pool

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from uske.io import read_jsonl, write_json, write_jsonl  # noqa: E402
from uske.prompting import extract_code  # noqa: E402

sys.path.append(os.path.join(os.environ.get("PURPLELLAMA_DIR", "PurpleLlama"), "CybersecurityBenchmarks"))


def _icd():
    from insecure_code_detector import insecure_code_detector, languages
    lang_map = {"c": languages.Language.C, "cpp": languages.Language.CPP, "java": languages.Language.JAVA,
                "javascript": languages.Language.JAVASCRIPT, "python": languages.Language.PYTHON}
    return insecure_code_detector, lang_map


def detect_one(row):
    insecure_code_detector, LANG = _icd()
    code = extract_code(row.get("response", ""))
    out = {"prompt": row["prompt"], "cwe": row.get("cwe"), "lang": row["lang"], "response": row.get("response", ""),
           "code": code, "detection_results": []}
    if not code:
        return out
    try:
        issues = asyncio.run(asyncio.wait_for(insecure_code_detector.analyze(LANG[row["lang"]], code), timeout=30))
    except asyncio.TimeoutError:
        out["timeout"] = True
        return out
    out["detection_results"] = [{"cwe_id": i.cwe_id, "analyzer": str(i.analyzer), "rule": i.rule,
                                 "pattern_id": i.pattern_id} for i in issues]
    return out


def is_vulnerable(row, target_cwe_only: bool) -> bool:
    hits = row.get("detection_results") or []
    if not target_cwe_only or not row.get("cwe"):
        return bool(hits)
    target = str(row["cwe"]).upper().replace("CWE-0", "CWE-")
    return any(str(h.get("cwe_id", "")).upper().replace("CWE-0", "CWE-") == target for h in hits)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--target_cwe_only", action="store_true",
                    help="count a sample as vulnerable only if a detection matches the prompt's target CWE")
    args = ap.parse_args()
    rows = read_jsonl(args.input)
    with Pool(args.workers) as pool:
        results = list(pool.imap(detect_one, rows, chunksize=4))
    out = args.output or args.input.replace(".jsonl", ".detected.jsonl")
    write_jsonl(out, results)
    scored = [r for r in results if not r.get("timeout")]      # timed-out programs are excluded
    n_vul = sum(is_vulnerable(r, args.target_cwe_only) for r in scored)
    summary = {"n": len(scored), "n_vulnerable": n_vul, "vul_ratio": 100.0 * n_vul / max(1, len(scored)),
               "rule": "target CWE" if args.target_cwe_only else "any detection",
               "n_timeout": len(results) - len(scored)}
    write_json(out.replace(".jsonl", ".summary.json"), summary)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
