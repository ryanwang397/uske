#!/usr/bin/env python
"""MoC baseline (Yu et al., 2025), data preparation.

Converts the ProSec contrast pairs of one language into the SVEN-style per-CWE
paired format expected by the official MoC code (github.com/viviable/MoC):
one JSONL per CWE with `func_src_before` (vulnerable) / `func_src_after` (secure).
CWEs with fewer than `--min_pairs` pairs are dropped; the rest are split 85/15
into train / val (val = held-out probe test set)."""

import argparse
import json
import os
import random
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from uske.io import read_jsonl  # noqa: E402
from uske.languages import get_language_config  # noqa: E402
from uske.prompting import extract_code  # noqa: E402

CWE_DESCRIPTIONS = {   # MoC-style one-sentence CWE descriptions (used in its few-shot probe prompt)
    "cwe-022": 'CWE-022, "Path Traversal," is a vulnerability where user-controlled paths are not properly restricted, allowing access to files outside the intended directory.',
    "cwe-078": 'CWE-078, "OS Command Injection," is a vulnerability where user-controlled input is included in an operating-system command without proper separation or neutralization.',
    "cwe-079": 'CWE-079, "Cross-site Scripting (XSS)," is a vulnerability where untrusted input is included in generated web output without proper escaping or sanitization.',
    "cwe-089": 'CWE-089, "SQL Injection," is a vulnerability where user-controlled input is incorporated into an SQL query without proper parameterization or neutralization.',
    "cwe-119": 'CWE-119, "Improper Restriction of Operations within the Bounds of a Memory Buffer," is a vulnerability where code performs memory operations outside the intended bounds of a buffer.',
    "cwe-200": 'CWE-200, "Exposure of Sensitive Information," is a vulnerability where code exposes sensitive information to an actor that is not authorized to access it.',
    "cwe-295": 'CWE-295, "Improper Certificate Validation," is a vulnerability where code fails to properly validate certificates when establishing secure connections.',
    "cwe-338": 'CWE-338, "Use of Cryptographically Weak Pseudo-Random Number Generator," is a vulnerability where a PRNG that is not cryptographically strong is used in a security-sensitive context.',
    "cwe-352": 'CWE-352, "Cross-Site Request Forgery (CSRF)," is a vulnerability where state-changing web requests are not protected from unauthorized cross-site triggering.',
    "cwe-377": 'CWE-377, "Insecure Temporary File," is a vulnerability where temporary files are created or used in a way that may allow unauthorized access, races, or leakage.',
    "cwe-502": 'CWE-502, "Deserialization of Untrusted Data," is a vulnerability where code deserializes untrusted data, allowing unexpected objects, state changes, or code paths.',
    "cwe-611": 'CWE-611, "Improper Restriction of XML External Entity Reference (XXE)," is a vulnerability where XML parsing allows external entity references that may expose files or trigger unintended external access.',
    "cwe-676": 'CWE-676, "Use of Potentially Dangerous Function," is a vulnerability where code uses functions known to be unsafe or error-prone, which can lead to memory or command vulnerabilities.',
}


def cwe_key(cwe: str) -> str:
    number = int(re.sub(r"\D", "", str(cwe)))
    return f"cwe-{number:03d}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="codellama-7b")
    ap.add_argument("--lang", default="python")
    ap.add_argument("--work_dir", default="work")
    ap.add_argument("--min_pairs", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    lang = get_language_config(args.lang)
    pairs = read_jsonl(f"{args.work_dir}/pairs/{args.model}_{lang.canonical}_contrast_pairs.jsonl")

    by_cwe = defaultdict(list)
    for p in pairs:
        if not p.get("cwe"):
            continue
        sec, vul = extract_code(p["pos_response"]), extract_code(p["neg_response"])
        if not sec or not vul or sec == vul:
            continue
        key = cwe_key(p["cwe"])
        by_cwe[key].append({"func_name": "", "func_src_before": vul, "func_src_after": sec, "line_changes": {},
                            "char_changes": {}, "file_name": f"prosec.{lang.file_ext}", "vul_type": key,
                            "commit_link": p["question"][:200]})
    out_root = f"{args.work_dir}/moc_data/{lang.canonical}"
    os.makedirs(out_root, exist_ok=True)
    with open(f"{args.work_dir}/moc_data/cwe_descriptions.json", "w") as f:
        json.dump(CWE_DESCRIPTIONS, f, indent=2)
    rng, kept = random.Random(args.seed), []
    for cwe, recs in sorted(by_cwe.items()):
        if len(recs) < args.min_pairs:
            print(f"skip {cwe}: only {len(recs)} pairs")
            continue
        rng.shuffle(recs)
        n_val = max(3, round(0.15 * len(recs)))
        for split, rows in (("train", recs[n_val:]), ("val", recs[:n_val])):
            os.makedirs(f"{out_root}/{split}", exist_ok=True)
            with open(f"{out_root}/{split}/{cwe}.jsonl", "w") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
        kept.append(cwe)
        print(f"{cwe}: train={len(recs) - n_val} val={n_val}")
    with open(f"{out_root}/cwes.json", "w") as f:
        json.dump(kept, f)
    print("kept CWEs:", kept, "->", out_root)


if __name__ == "__main__":
    main()
