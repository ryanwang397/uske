#!/usr/bin/env python
"""Build the CyberSecEval subset used in the paper.

Input : `CybersecurityBenchmarks/datasets/instruct/instruct-v2.json` from Meta's PurpleLlama
        repository (we used the file at commit 31a82e49; 1681 test cases).
Rule  : keep a test case if (language, cwe_identifier) is one of the evaluated pairs of
        Appendix Table 5 (same CWE sets as the secure prompt).  No other filtering.
Output: a JSON list with the original fields; 388 test cases over the five languages
        (C 32, C++ 53, Java 66, JavaScript 84, Python 153)."""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from uske.io import read_json, write_json  # noqa: E402
from uske.languages import get_language_config, supported_langs  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="path to instruct-v2.json")
    ap.add_argument("--output", default="data/cyberseceval_filtered.json")
    args = ap.parse_args()
    allowed = {(get_language_config(l).cyberseceval_lang, c) for l in supported_langs()
               for c in get_language_config(l).cwes}
    rows = [r for r in read_json(args.input)
            if (str(r.get("language", "")).lower(), str(r.get("cwe_identifier", "")).upper()) in allowed]
    write_json(args.output, rows)
    counts = {}
    for r in rows:
        counts[r["language"]] = counts.get(r["language"], 0) + 1
    print(f"{len(rows)} test cases -> {args.output}", counts)


if __name__ == "__main__":
    main()
