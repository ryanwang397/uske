#!/usr/bin/env python
"""Step 3: keep prompts whose baseline completion is vulnerable and whose
secure-prompt completion is clean (Appendix B.2).

Inputs are the detector outputs of step 2 (one JSONL per template), each row
carrying `prompt`, `response` and `detection_results`."""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from uske.io import read_jsonl, write_jsonl  # noqa: E402
from uske.languages import get_language_config  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="codellama-7b")
    ap.add_argument("--lang", default="python")
    ap.add_argument("--work_dir", default="work")
    ap.add_argument("--baseline_detected", default=None)
    ap.add_argument("--secure_detected", default=None)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()
    lang = get_language_config(args.lang)

    base = args.baseline_detected or f"{args.work_dir}/detected/{args.model}_{lang.canonical}_baseline.jsonl"
    sec = args.secure_detected or f"{args.work_dir}/detected/{args.model}_{lang.canonical}_secure.jsonl"
    out = args.output or f"{args.work_dir}/pairs/{args.model}_{lang.canonical}_contrast_pairs.jsonl"

    secure_by_prompt = {r["prompt"]: r for r in read_jsonl(sec)}
    pairs = []
    for b in read_jsonl(base):
        s = secure_by_prompt.get(b["prompt"])
        if s is None:
            continue
        if b.get("detection_results") and not s.get("detection_results"):
            pairs.append({"question": b["prompt"], "lang": lang.detector_lang, "cwe": b.get("cwe"),
                          "neg_response": b["response"], "pos_response": s["response"]})
    write_jsonl(out, pairs)
    print(f"{len(pairs)} contrast pairs -> {out}")


if __name__ == "__main__":
    main()
