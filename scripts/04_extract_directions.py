#!/usr/bin/env python
"""Step 4: extract per-layer security concept vectors (attention and MLP blocks)
from the contrast pairs, together with the mean block inputs used as editing context."""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from uske.directions import extract_directions  # noqa: E402
from uske.generation import load_hf_model  # noqa: E402
from uske.io import read_jsonl  # noqa: E402
from uske.languages import get_language_config  # noqa: E402
from uske.prompting import resolve_model_id  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="codellama-7b")
    ap.add_argument("--lang", default="python")
    ap.add_argument("--work_dir", default="work")
    ap.add_argument("--pairs", default=None)
    ap.add_argument("--output", default=None)
    ap.add_argument("--seed", type=int, default=20)
    args = ap.parse_args()
    lang = get_language_config(args.lang)
    torch.manual_seed(args.seed)

    pairs = read_jsonl(args.pairs or f"{args.work_dir}/pairs/{args.model}_{lang.canonical}_contrast_pairs.jsonl")
    model, tok = load_hf_model(resolve_model_id(args.model))
    directions = extract_directions(model, tok, pairs, lang.canonical, device=next(model.parameters()).device)
    directions["meta"].update({"model": args.model})
    out = args.output or f"{args.work_dir}/directions/{args.model}_{lang.canonical}_directions.pt"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    torch.save(directions, out)
    print(f"directions from {len(pairs)} pairs -> {out}")


if __name__ == "__main__":
    main()
