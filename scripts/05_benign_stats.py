#!/usr/bin/env python
"""Step 5: benign utility statistics (Section 3.2 / Appendix F).

5a) `--stage generate`: let the base model answer MBPP training tasks (ids 601-974)
    with the baseline prompt, task text rewritten to the target language.
5b) `--stage stats`:    run prompt + own completion through the model and collect
    per-head PCA bases and MLP second moments over the response span."""

import argparse
import os
import sys

import torch
from transformers import AutoConfig, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from uske.benign_stats import extract_benign_stats  # noqa: E402
from uske.data import load_mbpp_train  # noqa: E402
from uske.generation import load_hf_model, vllm_generate  # noqa: E402
from uske.io import read_jsonl, write_jsonl  # noqa: E402
from uske.languages import get_language_config, rewrite_task_language  # noqa: E402
from uske.prompting import benign_code_prompt, resolve_model_id  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["generate", "stats"], required=True)
    ap.add_argument("--model", default="codellama-7b")
    ap.add_argument("--lang", default="python")
    ap.add_argument("--work_dir", default="work")
    ap.add_argument("--top_k", type=int, default=16)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--tensor_parallel_size", type=int, default=1)
    ap.add_argument("--seed", type=int, default=20)
    args = ap.parse_args()
    lang = get_language_config(args.lang)
    model_id = resolve_model_id(args.model)
    responses_path = f"{args.work_dir}/benign/{args.model}_{lang.canonical}_mbpp_responses.jsonl"

    if args.stage == "generate":
        tok = AutoTokenizer.from_pretrained(model_id)
        rows = load_mbpp_train()
        tasks = [rewrite_task_language(r["text"], lang.canonical) for r in rows]
        prompts = [benign_code_prompt(tok, t, lang.canonical) for t in tasks]
        max_pos = AutoConfig.from_pretrained(model_id).max_position_embeddings
        texts = vllm_generate(model_id, prompts, args.max_new_tokens, seed=args.seed,
                              tensor_parallel_size=args.tensor_parallel_size, max_model_len=min(2048, max_pos))
        write_jsonl(responses_path, [{"task_id": int(r["task_id"]), "task": t, "response": x}
                                     for r, t, x in zip(rows, tasks, texts)])
        print(f"{len(texts)} benign responses -> {responses_path}")
        return

    torch.manual_seed(args.seed)
    examples = read_jsonl(responses_path)
    model, tok = load_hf_model(model_id)
    stats = extract_benign_stats(model, tok, examples, lang.canonical, top_k=args.top_k,
                                 device=next(model.parameters()).device)
    stats["meta"].update({"model": args.model, "dataset": "MBPP train (task_id 601-974)"})
    out = f"{args.work_dir}/benign/{args.model}_{lang.canonical}_benign_stats_K{args.top_k}.pt"
    torch.save(stats, out)
    print(f"benign stats from {stats['meta']['num_samples']} samples -> {out}")


if __name__ == "__main__":
    main()
