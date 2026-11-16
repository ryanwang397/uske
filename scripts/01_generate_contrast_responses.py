#!/usr/bin/env python
"""Step 1: generate completions on the ProSec 40% adaptation split with the baseline
template (candidate vulnerable completions) and with the secure template
(candidate secure completions).  Greedy decoding, 2048 new tokens."""

import argparse
import os
import sys

from transformers import AutoConfig, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from uske.data import load_prosec, prosec_tasks  # noqa: E402
from uske.generation import vllm_generate  # noqa: E402
from uske.io import write_jsonl  # noqa: E402
from uske.languages import get_language_config  # noqa: E402
from uske.prompting import resolve_model_id, secure_code_prompt  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="codellama-7b")
    ap.add_argument("--lang", default="python")
    ap.add_argument("--template_mode", choices=["baseline", "secure"], required=True)
    ap.add_argument("--work_dir", default="work")
    ap.add_argument("--max_new_tokens", type=int, default=2048)
    ap.add_argument("--tensor_parallel_size", type=int, default=1)
    ap.add_argument("--seed", type=int, default=20)
    args = ap.parse_args()
    lang = get_language_config(args.lang)
    model_id = resolve_model_id(args.model)

    tasks = prosec_tasks(load_prosec(lang.canonical, split="adapt", seed=args.seed))
    tok = AutoTokenizer.from_pretrained(model_id)
    prompts = [secure_code_prompt(tok, t["prompt"], lang.canonical, args.template_mode) for t in tasks]
    max_pos = AutoConfig.from_pretrained(model_id).max_position_embeddings
    texts = vllm_generate(model_id, prompts, args.max_new_tokens, seed=args.seed,
                          tensor_parallel_size=args.tensor_parallel_size,
                          max_model_len=min(args.max_new_tokens + 1024, max_pos))
    out = f"{args.work_dir}/responses/{args.model}_{lang.canonical}_{args.template_mode}.jsonl"
    write_jsonl(out, [{"prompt": t["prompt"], "cwe": t["cwe"], "lang": lang.detector_lang, "response": x}
                      for t, x in zip(tasks, texts)])
    print(f"{len(texts)} {args.template_mode} responses -> {out}")


if __name__ == "__main__":
    main()
