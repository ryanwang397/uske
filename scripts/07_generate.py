#!/usr/bin/env python
"""Step 7: greedy generation on the four benchmarks for the base model, the PKI prompt,
an edited model (USKE / target-only / ablations), any other checkpoint (e.g. ProSec-FT),
or a steered model (RSS / CLS / SCS-Code).

Outputs (under work/generations/<dataset>_<lang>/):
  prosec_test, cyberseceval -> <tag>.jsonl   rows {prompt, cwe, lang, response}  (score with scripts/02_detect.py)
  humaneval                 -> <tag>.json    list of [prompt + completion]        (bigcode harness --load_generations_path)
  mbxp                      -> <tag>.jsonl   {task_id, language, completion, completion_id} (mxeval evaluate_functional_correctness)
"""

import argparse
import atexit
import os
import sys
import tempfile

import torch
from transformers import AutoConfig, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from uske.benchmarks import SECURITY_BUDGET, UTILITY_BUDGET, load_security_benchmark, load_utility_benchmark  # noqa: E402
from uske.edits import export_edited_model, load_edit_pack  # noqa: E402
from uske.generation import hf_generate, load_hf_model, truncate_at_stop, vllm_generate  # noqa: E402
from uske.io import write_json, write_jsonl  # noqa: E402
from uske.languages import get_language_config  # noqa: E402
from uske.prompting import resolve_model_id  # noqa: E402
from uske.steering import register_component_steering, register_residual_steering  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="codellama-7b", help="base model alias / HF id (tokenizer + prompts)")
    ap.add_argument("--lang", default="python")
    ap.add_argument("--dataset", choices=["prosec_test", "cyberseceval", "humaneval", "mbxp"], required=True)
    ap.add_argument("--cyberseceval_path", default="data/cyberseceval_filtered.json")
    ap.add_argument("--template_mode", choices=["baseline", "secure"], default="baseline",
                    help="'secure' = prompt knowledge injection (PKI) baseline")
    ap.add_argument("--edit_pack", default=None, help="edit pack from scripts/06_build_edits.py")
    ap.add_argument("--model_path", default=None, help="generate from this checkpoint instead of --model")
    ap.add_argument("--steer", choices=["none", "residual", "component"], default="none",
                    help="residual = RSS (or SCS-Code with --layers <L> --gamma 1), component = CLS")
    ap.add_argument("--directions", default=None, help="direction checkpoint for --steer")
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--layers", default="all")
    ap.add_argument("--normalize_steering", action="store_true", help="add unit vectors instead of raw mean differences")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--work_dir", default="work")
    ap.add_argument("--max_samples", type=int, default=-1)
    ap.add_argument("--tensor_parallel_size", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20)
    args = ap.parse_args()
    lang = get_language_config(args.lang)
    base_id = resolve_model_id(args.model)
    tok = AutoTokenizer.from_pretrained(base_id)
    torch.manual_seed(args.seed)

    if args.dataset in ("prosec_test", "cyberseceval"):
        rows, prompts, stop = load_security_benchmark(args.dataset, lang.canonical, tok, args.template_mode,
                                                      args.cyberseceval_path)
        budget = SECURITY_BUDGET
    else:
        rows, prompts, stop = load_utility_benchmark(args.dataset, lang.canonical)
        budget = UTILITY_BUDGET
    if args.max_samples > 0:
        rows, prompts = rows[:args.max_samples], prompts[:args.max_samples]

    tag = args.tag or (os.path.splitext(os.path.basename(args.edit_pack))[0] if args.edit_pack else
                       f"{args.model}_{args.template_mode}" + (f"_{args.steer}_g{args.gamma}" if args.steer != "none" else ""))

    if args.steer != "none":                                   # hooks need the transformers path
        model, tok = load_hf_model(args.model_path or base_id)
        directions = torch.load(args.directions, map_location="cpu", weights_only=False)
        if args.steer == "residual":
            register_residual_steering(model, directions, args.gamma, args.layers, args.normalize_steering)
        else:
            register_component_steering(model, directions, args.gamma, args.layers, args.normalize_steering)
        texts = hf_generate(model, tok, prompts, budget, batch_size=args.batch_size)
    else:
        model_path = args.model_path or base_id
        if args.edit_pack:
            tmp = tempfile.TemporaryDirectory(prefix="uske_edited_")
            atexit.register(tmp.cleanup)
            model_path = export_edited_model(base_id, load_edit_pack(args.edit_pack), tmp.name)
        max_pos = AutoConfig.from_pretrained(base_id).max_position_embeddings
        texts = vllm_generate(model_path, prompts, budget, seed=args.seed, stop=stop,
                              tensor_parallel_size=args.tensor_parallel_size,
                              max_model_len=min(budget + 1024, max_pos))
    if stop:
        texts = [truncate_at_stop(t, stop) for t in texts]

    out_dir = f"{args.work_dir}/generations/{args.dataset}_{lang.canonical}"
    if args.dataset in ("prosec_test", "cyberseceval"):
        out = f"{out_dir}/{tag}.jsonl"
        write_jsonl(out, [{"prompt": r["prompt"], "cwe": r.get("cwe"), "lang": r["lang"], "response": t.strip()}
                          for r, t in zip(rows, texts)])
    elif args.dataset == "humaneval":
        out = f"{out_dir}/{tag}.json"
        write_json(out, [[p + t] for p, t in zip(prompts, texts)])
    else:
        out = f"{out_dir}/{tag}.jsonl"
        write_jsonl(out, [{"task_id": r["task_id"], "language": r.get("language"), "completion": t, "completion_id": 0}
                          for r, t in zip(rows, texts)])
    print(f"{len(texts)} generations -> {out}")


if __name__ == "__main__":
    main()
