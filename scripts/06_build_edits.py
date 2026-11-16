#!/usr/bin/env python
"""Step 6: utility-regularized edit allocation (Section 3.3) and optional export of the
edited checkpoint.

Examples
  # USKE, attention-only edits
  python scripts/06_build_edits.py --model codellama-7b --lang python --rho_attn 0.3 --alpha 0.8 --beta 0.15
  # target-only allocation (Steer2Edit rule) = same call with --beta 0
  # ablations: --key_mode {conditional,mean,random}, --alpha 0 (L2-only) / 0.99 (near-L1), --pca_top_k {4,8,16}
"""

import argparse
import os
import sys

import torch
from transformers import AutoModelForCausalLM

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from uske.edits import EditConfig, build_edit_pack, export_edited_model, save_edit_pack, summarize_pack  # noqa: E402
from uske.io import write_json  # noqa: E402
from uske.languages import get_language_config  # noqa: E402
from uske.prompting import resolve_model_id  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="codellama-7b")
    ap.add_argument("--lang", default="python")
    ap.add_argument("--work_dir", default="work")
    ap.add_argument("--directions", default=None)
    ap.add_argument("--benign_stats", default=None)
    ap.add_argument("--rho_attn", type=float, default=-1.0)
    ap.add_argument("--rho_mlp", type=float, default=-1.0)
    ap.add_argument("--alpha", type=float, default=0.90)
    ap.add_argument("--beta", type=float, default=0.15)
    ap.add_argument("--key_mode", choices=["conditional", "mean", "random"], default="conditional")
    ap.add_argument("--pca_top_k", type=int, default=16)
    ap.add_argument("--seed", type=int, default=20)
    ap.add_argument("--export_dir", default=None, help="also write a merged HF checkpoint here")
    args = ap.parse_args()
    if args.rho_attn <= 0 and args.rho_mlp <= 0:
        ap.error("set --rho_attn and/or --rho_mlp > 0")
    if not 0.0 <= args.alpha < 1.0:
        ap.error("--alpha must be in [0, 1); use 0.99 for the near-L1 ablation")
    lang = get_language_config(args.lang)
    model_id = resolve_model_id(args.model)

    directions = torch.load(args.directions or f"{args.work_dir}/directions/{args.model}_{lang.canonical}_directions.pt",
                            map_location="cpu", weights_only=False)
    stats = None
    if args.beta > 0:
        stats = torch.load(args.benign_stats or
                           f"{args.work_dir}/benign/{args.model}_{lang.canonical}_benign_stats_K16.pt",
                           map_location="cpu", weights_only=False)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="cpu",
                                                 low_cpu_mem_usage=True)
    cfg = EditConfig(rho_attn=args.rho_attn, rho_mlp=args.rho_mlp, alpha=args.alpha, beta=args.beta,
                     key_mode=args.key_mode, pca_top_k=args.pca_top_k, seed=args.seed)
    pack = build_edit_pack(model, directions, stats, cfg)
    out_dir = f"{args.work_dir}/edits/{args.model}_{lang.canonical}"
    path = os.path.join(out_dir, cfg.name())
    save_edit_pack(pack, path)
    summary = summarize_pack(pack)
    write_json(path.replace(".pt", ".json"), summary)
    print(f"{summary['num_heads']} heads, {summary['num_neurons']} neurons edited -> {path}")
    if args.export_dir:
        print("merged checkpoint ->", export_edited_model(model_id, pack, args.export_dir))


if __name__ == "__main__":
    main()
