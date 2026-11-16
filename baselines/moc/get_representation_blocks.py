#!/usr/bin/env python
"""MoC step 1 (representation extraction) with the official code (`get_representation.py`
from github.com/viviable/MoC), with one change: block hidden states are taken from the
decoder-layer outputs through forward hooks instead of `output_hidden_states`, whose last
entry has the final RMSNorm applied in HF Llama.  The official inference hook modifies the
raw output of `layers[steer_layer]`, so this keeps the probe / correction space identical
to the steered space.  Set MOC_SRC to `<MoC clone>/code/src`."""

import argparse
import json
import os
import random
import sys

import torch

sys.path.insert(0, os.environ.get("MOC_SRC", "MoC/code/src"))
import get_representation as GR  # noqa: E402
import utils as MOC_UTILS  # noqa: E402
from utils import CWES  # noqa: E402

if os.environ.get("MOC_CWE_DESC_JSON"):
    MOC_UTILS.CWE_DESCRIPTIONS.update(json.load(open(os.environ["MOC_CWE_DESC_JSON"])))

_STORE = []


def _install(model):
    layers = model.model.layers
    _STORE.extend([None] * len(layers))

    def make(i):
        def hook(module, inputs, output):
            _STORE[i] = output[0] if isinstance(output, tuple) else output
        return hook

    for i, layer in enumerate(layers):
        layer.register_forward_hook(make(i))


@torch.no_grad()
def forward_hidden_states_blocks(model, tokenizer, text, max_len, device, span=None, last_k=0):
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_len).to(device)
    model(**enc, use_cache=False)
    T = enc.input_ids.size(1)
    if last_k > 0:
        m, n = max(0, T - last_k), T
    elif span is None:
        m, n = T - 1, T
    else:
        m, n = max(0, min(int(span[0]), T - 1)), None
        n = max(m + 1, min(int(span[1]), T))
    pooled = torch.stack([h[0, m:n, :].mean(dim=0).detach().cpu().float() for h in _STORE], dim=0)
    return pooled, (m, n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--save_path", required=True)
    ap.add_argument("--splits", nargs="+", default=["train", "val"])
    ap.add_argument("--cwes", nargs="+", default=CWES)
    ap.add_argument("--few_shot_pairs", type=int, default=2)
    ap.add_argument("--max_examples", type=int, default=0)
    ap.add_argument("--max_len", type=int, default=4096)
    ap.add_argument("--pool", choices=["span_mean", "last_token", "lastk_mean", "code_mean"], default="span_mean")
    ap.add_argument("--last_k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = GR.load_model(args.model_name)
    _install(model)
    GR.forward_hidden_states = forward_hidden_states_blocks
    for split in args.splits:
        for cwe in args.cwes:
            GR.process_cwe(args, model, tokenizer, cwe, split, device)


if __name__ == "__main__":
    main()
