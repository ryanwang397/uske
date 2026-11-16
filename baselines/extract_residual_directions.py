#!/usr/bin/env python
"""Residual-stream directions for the RSS baseline: per decoder layer, the mean residual
activation over the response span of the secure completions minus that of the vulnerable
completions (same contrast pairs and prompts as step 4)."""

import argparse
import os
import sys

import torch
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from uske.activations import decoder_layers, response_span  # noqa: E402
from uske.generation import load_hf_model  # noqa: E402
from uske.io import read_jsonl  # noqa: E402
from uske.languages import get_language_config  # noqa: E402
from uske.prompting import resolve_model_id, secure_code_prompt  # noqa: E402


@torch.no_grad()
def layer_means(model, tok, pairs, key, template_mode, lang, device):
    layers = decoder_layers(model)
    store = {}

    def make(idx):
        def hook(module, inputs, output):
            store[idx] = (output[0] if isinstance(output, tuple) else output).detach()
        return hook

    handles = [l.register_forward_hook(make(i)) for i, l in enumerate(layers)]
    acc, n = None, 0
    for p in tqdm(pairs, ncols=80, desc=template_mode):
        prompt = secure_code_prompt(tok, p["question"], lang, template_mode).strip()
        start, end = response_span(tok, prompt, p[key])
        toks = tok(prompt + p[key], return_tensors="pt", truncation=True, max_length=4096).to(device)
        end = min(end, toks.input_ids.shape[1])
        if end <= start:
            continue
        model(**toks)
        span = torch.stack([store[i][0, start - 1:end - 1].float().mean(0).cpu() for i in range(len(layers))])
        acc = span if acc is None else acc + span
        n += 1
    for h in handles:
        h.remove()
    return acc / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="codellama-7b")
    ap.add_argument("--lang", default="python")
    ap.add_argument("--work_dir", default="work")
    args = ap.parse_args()
    lang = get_language_config(args.lang)
    pairs = read_jsonl(f"{args.work_dir}/pairs/{args.model}_{lang.canonical}_contrast_pairs.jsonl")
    model, tok = load_hf_model(resolve_model_id(args.model))
    device = next(model.parameters()).device
    pos = layer_means(model, tok, pairs, "pos_response", "secure", lang.canonical, device)
    neg = layer_means(model, tok, pairs, "neg_response", "baseline", lang.canonical, device)
    L = len(pos)
    ckpt = {"post": {"pos_mean": {f"model.layers[{i}]": pos[i] for i in range(L)},
                     "neg_mean": {f"model.layers[{i}]": neg[i] for i in range(L)},
                     "direction": {f"model.layers[{i}]": pos[i] - neg[i] for i in range(L)}},
            "meta": {"model": args.model, "lang": lang.canonical, "num_pairs": len(pairs)}}
    out = f"{args.work_dir}/directions/{args.model}_{lang.canonical}_residual_directions.pt"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    torch.save(ckpt, out)
    print(f"residual directions -> {out}")


if __name__ == "__main__":
    main()
