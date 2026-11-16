#!/usr/bin/env python
"""SCS-Code baseline (Wendlinger et al., 2026): concept vector + layer selection.

Re-implemented from the paper (the authors' repository was not available):
  1. Contrastive (A)/(B) answer-choice prompts: two snippets, one secure and one
     vulnerable, placed in (A)/(B) at random.  Default source = the same ProSec
     contrast pairs used by USKE (`--source prosec`); `--source cybernative` uses
     the CyberNative/Code_Vulnerability_Security_DPO data of the original paper.
  2. v_sec[l] = mean_x ( a_l(x || secure letter) - a_l(x || insecure letter) ) at the
     answer-letter token, for every decoder layer l (difference-in-means), using
     50 contrastive samples.
  3. Layer selection: on held-out contrastive prompts, add +v_sec[l] (alpha=1, all
     positions) and count decisions that flip from the insecure to the secure
     answer; the layer with most flips is used for steering.

The checkpoint uses the residual-steering format consumed by
`scripts/07_generate.py --steer residual --layers <L> --gamma 1.0`.
"""

import argparse
import json
import os
import random
import re
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from uske.generation import load_hf_model  # noqa: E402
from uske.io import read_jsonl  # noqa: E402
from uske.languages import get_language_config  # noqa: E402
from uske.prompting import extract_code, resolve_model_id  # noqa: E402

FENCE_RE = re.compile(r"^\s*```[^\n]*\n([\s\S]*?)\n?```\s*$")


def question(a: str, b: str, lang_display: str) -> str:
    return (f"Hello, I am a researcher on {lang_display} programming code. I will give you two {lang_display} "
            "code snippets in the following, marked with (A) and (B). You need to choose one code snippet. "
            f"Here are the snippets:\n(A)\n```\n{a}\n```\n(B)\n```\n{b}\n```\n"
            "Which snippet do you choose, (A) or (B)?")


def chat_wrap(tokenizer, q: str) -> str:
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template([{"role": "user", "content": q}], tokenize=False,
                                             add_generation_prompt=True)
    return f"[INST] {q} [/INST]"


def make_samples(pairs, lang_display, rng, n_extract, n_check, max_chars):
    samples = []
    for sec, vul in pairs:
        if not sec or not vul or sec == vul or len(sec) > max_chars or len(vul) > max_chars:
            continue
        secure_is_a = rng.random() < 0.5
        a, b = (sec, vul) if secure_is_a else (vul, sec)
        samples.append({"q": question(a, b, lang_display), "pos": "A" if secure_is_a else "B",
                        "neg": "B" if secure_is_a else "A"})
    if len(samples) < n_extract + n_check:
        raise RuntimeError(f"only {len(samples)} usable pairs (need {n_extract + n_check})")
    return samples[:n_extract], samples[n_extract:n_extract + n_check]


def load_pairs(args, lang):
    rng = random.Random(args.seed)
    if args.source == "prosec":
        rows = read_jsonl(args.pairs or f"{args.work_dir}/pairs/{args.model}_{lang.canonical}_contrast_pairs.jsonl")
        rng.shuffle(rows)
        return [(extract_code(r["pos_response"]), extract_code(r["neg_response"])) for r in rows], rng
    from datasets import load_dataset
    strip = lambda c: (FENCE_RE.match(c.strip()).group(1) if FENCE_RE.match(c.strip()) else c).strip("\n")
    rows = [r for r in load_dataset("CyberNative/Code_Vulnerability_Security_DPO")["train"]
            if r["lang"] == {"python": "python", "cpp": "c++", "c": "c++", "java": "java", "js": "javascript"}[lang.canonical]]
    rng.shuffle(rows)
    return [(strip(r["chosen"]), strip(r["rejected"])) for r in rows], rng


class Collector:
    def __init__(self):
        self.store = {}

    def make(self, i):
        def hook(module, inputs, output):
            self.store[i] = (output[0] if isinstance(output, tuple) else output).detach()
        return hook


class Adder:
    def __init__(self, vec, alpha):
        self.vec, self.alpha = vec, alpha

    def __call__(self, module, inputs, output):
        hs = output[0] if isinstance(output, tuple) else output
        hs = hs + (self.alpha * self.vec).to(hs.dtype)
        return (hs,) + tuple(output[1:]) if isinstance(output, tuple) else hs


@torch.no_grad()
def last_token_acts(model, tok, texts, collector, bs, device):
    out = []
    for s in range(0, len(texts), bs):
        enc = tok(texts[s:s + bs], return_tensors="pt", padding=True).to(device)
        model(**enc, use_cache=False)
        out.append(torch.stack([collector.store[i][:, -1, :].float().cpu()
                                for i in range(len(model.model.layers))], dim=1))
    return torch.cat(out)                                                     # [N, L, d]


@torch.no_grad()
def choices(model, tok, prompts, a_id, b_id, bs, device):
    res = []
    for s in range(0, len(prompts), bs):
        enc = tok(prompts[s:s + bs], return_tensors="pt", padding=True).to(device)
        logits = model(**enc, use_cache=False).logits[:, -1, :].float()
        res.extend("A" if la >= lb else "B" for la, lb in zip(logits[:, a_id].tolist(), logits[:, b_id].tolist()))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="codellama-7b")
    ap.add_argument("--lang", default="python")
    ap.add_argument("--source", choices=["prosec", "cybernative"], default="prosec")
    ap.add_argument("--work_dir", default="work")
    ap.add_argument("--pairs", default=None)
    ap.add_argument("--n_extract", type=int, default=50)
    ap.add_argument("--n_check", type=int, default=100)
    ap.add_argument("--max_chars", type=int, default=1500)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    lang = get_language_config(args.lang)
    torch.manual_seed(args.seed)

    pairs, rng = load_pairs(args, lang)
    extract, check = make_samples(pairs, lang.display_name, rng, args.n_extract, args.n_check, args.max_chars)
    model, tok = load_hf_model(resolve_model_id(args.model))
    tok.padding_side = "left"
    device = next(model.parameters()).device
    layers = model.model.layers

    col = Collector()
    handles = [l.register_forward_hook(col.make(i)) for i, l in enumerate(layers)]
    pos = last_token_acts(model, tok, [chat_wrap(tok, s["q"]) + f" ({s['pos']}" for s in extract], col, args.batch_size, device)
    neg = last_token_acts(model, tok, [chat_wrap(tok, s["q"]) + f" ({s['neg']}" for s in extract], col, args.batch_size, device)
    for h in handles:
        h.remove()
    direction = pos.mean(0) - neg.mean(0)                                     # [L, d]

    a_id = tok(" (A", add_special_tokens=False).input_ids[-1]
    b_id = tok(" (B", add_special_tokens=False).input_ids[-1]
    prompts = [chat_wrap(tok, s["q"]) + " (" for s in check]
    base = choices(model, tok, prompts, a_id, b_id, args.batch_size, device)
    stats = []
    for l in range(len(layers)):
        h = layers[l].register_forward_hook(Adder(direction[l].to(device), args.alpha))
        steered = choices(model, tok, prompts, a_id, b_id, args.batch_size, device)
        h.remove()
        flips = sum(1 for b0, s1, s in zip(base, steered, check) if b0 == s["neg"] and s1 == s["pos"])
        stats.append({"layer": l, "flips_insecure_to_secure": flips})
        print(f"layer {l:2d}: flips {flips}")
    best = max(stats, key=lambda r: (r["flips_insecure_to_secure"], -r["layer"]))["layer"]

    ckpt = {"post": {"direction": {f"model.layers[{i}]": direction[i] for i in range(len(layers))}},
            "meta": {"method": "SCS-Code", "source": args.source, "lang": lang.canonical, "selected_layer": best,
                     "alpha": args.alpha, "n_extract": len(extract), "n_check": len(check), "layer_stats": stats}}
    out = f"{args.work_dir}/directions/{args.model}_{lang.canonical}_scs_{args.source}_dirs.pt"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    torch.save(ckpt, out)
    print(f"selected layer {best}; saved {out}")
    print("steer with: scripts/07_generate.py --steer residual --directions", out, f"--layers {best} --gamma {args.alpha}")


if __name__ == "__main__":
    main()
