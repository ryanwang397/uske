#!/usr/bin/env python
"""MoC baseline generation with the OFFICIAL steering hook (`MoCSteerer` from
github.com/viviable/MoC, code/src/moc_generate.py).  Per generated token
    s_last += alpha(t) * sum_j [ p_vul_j(s_last) > 0.5 ] * Delta s_j,   alpha(t) = exp(-decay * t).
Prompts, decoding budget (greedy; 2048 / 512 new tokens) and output formats are those of
scripts/07_generate.py, so the outputs are scored the same way.  Set MOC_SRC to <MoC>/code/src."""

import argparse
import math
import os
import sys

import torch
from transformers import StoppingCriteria, StoppingCriteriaList

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.environ.get("MOC_SRC", "MoC/code/src"))
from moc_generate import MoCSteerer, get_decoder_layers, load_all_probes, load_static_corrections  # noqa: E402
from uske.benchmarks import SECURITY_BUDGET, UTILITY_BUDGET, load_security_benchmark, load_utility_benchmark  # noqa: E402
from uske.generation import load_hf_model, truncate_at_stop  # noqa: E402
from uske.io import write_json, write_jsonl  # noqa: E402
from uske.languages import get_language_config  # noqa: E402
from uske.prompting import resolve_model_id  # noqa: E402


class StatsSteerer(MoCSteerer):
    """Official hook plus gate statistics (how often each CWE probe fires, correction norm)."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.n_steps, self.fires, self.delta_norms, self.hidden_norm = 0, {}, [], 0.0

    @torch.no_grad()
    def __call__(self, module, inputs, output):
        if self._enabled and self.mode == "harden" and self._prompt_seen:
            hs = output[0] if isinstance(output, tuple) else output
            last = hs[:, -1, :].float()
            self.n_steps += last.size(0)
            total = torch.zeros_like(last)
            for cwe, ent in self.probes.items():
                inp = last if ent["reducer"] is None else ent["reducer"].transform(last)
                gate = torch.softmax(ent["probe"](inp), dim=-1)[:, 0] > 0.5
                self.fires[cwe] = self.fires.get(cwe, 0) + int(gate.sum())
                if cwe in self.static_corr:
                    total = total + gate.float().unsqueeze(-1) * self.static_corr[cwe].unsqueeze(0)
            self.delta_norms.append((math.exp(-self.decay_rate * self.t) * total.norm(dim=-1)).mean().item())
            self.hidden_norm = last.norm(dim=-1).mean().item()
        return super().__call__(module, inputs, output)

    def report(self):
        if not self.n_steps:
            return "no steered steps"
        rates = {c: round(n / self.n_steps, 3) for c, n in sorted(self.fires.items())}
        return (f"steered token-steps={self.n_steps} gate-fire-rate={rates} "
                f"mean|alpha*delta|={sum(self.delta_norms) / len(self.delta_norms):.3f} hidden-norm~{self.hidden_norm:.1f}")


class StopOnStrings(StoppingCriteria):
    def __init__(self, tokenizer, stops, prompt_len):
        self.tokenizer, self.stops, self.prompt_len = tokenizer, stops, prompt_len

    def __call__(self, input_ids, scores, **kwargs):
        texts = self.tokenizer.batch_decode(input_ids[:, self.prompt_len:], skip_special_tokens=True)
        return all(any(s in t for s in self.stops) for t in texts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="codellama-7b")
    ap.add_argument("--lang", default="python")
    ap.add_argument("--dataset", choices=["prosec_test", "cyberseceval", "humaneval", "mbxp"], required=True)
    ap.add_argument("--cyberseceval_path", default="data/cyberseceval_filtered.json")
    ap.add_argument("--probe_dir", required=True)
    ap.add_argument("--corrections", required=True, help="static_n.pt (official default) or static_g.pt")
    ap.add_argument("--cwes", nargs="+", required=True)
    ap.add_argument("--steer_layer", type=int, default=-1)
    ap.add_argument("--decay_rate", type=float, default=0.05)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--work_dir", default="work")
    ap.add_argument("--tag", default="moc")
    ap.add_argument("--max_samples", type=int, default=-1)
    args = ap.parse_args()
    lang = get_language_config(args.lang)
    model, tok = load_hf_model(resolve_model_id(args.model))
    tok.padding_side = "left"
    device = next(model.parameters()).device

    if args.dataset in ("prosec_test", "cyberseceval"):
        rows, prompts, stop = load_security_benchmark(args.dataset, lang.canonical, tok, "baseline", args.cyberseceval_path)
        budget = SECURITY_BUDGET
    else:
        rows, prompts, stop = load_utility_benchmark(args.dataset, lang.canonical)
        budget = UTILITY_BUDGET
    if args.max_samples > 0:
        rows, prompts = rows[:args.max_samples], prompts[:args.max_samples]

    probes = load_all_probes(args.probe_dir, args.cwes, device, args.steer_layer)
    static_corr = load_static_corrections(args.corrections, args.cwes, device)
    steerer = StatsSteerer(probes, static_corr, None, mode="harden", decay_rate=args.decay_rate)
    handle = get_decoder_layers(model)[args.steer_layer].register_forward_hook(steerer)

    texts = [None] * len(prompts)
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    for s in range(0, len(order), args.batch_size):
        idx = order[s:s + args.batch_size]
        enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True).to(device)
        steerer.reset()
        crit = StoppingCriteriaList([StopOnStrings(tok, stop, enc.input_ids.size(1))]) if stop else None
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=budget, do_sample=False, pad_token_id=tok.pad_token_id,
                                 stopping_criteria=crit)
        for i, t in zip(idx, tok.batch_decode(gen[:, enc.input_ids.size(1):], skip_special_tokens=True)):
            texts[i] = truncate_at_stop(t, stop) if stop else t
    handle.remove()
    print("[MoC stats]", steerer.report())

    out_dir = f"{args.work_dir}/generations/{args.dataset}_{lang.canonical}"
    if args.dataset in ("prosec_test", "cyberseceval"):
        out = f"{out_dir}/{args.tag}.jsonl"
        write_jsonl(out, [{"prompt": r["prompt"], "cwe": r.get("cwe"), "lang": r["lang"], "response": t.strip()}
                          for r, t in zip(rows, texts)])
    elif args.dataset == "humaneval":
        out = f"{out_dir}/{args.tag}.json"
        write_json(out, [[p + t] for p, t in zip(prompts, texts)])
    else:
        out = f"{out_dir}/{args.tag}.jsonl"
        write_jsonl(out, [{"task_id": r["task_id"], "language": r.get("language"), "completion": t, "completion_id": 0}
                          for r, t in zip(rows, texts)])
    print(f"{len(texts)} generations -> {out}")


if __name__ == "__main__":
    main()
