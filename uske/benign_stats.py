"""Benign activation statistics for utility-risk estimation (Section 3.2, Appendix F).

Attention heads: per-head covariance of the (response-averaged) head output that
feeds `o_proj`, reduced to its top-K principal directions and normalized
variance weights.  MLP neurons: first/second moments of the activation feeding
`down_proj`, with the second moment normalized within each layer.
"""

from __future__ import annotations

import torch
from tqdm import tqdm

from .activations import ComponentRecorder, attn_key, mlp_key, response_span, span_mean
from .languages import get_language_config
from .prompting import benign_code_prompt


def top_pca_from_cov(cov: torch.Tensor, top_k: int):
    eigvals, eigvecs = torch.linalg.eigh(cov.double())
    eigvals = eigvals.clamp_min(0.0)
    order = torch.argsort(eigvals, descending=True)[: max(0, min(top_k, cov.shape[0]))]
    basis = eigvecs[:, order].T.contiguous().float()          # [K, D]
    total = eigvals.sum()
    weight = (eigvals[order] / total).float() if total > 0 else torch.zeros(len(order))
    return basis, weight


@torch.no_grad()
def extract_benign_stats(model, tokenizer, examples, lang: str, top_k: int = 16,
                         device="cuda", max_len: int = 4096) -> dict:
    """`examples`: iterable of dicts with `task` (benign task text) and `response`
    (the base model's own completion)."""
    lang = get_language_config(lang).canonical
    cfg = model.config
    L = len(model.model.layers)
    H = cfg.num_attention_heads
    D = getattr(cfg, "head_dim", None) or cfg.hidden_size // H
    N = cfg.intermediate_size

    attn_sum = torch.zeros(L, H, D, dtype=torch.float64)
    attn_xxt = torch.zeros(L, H, D, D, dtype=torch.float64)
    mlp_sum = torch.zeros(L, N, dtype=torch.float64)
    mlp_sq = torch.zeros(L, N, dtype=torch.float64)
    count = 0
    with ComponentRecorder(model, record_outputs=False) as rec:
        for ex in tqdm(list(examples), ncols=80, desc="benign stats"):
            prompt = benign_code_prompt(tokenizer, ex["task"], lang).strip()
            start, end = response_span(tokenizer, prompt, ex["response"])
            toks = tokenizer(prompt + ex["response"], return_tensors="pt", truncation=True,
                             max_length=max_len).to(device)
            end = min(end, toks.input_ids.shape[1])
            if end <= start:
                continue
            rec.clear()
            model(**toks)
            a = span_mean(rec.stacked(rec.attn_in), start, end).double().cpu().view(L, H, D)
            m = span_mean(rec.stacked(rec.mlp_in), start, end).double().cpu()
            attn_sum += a
            attn_xxt += torch.einsum("lhd,lhe->lhde", a, a)
            mlp_sum += m
            mlp_sq += m.pow(2)
            count += 1
    if count == 0:
        raise RuntimeError("No benign example produced a response span.")

    stats = {"meta": {"lang": lang, "num_samples": count, "top_k": top_k,
                      "num_layers": L, "num_heads": H, "head_dim": D, "num_neurons": N},
             "attn": {}, "mlp": {}}
    for i in range(L):
        head_mean = (attn_sum[i] / count)                                   # [H, D]
        bases, weights = [], []
        for h in range(H):
            cov = attn_xxt[i, h] / count - torch.outer(head_mean[h], head_mean[h])
            b, w = top_pca_from_cov(cov, top_k)
            bases.append(b)
            weights.append(w)
        stats["attn"][attn_key(i)] = {
            "head_mean": head_mean.float(),
            "head_pca_basis": torch.stack(bases),        # [H, K, D]
            "head_pca_weight": torch.stack(weights),     # [H, K]
        }
        mean = (mlp_sum[i] / count).float()
        second = (mlp_sq[i] / count).float()
        stats["mlp"][mlp_key(i)] = {
            "mean": mean,
            "second_moment": second,
            "variance": (second - mean.pow(2)).clamp_min(0.0),
            "normalized_second_moment": second / second.max().clamp_min(1e-12),
        }
    return stats
