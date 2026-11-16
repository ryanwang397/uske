"""Security concept vectors from contrastive (vulnerable, secure) completions.

For every layer and block type (attention / MLP) we store the mean block output
on secure completions, on vulnerable completions, and their difference
(Eq. 1-2 of the paper).  We also store the mean block *input* (the activation
`z_e` feeding the write matrix), which the editing stage uses as the typical
editing context `z_bar` (Eq. 4).
"""

from __future__ import annotations

import torch
from tqdm import tqdm

from .activations import ComponentRecorder, attn_key, mlp_key, response_span, span_mean
from .languages import get_language_config
from .prompting import secure_code_prompt


@torch.no_grad()
def _collect(model, tokenizer, questions, responses, lang, template_mode, device, max_len):
    """Per-sample response-span means: four lists of [num_layers, dim] tensors."""
    attn_post, attn_pre, mlp_post, mlp_pre = [], [], [], []
    with ComponentRecorder(model) as rec:
        for q, resp in tqdm(list(zip(questions, responses)), ncols=80, desc=template_mode):
            prompt = secure_code_prompt(tokenizer, q, lang, template_mode).strip()
            start, end = response_span(tokenizer, prompt, resp)
            toks = tokenizer(prompt + resp, return_tensors="pt", truncation=True, max_length=max_len).to(device)
            end = min(end, toks.input_ids.shape[1])
            if end <= start:
                continue
            rec.clear()
            model(**toks)
            attn_post.append(span_mean(rec.stacked(rec.attn_out), start, end).float().cpu())
            attn_pre.append(span_mean(rec.stacked(rec.attn_in), start, end).float().cpu())
            mlp_post.append(span_mean(rec.stacked(rec.mlp_out), start, end).float().cpu())
            mlp_pre.append(span_mean(rec.stacked(rec.mlp_in), start, end).float().cpu())
    if not attn_post:
        raise ValueError("No valid (prompt, response) pairs produced a response span.")
    return [torch.stack(x).mean(0) for x in (attn_post, attn_pre, mlp_post, mlp_pre)]


@torch.no_grad()
def extract_directions(model, tokenizer, pairs, lang: str, device="cuda", max_len: int = 4096) -> dict:
    """`pairs`: iterable of dicts with `question`, `neg_response` (vulnerable, baseline
    template) and `pos_response` (secure, secure template)."""
    lang = get_language_config(lang).canonical
    questions = [p["question"] for p in pairs]
    pos = _collect(model, tokenizer, questions, [p["pos_response"] for p in pairs], lang, "secure", device, max_len)
    neg = _collect(model, tokenizer, questions, [p["neg_response"] for p in pairs], lang, "baseline", device, max_len)
    pos_attn_post, pos_attn_pre, pos_mlp_post, pos_mlp_pre = pos
    neg_attn_post, neg_attn_pre, neg_mlp_post, neg_mlp_pre = neg

    out = {"pre": {"pos_mean": {}, "neg_mean": {}, "direction": {}},
           "post": {"pos_mean": {}, "neg_mean": {}, "direction": {}},
           "meta": {"lang": lang, "num_pairs": len(questions)}}
    for i in range(pos_attn_post.shape[0]):
        for key, pp, npost, ppre, npre in (
            (attn_key(i), pos_attn_post[i], neg_attn_post[i], pos_attn_pre[i], neg_attn_pre[i]),
            (mlp_key(i), pos_mlp_post[i], neg_mlp_post[i], pos_mlp_pre[i], neg_mlp_pre[i]),
        ):
            out["post"]["pos_mean"][key], out["post"]["neg_mean"][key] = pp, npost
            out["post"]["direction"][key] = pp - npost
            out["pre"]["pos_mean"][key], out["pre"]["neg_mean"][key] = ppre, npre
            out["pre"]["direction"][key] = ppre - npre
    return out
