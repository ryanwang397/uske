"""Batched greedy generation helpers.

* `vllm_generate` is used for plain checkpoints (base, PKI prompt, edited models
  exported with `export_edited_model`).
* `hf_generate` is used when forward hooks must stay active during decoding
  (the RSS / CLS steering baselines).
"""

from __future__ import annotations

import torch


def vllm_generate(model_path: str, prompts: list[str], max_new_tokens: int, seed: int = 20,
                  tensor_parallel_size: int = 1, max_model_len: int | None = None,
                  stop: list[str] | None = None) -> list[str]:
    from vllm import LLM, SamplingParams

    llm = LLM(model=model_path, tensor_parallel_size=tensor_parallel_size, dtype="bfloat16",
              seed=seed, max_model_len=max_model_len)
    params = SamplingParams(n=1, temperature=0.0, top_p=1.0, top_k=-1, max_tokens=max_new_tokens,
                            stop=stop or None)
    outputs = llm.generate(prompts, params)
    return [o.outputs[0].text for o in outputs]


@torch.no_grad()
def hf_generate(model, tokenizer, prompts: list[str], max_new_tokens: int, batch_size: int = 8) -> list[str]:
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    device = next(model.parameters()).device
    texts = []
    for b in range(0, len(prompts), batch_size):
        batch = tokenizer(prompts[b:b + batch_size], return_tensors="pt", padding=True).to(device)
        out = model.generate(**batch, do_sample=False, max_new_tokens=max_new_tokens,
                             pad_token_id=tokenizer.pad_token_id)
        new_tokens = out[:, batch.input_ids.shape[1]:]
        texts.extend(tokenizer.batch_decode(new_tokens, skip_special_tokens=True))
    return texts


def truncate_at_stop(text: str, stops: list[str] | None) -> str:
    cut = len(text)
    for s in stops or []:
        i = text.find(s)
        if 0 <= i < cut:
            cut = i
    return text[:cut]


def load_hf_model(model_path: str, device: str = "cuda"):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16).to(device).eval()
    return model, tok
