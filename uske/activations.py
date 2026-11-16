"""Forward hooks that record component inputs/outputs.

Components are the residual-stream-writing units of a decoder layer:
  * attention heads  -> inputs / outputs of `self_attn.o_proj`
  * MLP neurons      -> inputs / outputs of `mlp.down_proj`
"""

from __future__ import annotations

import torch


def attn_key(layer_idx: int) -> str:
    return f"model.layers[{layer_idx}].self_attn.o_proj"


def mlp_key(layer_idx: int) -> str:
    return f"model.layers[{layer_idx}].mlp.down_proj"


def decoder_layers(model):
    inner = getattr(model, "model", model)
    if not hasattr(inner, "layers"):
        raise RuntimeError("Unexpected model structure: expected `model.model.layers`.")
    return inner.layers


class ComponentRecorder:
    """Context manager that stores, per layer, the o_proj / down_proj inputs and outputs
    of the most recent forward pass (one sequence at a time)."""

    def __init__(self, model, record_outputs: bool = True):
        self.layers = decoder_layers(model)
        self.record_outputs = record_outputs
        self.attn_in, self.attn_out, self.mlp_in, self.mlp_out = {}, {}, {}, {}
        self._handles = []

    def _hook(self, store_in, store_out, idx):
        def fn(module, inputs, output):
            store_in[idx] = inputs[0].detach()
            if self.record_outputs:
                store_out[idx] = output.detach()
        return fn

    def __enter__(self):
        for i, layer in enumerate(self.layers):
            self._handles.append(layer.self_attn.o_proj.register_forward_hook(
                self._hook(self.attn_in, self.attn_out, i)))
            self._handles.append(layer.mlp.down_proj.register_forward_hook(
                self._hook(self.mlp_in, self.mlp_out, i)))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles = []

    def clear(self):
        for d in (self.attn_in, self.attn_out, self.mlp_in, self.mlp_out):
            d.clear()

    def stacked(self, store: dict) -> torch.Tensor:
        """[num_layers, batch, seq, dim] in layer order."""
        return torch.stack([store[i] for i in range(len(self.layers))], dim=0)


def span_mean(stacked: torch.Tensor, start: int, end: int) -> torch.Tensor:
    """Average over the response span.

    `start` is the number of prompt tokens and `end` the total length.  The
    hidden state at position t produces token t+1, so positions
    [start-1, end-1) are the ones that write the response tokens.
    Returns [num_layers, dim] for a batch of one sequence.
    """
    return stacked[:, :, start - 1:end - 1, :].mean(dim=2).squeeze(1)


def response_span(tokenizer, prompt: str, response: str) -> tuple[int, int]:
    start = len(tokenizer(prompt).input_ids)
    end = len(tokenizer(prompt + response).input_ids)
    return start, end
