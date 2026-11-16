"""Inference-time steering hooks for the knowledge-injection baselines (Appendix B.2).

* RSS (residual-stream steering): add gamma * d_l to the output of decoder layer l,
  where d_l is the secure-vs-vulnerable mean difference of the layer's residual
  stream.  The same hook with a single layer and gamma = 1 implements the
  SCS-Code baseline (vector from `baselines/scs_code/build_scs_vector.py`).
* CLS (component-level steering): add gamma * s_{l,b} to the output of the attention
  block (`o_proj`) and the MLP block (`down_proj`) of every layer, using the same
  component-level vectors that USKE converts into weight edits.

Both hooks act on every token position (prompt and generation).  In our experiments the
vectors are added as extracted (mean differences, not unit-normalized) and `gamma` is the
searched strength; `normalize=True` switches to unit vectors.
"""

from __future__ import annotations

import re

import torch

from .activations import attn_key, decoder_layers, mlp_key


def parse_layers(spec: str, num_layers: int) -> list[int]:
    """'all' | '12' | '0-8' | '0,4,8-16' (ranges are half-open like Python slices)."""
    spec = str(spec or "all").strip().lower()
    if spec == "all":
        return list(range(num_layers))
    out = set()
    for part in re.split(r"[,\s]+", spec):
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.update(range(max(0, int(a)), min(num_layers, int(b))))
        else:
            out.add(int(part))
    return sorted(i for i in out if 0 <= i < num_layers)


class _AddVector:
    def __init__(self, vec: torch.Tensor, scale: float):
        self.vec, self.scale = vec, float(scale)

    def __call__(self, module, inputs, output):
        if isinstance(output, tuple):
            hs = output[0]
            return (hs + (self.scale * self.vec).to(hs.device, hs.dtype),) + tuple(output[1:])
        return output + (self.scale * self.vec).to(output.device, output.dtype)


def _maybe_unit(vec: torch.Tensor, normalize: bool) -> torch.Tensor:
    vec = vec.float()
    return vec / vec.norm(p=2).clamp_min(1e-12) if normalize else vec


def register_residual_steering(model, directions: dict, gamma: float, layers_spec: str = "all",
                               normalize: bool = False) -> list:
    """`directions['post']['direction']['model.layers[i]']` -> per-layer residual vectors."""
    layers = decoder_layers(model)
    handles = []
    for i in parse_layers(layers_spec, len(layers)):
        vec = _maybe_unit(directions["post"]["direction"][f"model.layers[{i}]"], normalize)
        handles.append(layers[i].register_forward_hook(_AddVector(vec, gamma)))
    return handles


def register_component_steering(model, directions: dict, gamma: float, layers_spec: str = "all",
                                normalize: bool = False) -> list:
    """`directions['post']['direction'][attn_key(i) | mlp_key(i)]` -> block-output vectors."""
    layers = decoder_layers(model)
    handles = []
    for i in parse_layers(layers_spec, len(layers)):
        for key, module in ((attn_key(i), layers[i].self_attn.o_proj), (mlp_key(i), layers[i].mlp.down_proj)):
            vec = _maybe_unit(directions["post"]["direction"][key], normalize)
            handles.append(module.register_forward_hook(_AddVector(vec, gamma)))
    return handles


def remove_hooks(handles: list) -> None:
    for h in handles:
        h.remove()
