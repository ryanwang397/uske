"""Build, save, load and apply sparse rank-one component edits (Section 3.1, 3.3).

An *edit pack* stores compact factors instead of dense weight deltas:

    attn: {(layer, head):   {"lambda", "k_hat"[head_dim], "g_sec", "r_util"}}
    mlp:  {(layer, neuron): {"lambda", "g_sec", "r_util"}}
    meta: {"vhat_attn": {layer: s_hat}, "vhat_mlp": {layer: s_hat}, hyper-parameters}

and is materialized as
    o_proj.weight[:, head block] += lambda * s_hat k_hat^T        (attention head)
    down_proj.weight[:, neuron]  += lambda * s_hat                 (MLP neuron)
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass

import torch
from tqdm import tqdm

from .activations import attn_key, decoder_layers, mlp_key
from .allocation import (attn_utility_risk, choose_attn_key, choose_mlp_sign, closed_form_lambda,
                         mlp_utility_risk, security_gain)


@dataclass
class EditConfig:
    rho_attn: float = -1.0        # <= 0 disables attention-head edits
    rho_mlp: float = -1.0         # <= 0 disables MLP-neuron edits
    alpha: float = 0.90           # Elastic-Net sparsity (0 = L2-only, ->1 = L1-dominant)
    beta: float = 0.15            # utility-regularization strength (0 = target-only / Steer2Edit allocation)
    key_mode: str = "conditional" # input direction: conditional | mean | random  (ablation)
    pca_top_k: int = 16           # number of principal directions used for attention utility risk
    seed: int = 20                # only used by key_mode=random

    def name(self) -> str:
        f = lambda x: (f"{x:.4f}").rstrip("0").rstrip(".") or "0"
        return (f"rank1__ra{f(self.rho_attn)}_rm{f(self.rho_mlp)}_a{f(self.alpha)}_b{f(self.beta)}"
                f"_key-{self.key_mode}_K{self.pca_top_k}.pt")


def _unit(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(p=2).clamp_min(1e-12)


@torch.no_grad()
def build_edit_pack(model, directions: dict, benign_stats: dict | None, cfg: EditConfig) -> dict:
    """Score every attention head and MLP neuron and allocate edit magnitudes in closed form."""
    layers = decoder_layers(model)
    mcfg = model.config
    H = mcfg.num_attention_heads
    D = getattr(mcfg, "head_dim", None) or mcfg.hidden_size // H
    N = mcfg.intermediate_size
    use_risk = cfg.beta > 0 and benign_stats is not None
    gen = torch.Generator().manual_seed(cfg.seed) if cfg.key_mode == "random" else None

    attn_factors, mlp_factors, vhat_attn, vhat_mlp = {}, {}, {}, {}
    for i in tqdm(range(len(layers)), ncols=80, desc="allocating edits"):
        ak, mk = attn_key(i), mlp_key(i)
        s_attn = _unit(directions["post"]["direction"][ak].float())
        s_mlp = _unit(directions["post"]["direction"][mk].float())
        vhat_attn[i], vhat_mlp[i] = s_attn, s_mlp
        z_attn = 0.5 * (directions["pre"]["pos_mean"][ak] + directions["pre"]["neg_mean"][ak]).float()
        z_mlp = 0.5 * (directions["pre"]["pos_mean"][mk] + directions["pre"]["neg_mean"][mk]).float()
        a_stats = benign_stats["attn"].get(ak) if use_risk else None
        m_stats = benign_stats["mlp"].get(mk) if use_risk else None

        if cfg.rho_attn > 0:
            Wo = layers[i].self_attn.o_proj.weight.data.float().cpu()          # [hidden, H*D]
            z_heads = z_attn.view(H, D)
            for h in range(H):
                block = Wo[:, h * D:(h + 1) * D]                                # [hidden, D]
                g = security_gain(s_attn, block @ z_heads[h])
                k_hat = choose_attn_key(cfg.key_mode, block.T @ s_attn, z_heads[h], gen)
                if k_hat is None:
                    continue
                r = attn_utility_risk(a_stats["head_pca_basis"][h], a_stats["head_pca_weight"][h],
                                      k_hat, cfg.pca_top_k) if a_stats else 0.0
                lam = closed_form_lambda(g.view(1), torch.tensor([r]), cfg.rho_attn, cfg.alpha, cfg.beta).item()
                if lam != 0.0:
                    attn_factors[(i, h)] = {"lambda": lam, "k_hat": k_hat, "g_sec": float(g), "r_util": r}

        if cfg.rho_mlp > 0:
            Wd = layers[i].mlp.down_proj.weight.data.float().cpu()             # [hidden, N]
            g_vec = torch.nn.functional.cosine_similarity(s_mlp.unsqueeze(1), Wd * z_mlp.unsqueeze(0), dim=0)
            r_vec = mlp_utility_risk(m_stats, N) if use_risk else torch.zeros(N)
            lam_vec = closed_form_lambda(g_vec, r_vec, cfg.rho_mlp, cfg.alpha, cfg.beta)
            cond = Wd.T @ s_mlp                                                 # [N]
            for n in torch.nonzero(lam_vec != 0.0).flatten().tolist():
                sign = choose_mlp_sign(cfg.key_mode, float(cond[n]), float(z_mlp[n]), gen)
                mlp_factors[(i, n)] = {"lambda": float(lam_vec[n]) * sign, "g_sec": float(g_vec[n]),
                                       "r_util": float(r_vec[n])}

    return {"attn": attn_factors, "mlp": mlp_factors,
            "meta": {"vhat_attn": vhat_attn, "vhat_mlp": vhat_mlp, "num_heads": H, "head_dim": D,
                     "hidden_size": mcfg.hidden_size, "utility_risk_used": use_risk, **asdict(cfg)}}


@torch.no_grad()
def apply_edit_pack(model, pack: dict) -> None:
    """Merge the rank-one edits into the model weights in place."""
    layers = decoder_layers(model)
    D = int(pack["meta"]["head_dim"])
    for (i, h), f in pack["attn"].items():
        W = layers[i].self_attn.o_proj.weight
        delta = f["lambda"] * torch.outer(pack["meta"]["vhat_attn"][i], f["k_hat"])
        W.data[:, h * D:(h + 1) * D].add_(delta.to(device=W.device, dtype=W.dtype))
    for (i, n), f in pack["mlp"].items():
        W = layers[i].mlp.down_proj.weight
        W.data[:, n].add_((f["lambda"] * pack["meta"]["vhat_mlp"][i]).to(device=W.device, dtype=W.dtype))


def save_edit_pack(pack: dict, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(pack, path)


def load_edit_pack(path: str) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def summarize_pack(pack: dict) -> dict:
    per_layer = {}
    for (i, _), f in pack["attn"].items():
        per_layer.setdefault(i, {"heads": 0, "neurons": 0})["heads"] += 1
    for (i, _), f in pack["mlp"].items():
        per_layer.setdefault(i, {"heads": 0, "neurons": 0})["neurons"] += 1
    return {"num_heads": len(pack["attn"]), "num_neurons": len(pack["mlp"]),
            "per_layer": dict(sorted(per_layer.items()))}


def export_edited_model(model_id: str, pack: dict, out_dir: str, dtype=torch.bfloat16) -> str:
    """Load the base checkpoint on CPU, apply the edits and save a standard HF checkpoint
    (loadable by transformers or vLLM with no inference-time intervention)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, device_map="cpu",
                                                 low_cpu_mem_usage=True)
    apply_edit_pack(model, pack)
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir, safe_serialization=True)
    AutoTokenizer.from_pretrained(model_id).save_pretrained(out_dir)
    return out_dir
