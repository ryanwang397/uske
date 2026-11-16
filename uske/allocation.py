"""Security gain, utility risk and the closed-form edit allocation (Section 3.2-3.3).

    lambda_e = sign(g_e) * [ |g_e| - beta * r_e - rho * alpha ]_+ / ( rho * (1 - alpha) )

`beta = 0` removes the utility-risk term and recovers the target-only
allocation of Steer2Edit (Appendix B.4).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def closed_form_lambda(g: torch.Tensor, r: torch.Tensor, rho: float, alpha: float, beta: float) -> torch.Tensor:
    """Vectorized soft-thresholding rule (Eq. 8). `rho <= 0` disables the component type."""
    if rho is None or rho <= 0:
        return torch.zeros_like(g)
    alpha = min(max(float(alpha), 0.0), 0.9999)
    threshold = rho * alpha + beta * r.abs()
    num = torch.clamp(g.abs() - threshold, min=0.0)
    return torch.sign(g) * num / (rho * (1.0 - alpha))


def security_gain(v_hat: torch.Tensor, write_vec: torch.Tensor) -> torch.Tensor:
    """cos(s_hat, W_e z_bar_e) for one component (Eq. 4)."""
    return F.cosine_similarity(v_hat, write_vec, dim=0)


def attn_utility_risk(head_basis: torch.Tensor, head_weight: torch.Tensor, k_hat: torch.Tensor,
                      top_k: int | None = None) -> float:
    """Variance-weighted overlap of the edit input direction with the benign PCA subspace (Eq. 5)."""
    if top_k is not None and top_k > 0:
        head_basis, head_weight = head_basis[:top_k], head_weight[:top_k]
    if head_basis.numel() == 0:
        return 0.0
    proj = head_basis.float() @ k_hat.float()
    return float((head_weight.float().clamp_min(0.0) * proj.pow(2)).sum())


def mlp_utility_risk(layer_stats: dict | None, num_neurons: int) -> torch.Tensor:
    """Layer-normalized second moment of benign neuron activations (Eq. 6)."""
    if not layer_stats:
        return torch.zeros(num_neurons)
    risk = layer_stats.get("normalized_second_moment")
    if risk is None:
        second = layer_stats["second_moment"]
        risk = second / second.max().clamp_min(1e-12)
    return risk.float()


def choose_attn_key(mode: str, conditional_key: torch.Tensor, x_mean_head: torch.Tensor,
                    generator: torch.Generator | None):
    """Input direction k_hat of a head edit.  `conditional` is the method (Eq. 3);
    `mean` and `random` are the ablation alternatives (Appendix I.2)."""
    if mode == "conditional":
        key = conditional_key
    elif mode == "mean":
        key = x_mean_head
    elif mode == "random":
        key = torch.randn(conditional_key.shape, generator=generator, device=conditional_key.device)
    else:
        raise ValueError(f"Unsupported key_mode: {mode}")
    norm = key.norm(p=2)
    return None if norm.item() == 0.0 else key / norm


def choose_mlp_sign(mode: str, conditional_scalar: float, x_mean_scalar: float,
                    generator: torch.Generator | None) -> float:
    """For an MLP neuron the input direction is a scalar orientation absorbed into lambda."""
    if mode == "conditional":
        return 1.0 if conditional_scalar >= 0.0 else -1.0
    if mode == "mean":
        val = x_mean_scalar if x_mean_scalar != 0.0 else conditional_scalar
        return 1.0 if val >= 0.0 else -1.0
    if mode == "random":
        return 1.0 if torch.randint(0, 2, (1,), generator=generator).item() == 1 else -1.0
    raise ValueError(f"Unsupported key_mode: {mode}")
