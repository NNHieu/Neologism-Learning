"""Training objectives for neologism learning (paper Sec 3, eq. 1-2)."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def apo_up_loss(
    logp_chosen: torch.Tensor,
    logp_rejected: torch.Tensor,
    ref_logp_chosen: torch.Tensor,
    ref_logp_rejected: torch.Tensor,
    beta: float = 0.1,
) -> torch.Tensor:
    """APO-up: a DPO variant with an extra term on the absolute chosen likelihood.

    L = -logσ(β[(logp_c-ref_c) - (logp_r-ref_r)])   # prefer chosen over rejected
        -logσ(β(logp_c - ref_c))                     # raise chosen likelihood
    """
    pref = (logp_chosen - ref_logp_chosen) - (logp_rejected - ref_logp_rejected)
    chosen_abs = logp_chosen - ref_logp_chosen
    loss = -F.logsigmoid(beta * pref) - F.logsigmoid(beta * chosen_abs)
    return loss.mean()


def hinge_norm_penalty(embedding: torch.Tensor, lam: float = 0.1) -> torch.Tensor:
    """λ·max(‖e‖ - 1, 0): keep the learned embedding norm around 1 (paper A.6)."""
    return lam * torch.clamp(embedding.norm() - 1.0, min=0.0)
