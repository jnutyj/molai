# molai/training/rl.py

import torch
import torch.nn.functional as F
from typing import Callable


def policy_gradient_step(
    log_probs: torch.Tensor,
    rewards: torch.Tensor,
    optimizer,
    baseline: float = 0.0,
):
    """
    Perform a single REINFORCE update step.

    Parameters
    ----------
    log_probs : Tensor
        Shape (batch_size, seq_len)
    rewards : Tensor
        Shape (batch_size,)
    optimizer : torch.optim.Optimizer
    baseline : float or Tensor
        Baseline to reduce variance
    """

    # Advantage
    advantage = rewards - baseline

    # Policy gradient loss
    loss = -(log_probs.sum(dim=1) * advantage).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()

