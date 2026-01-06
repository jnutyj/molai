import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Callable
from rdkit import Chem
import numpy as np


def rl_finetune_generator(
    model: nn.Module,
    reward_fn: Callable[[list], torch.Tensor],
    tokenizer,
    device: str = "cpu",
    max_len: int = 120,
    batch_size: int = 16,
    lr: float = 1e-4,
    epochs: int = 10,
    temperature: float = 1.0,
):
    """
    REINFORCE fine-tuning for SMILES generator

    reward_fn(smiles_list) -> torch.Tensor [batch]
    """

    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    pad_idx = tokenizer.token_to_idx["<pad>"]
    bos_idx = tokenizer.token_to_idx["<bos>"]
    eos_idx = tokenizer.token_to_idx["<eos>"]

    for epoch in range(epochs):
        model.train()

        trajectories = []
        rewards = []

        # -------------------------
        # Sample trajectories
        # -------------------------
        for _ in range(batch_size):
            x = torch.tensor([[bos_idx]], device=device)
            hidden = None
            log_probs = []
            token_ids = []

            for _ in range(max_len):
                logits, hidden = model(x, hidden)
                logits = logits[:, -1, :]
                probs = F.softmax(logits / temperature, dim=-1)

                dist = torch.distributions.Categorical(probs)
                idx = dist.sample()

                log_probs.append(dist.log_prob(idx))
                token_ids.append(idx.item())

                if idx.item() == eos_idx:
                    break

                x = idx.unsqueeze(0).unsqueeze(0)

            smiles = tokenizer.decode(token_ids)

            # validity check
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                reward = torch.tensor(0.0, device=device)
            else:
                reward = reward_fn([smiles]).to(device).squeeze()

            trajectories.append(torch.stack(log_probs))
            rewards.append(reward)

        rewards = torch.stack(rewards)

        # -------------------------
        # Baseline + normalization
        # -------------------------
        baseline = rewards.mean()
        advantages = rewards - baseline
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # -------------------------
        # Policy gradient update
        # -------------------------
        optimizer.zero_grad()
        loss = 0.0

        for log_probs, adv in zip(trajectories, advantages):
            loss += -log_probs.sum() * adv

        loss /= batch_size
        loss.backward()
        optimizer.step()

        print(
            f"Epoch {epoch+1:02d} | "
            f"RL Loss: {loss.item():.4f} | "
            f"Avg Reward: {rewards.mean().item():.3f}"
        )

    return model

