import torch
import torch.nn as nn
import torch.optim as optim
from typing import Callable
import torch.nn.functional as F



########################################
# REINFORCEMENT LEARNING (Policy Gradient)
########################################

def rl_finetune_generator(
        model: nn.Module,
        reward_fn: Callable[[list], torch.Tensor], ## you need to provide reward function
        tokenizer,
        device: str = "cpu",
        max_len: int = 120,
        batch_size: int = 16,
        lr: float = 1e-4,
        epochs: int = 10,
        temperature: float = 1.0,
):
    """
    Fine-tune generator using REINFORCE with property reward

    TODO: IT DOES NOT return a model!!!
    """
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)


    for epoch in range(epochs):
        model.train()
        batch_rewards = []
        for _ in range(batch_size):
            # Sample sequences
            x = torch.tensor([[tokenizer.token_to_idx["<bos>"]]]).to(device)
            hidden = None
            log_probs = []
            seq_tokens = []

            for _ in range(max_len):
                logits, hidden = model(x, hidden) if hasattr(model, 'lstm') else (model(x), None)
                logits = logits[:, -1, :]
                probs = F.softmax(logits / temperature, dim=-1)
                dist = torch.distributions.Categorical(probs)
                idx = dist.sample()
                log_prob = dist.log_prob(idx)
                log_probs.append(log_prob)
                seq_tokens.append(idx.item())
                if idx.item() == tokenizer.token_to_idx["<eos>"]:
                    break
                x = idx.unsqueeze(0).unsqueeze(0)
            # compute reward
            smi = tokenizer.decode(seq_tokens)
            reward = reward_fn([smi])  # returns tensor
            batch_rewards.append((torch.stack(log_probs), reward))
                
        # Policy gradient update
        optimizer.zero_grad()
        loss = 0
        for log_probs, reward in batch_rewards:
            loss += -log_probs.sum() * reward
        loss /= batch_size
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch+1:02d} | RL Loss: {loss.item():.4f}")


    
