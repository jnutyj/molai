import torch
import pickle
from typing import Callable

from molai.training.rl import policy_gradient_step
from molai.data.smiles import SmilesTokenizer
from molai.models.lstm import SmilesLSTMGenerator
from molai.application.filter import is_valid_smiles
from molai.application.score import score_smiles,compute_qed, compute_sa





############################################
# Sampling helper (pipeline-level)
############################################

def sample_smiles_with_logprobs(
    model,
    tokenizer,
    device,
    max_len=120,
    temperature= 1.0,
):
    model.eval()

    x = torch.tensor([[tokenizer.token_to_idx["<bos>"]]]).to(device)
    hidden = None

    log_probs = []
    token_ids = []

    for _ in range(max_len):
        logits, hidden = model(x, hidden)
        logits = logits[:, -1, :] / temperature

        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)

        idx = dist.sample()
        log_prob = dist.log_prob(idx)

        log_probs.append(log_prob)
        token_ids.append(idx.item())

        if idx.item() == tokenizer.token_to_idx["<eos>"]:
            break

        x = idx.unsqueeze(0).unsqueeze(0)

    smiles = tokenizer.decode(token_ids)
    return smiles, torch.stack(log_probs)

############################################
# example for reward function
############################################
def reward_fn(smiles_list, predictor):
    rewards = []
    for smi in smiles_list:
        if not is_valid_smiles(smi):
            rewards.append(-1.0)
            continue
        pred = score_smiles(predictor,smi) ### TODO: test it and make it correct
        qed = compute_qed(smi)
        sa = compute_sa(smi)
        reward=pred+0.5*qed-0.2*sa
        rewards.append(reward)
    return torch.tensor(rewards, dtype=torch.float)



############################################
# RL Optimization Pipeline
############################################

def rl_optimize(
    generator_ckpt: str,
    tokenizer_ckpt: str,
    reward_fn: Callable[[list], torch.Tensor],
    output_ckpt: str,
    device="cpu",
    epochs=10,
    batch_size=16,
    lr=1e-4,
    temperature=1.0,
):
    # ------------------
    # Load tokenizer
    # ------------------
    with open(tokenizer_ckpt, "rb") as f:
        tokenizer: SmilesTokenizer = pickle.load(f)

    # ------------------
    # Load generator
    # ------------------
    model = SmilesLSTMGenerator(tokenizer.vocab_size).to(device)
    model.load_state_dict(torch.load(generator_ckpt, map_location=device))

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # ------------------
    # RL Loop
    # ------------------
    for epoch in range(epochs):
        smiles_batch = []
        logprob_batch = []

        for _ in range(batch_size):
            smi, log_probs = sample_smiles_with_logprobs(
                model,
                tokenizer,
                device,
                temperature=temperature,
            )
            smiles_batch.append(smi)
            logprob_batch.append(log_probs)

        rewards = reward_fn(smiles_batch).to(device)

        # Pad log_probs
        max_len = max(lp.size(0) for lp in logprob_batch)
        padded = torch.zeros(len(logprob_batch), max_len).to(device)

        for i, lp in enumerate(logprob_batch):
            padded[i, : lp.size(0)] = lp

        loss = policy_gradient_step(
            log_probs=padded,
            rewards=rewards,
            optimizer=optimizer,
            baseline=rewards.mean().item(),
        )

        print(f"Epoch {epoch+1:02d} | RL loss: {loss:.4f}")

    torch.save(model.state_dict(), output_ckpt)
    print(f"RL generator saved to {output_ckpt}")

