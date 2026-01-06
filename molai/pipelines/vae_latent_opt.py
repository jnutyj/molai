# molai/pipelines/vae_latent_opt.py

import torch
import numpy as np
from typing import List, Callable
from rdkit import Chem


########################################
# Utilities
########################################

def is_valid_smiles(smiles: str) -> bool:
    return Chem.MolFromSmiles(smiles) is not None


########################################
# Latent optimization pipeline
########################################

def run_vae_latent_optimization(
    vae,
    predictor: Callable[[List[str]], np.ndarray],
    tokenizer,
    seed_smiles: List[str],
    device: str = "cpu",
    steps: int = 50,
    step_size: float = 0.1,
    maximize: bool = True,
):
    """
    Optimize latent vectors of a SMILES VAE to improve predicted property.

    Parameters
    ----------
    vae : trained VAE model
    predictor : callable
        Function mapping List[str] -> np.ndarray (property scores)
    tokenizer : SmilesTokenizer
    seed_smiles : list of SMILES strings
    steps : int
        Number of latent optimization steps
    step_size : float
        Latent gradient step size
    maximize : bool
        Whether to maximize or minimize property
    """

    vae.eval()
    vae.to(device)

    # ----------------------------------
    # Encode SMILES → latent z
    # ----------------------------------
    input_ids = [
        torch.tensor(tokenizer.encode(smi), dtype=torch.long)
        for smi in seed_smiles
    ]

    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids,
        batch_first=True,
        padding_value=tokenizer.token_to_idx["<pad>"],
    ).to(device)

    with torch.no_grad():
        z = vae.encode(input_ids)

    z = z.clone().detach().requires_grad_(True)

    optimizer = torch.optim.Adam([z], lr=step_size)

    history = []

    # ----------------------------------
    # Latent optimization loop
    # ----------------------------------
    for step in range(steps):
        optimizer.zero_grad()

        decoded = vae.decode(z)
        valid_mask = [is_valid_smiles(s) for s in decoded]

        if not any(valid_mask):
            print(f"Step {step:02d} | No valid molecules")
            continue

        valid_smiles = [s for s, v in zip(decoded, valid_mask) if v]
        scores = predictor(valid_smiles)

        scores = torch.tensor(scores, dtype=torch.float, device=device)
        reward = scores.mean()

        loss = -reward if maximize else reward
        loss.backward()
        optimizer.step()

        history.append({
            "step": step,
            "mean_score": reward.item(),
            "num_valid": len(valid_smiles),
        })

        print(
            f"Step {step:02d} | "
            f"Score: {reward.item():.3f} | "
            f"Valid: {len(valid_smiles)}"
        )

    # ----------------------------------
    # Final decoding
    # ----------------------------------
    final_smiles = vae.decode(z.detach())
    final_smiles = list(set(
        s for s in final_smiles if is_valid_smiles(s)
    ))

    return final_smiles, history

