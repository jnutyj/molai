# molai/application/score.py

import torch
from typing import List
from molai.data.smiles import SmilesTokenizer


########################################
# PROPERTY SCORING
########################################

@torch.no_grad()
def score_smiles(
    smiles: List[str],
    model,
    tokenizer: SmilesTokenizer,
    device: str = "cpu",
):
    """
    Score SMILES using a trained predictor
    """
    model.eval()
    model.to(device)

    scores = []

    for smi in smiles:
        ids = tokenizer.encode(smi)
        x = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)
        pred = model(x) ### TODO: need to know which model or based on model type for scoring. 
        scores.append(float(pred.item()))

    return scores

