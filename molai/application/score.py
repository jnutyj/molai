import torch
from typing import List
from molai.data.smiles import SmilesTokenizer
from rdkit import Chem
from rdkit.Chem import QED
import sascorer

########################################
# PROPERTY SCORING
########################################

@torch.no_grad()
def score_smiles_batch(
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





@torch.no_grad()
def score_smiles(
    smi: str,
    model,
    tokenizer: SmilesTokenizer,
    device: str = "cpu",
):
    """
    Score SMILES using a trained predictor
    """
    model.eval()
    model.to(device)



    
    ids = tokenizer.encode(smi)
    x = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)
    pred = model(x) ### TODO: need to know which model or based on model type for scoring. 
    

    return pred




def compute_qed(smi:str):
    mol=Chem.MolFromSmiles(smi)
    return QED.qed(mol)


def compute_sa(smi:str):
    mol=Chem.MolFromSmiles(smi)
    return sascorer.calculateScore(mol)


