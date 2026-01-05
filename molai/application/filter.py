# molai/application/filter.py

from typing import List, Set
from rdkit import Chem
from rdkit.Chem import QED
from molai.utils.chemistry import sa_score


########################################
# MOLECULE FILTERS
########################################

def is_valid_smiles(smiles: str) -> bool:
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.error')
    return Chem.MolFromSmiles(smiles) is not None


def filter_smiles(
    smiles: List[str],
    training_set: Set[str] = None,
    min_qed: float = 0.0,
    max_sa: float = 10.0,
):
    """
    Apply validity, novelty, SA, QED filters
    """
    filtered = []

    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue

        # novelty
        if training_set is not None and smi in training_set:
            continue

        # QED
        if QED.qed(mol) < min_qed:
            continue

        # SA
        if sa_score(mol) > max_sa:
            continue

        filtered.append(smi)

    return filtered

