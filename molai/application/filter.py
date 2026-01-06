from typing import List, Set
from rdkit import Chem
from rdkit.Chem import QED
import sascorer

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
    min_qed: float = 0.4,
    max_sa: float = 5.0,
):
    """
    Apply validity, novelty, SA, QED filters
    """
    filtered = []
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.error')
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
        if sascorer.calculateScore(mol) > max_sa:
            continue

        filtered.append(smi)

    return filtered

