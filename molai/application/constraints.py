from typing import List
from rdkit import Chem
from rdkit.Chem import Descriptors


########################################
# DESIGN CONSTRAINTS
########################################

def apply_constraints(
    smiles: List[str],
    min_mw: float = None,
    max_mw: float = None,
    max_logp: float = None,
    scaffold: str = None,
):
    """
    Apply physicochemical and scaffold constraints
    """
    constrained = []

    ref_scaffold = None
    if scaffold is not None:
        ref_scaffold = Chem.MolFromSmiles(scaffold)

    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue

        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)

        if min_mw and mw < min_mw:
            continue
        if max_mw and mw > max_mw:
            continue
        if max_logp and logp > max_logp:
            continue

        if ref_scaffold:
            if not mol.HasSubstructMatch(ref_scaffold):
                continue

        constrained.append(smi)

    return constrained

