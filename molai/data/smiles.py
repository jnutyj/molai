import re
from rdkit import Chem
import random

### This module is to process smiles from dataset before training

def tokenize_smiles(smiles):
    """
    Tokenize SMILES into meaningful chemical tokens
    """
    regex = (
        r"(?:\[[^\]]+\])"     # bracket atoms
        r"|Br|Cl"             # halogens
        r"|Si|Na|Ca|Li"       # multi-char elements (optional)
        r"|\d"                # ring numbers
        r"|=|#|-|\+"          # bonds
        r"|\(|\)"             # branches
        r"|\.|\/|\\"          # misc
        r"|[A-Za-z]"          # atoms
    )
    return re.findall(regex, smiles)


def randomize_smiles(smiles,n=5):
    """ Generate n randomized SMILES for a molecule
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []
    smi_list = []
    for _ in range(n):
        smi_list.append(Chem.MolToSmiles(mol,doRandom=True))
    return smi_list


def augment_smiles(smiles_list,n_aug=5):
    augmented = []
    for smi in smiles_list:
        #augmented.extend(randomize_smiles(smi,n_aug))
        augmented.append(randomize_smiles(smi,n_aug))
    return augmented

def is_valid_smiles(smiles: str) -> bool:
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.error')
    return Chem.MolFromSmiles(smiles) is not None

#smis=["CC(C)CN(C)Cc1cc(ccc1O)C(=O)c2cc(sc2)S(=O)(=O)N","CCO"]
#print(augment_smiles(smis))
