from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, MACCSkeys
import numpy as np
import pandas as pd
from typing import List


def maccs_fp(smiles: str) -> np.ndarray:
    """
    Generate MACCS fingerprint(166 bits)
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(166, dtype=int)

    fp = MACCSkeys.GenMACCSKeys(mol)
    arr = np.zeros((166,), dtype=int)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr





    
def morgan_fp(smiles: str, radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    """
    Generate Morgan fingerprint (ECFP) as binary numpy array
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_bits, dtype=int)

    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=int)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr





def dataframe_to_fingerprint_matrix(
        df: pd.DataFrame,
        smiles_col: str = "SMILES",
        fp_type: str = "morgan", **kwargs) -> pd.DataFrame:
    """
    Convert a DataFrame of SMILES to fingerprint features
    """
    feature_matrix = []
    for smi in df[smiles_col]:
        if fp_type.lower() == "maccs":
            feature_matrix.append(maccs_fp(smi))
        elif fp_type.lower() == "morgan":
            feature_matrix.append(morgan_fp(smi, **kwargs))
        else:
            raise ValueError(f"Unsupported fingerprint type: {fp_type}")

    return pd.DataFrame(feature_matrix)
