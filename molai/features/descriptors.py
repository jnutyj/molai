from rdkit import Chem
from rdkit.Chem import Descriptors
from typing import List, Dict
import pandas as pd


#####
# handles invalid SMILES
# easy to extend with custom descriptor lists
# output can be used for RandomForest, GradientBoosting, or neural networks



###############################
# list of common 2D descriptors
###############################


COMMON_DESCRIPTORS = [
    "MolWt",           # Molecular weight
    "MolLogP",         # Octanol-water partition coefficient
    "NumHDonors",      # Number of hydrogen donors
    "NumHAcceptors",   # Number of hydrogen acceptors
    "TPSA",            # Topological polar surface area
    "NumRotatableBonds",
    "RingCount",
    "FractionCSP3",
]


# Mapping from descriptor name → RDKit function
DESCRIPTOR_FUNCS: Dict[str, callable] = {
    "MolWt": Descriptors.MolWt,
    "MolLogP": Descriptors.MolLogP,
    "NumHDonors": Descriptors.NumHDonors,
    "NumHAcceptors": Descriptors.NumHAcceptors,
    "TPSA": Descriptors.TPSA,
    "NumRotatableBonds": Descriptors.NumRotatableBonds,
    "RingCount": Descriptors.RingCount,
    "FractionCSP3": Descriptors.FractionCSP3,
}




def mol_to_descriptors(smiles: str, descriptor_list: List[str] =COMMON_DESCRIPTORS) -> List[float]:
    """
    Conver a SMILES string to a list of 2D descriptors
    """


    mol = Chem.MolFromSmiles(smiles)
    if mol is None:

        return [float("nan")] * len(descriptor_list)

    features = [DESCRIPTOR_FUNCS[d](mol) for d in descriptor_list]
    return features




def dataframe_to_descriptor_matrix(df: pd.DataFrame, smiles_col: str = "SMILES", descriptor_list: List[str] = COMMON_DESCRIPTORS) -> pd.DataFrame:

    """
    Apply descriptor calculation to a DataFrame of SMILES
    """

    feature_matrix = []
    for smi in df[smiles_col]:
        feature_matrix.append(mol_to_descriptors(smi, descriptor_list))
        feature_df = pd.DataFrame(feature_matrix, columns=descriptor_list)

    return feature_df






