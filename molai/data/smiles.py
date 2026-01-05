import re
from rdkit import Chem
import random
from typing import List, Dict

### All SMILES text handling
# tokenization
# detokenization
# vocabulary building
# SMILES randomization(augmentation)

######################################
# SMILES tokenization
#####################################


# Regex adapted from DeepSMILES / ChemBERTa
SMILES_REGEX = (
    r"(\[[^\]]+]|"     # atom in brackets
    r"Br|Cl|"          # two-character atoms
    r"B|C|N|O|P|S|F|I|"# single-character atoms
    r"b|c|n|o|s|p|"    # aromatic atoms
    r"\(|\)|\.|=|#|-|\+|\\|/|:|~|@|\?|>|\*|\$|"
    r"%\d{2}|"         # ring numbers >9
    r"\d)"             # ring numbers
)

_token_pattern = re.compile(SMILES_REGEX)


class SmilesTokenizer:
    """
    SMILES tokenizer with vocabulary management

    """
    def __init__(self):
        self.special_tokens = ["<pad>", "<bos>", "<eos>", "<unk>"]
        self.token_to_idx: Dict[str, int] = {}
        self.idx_to_token: Dict[int, str] = {}



    def tokenize(self, smiles: str) -> List[str]:
        return _token_pattern.findall(smiles)

    def build_vocab(self,smiles_list: List[str]):
        tokens=set()
        for smi in smiles_list:
            tokens.update(self.tokenize(smi))


        vocab = self.special_tokens + sorted(tokens)
        self.token_to_idx = {tok: i for i,tok in enumerate(vocab)}
        self.idx_to_token = {i: tok for tok, i in self.token_to_idx.items()}

    def encode(
            self, 
            smiles: str, 
            max_length: int,
            padding: bool = True,
            truncation: bool = True,
            add_special_tokens: bool = True) -> List[int]:
        
        tokens = self.tokenize(smiles)
        ids = [self.token_to_idx.get(t, self.token_to_idx["<unk>"]) for t in tokens]
        if add_special_tokens:
            ids = [self.token_to_idx["<bos>"]] + ids + [self.token_to_idx["<eos>"]]

        if truncation:
            ids = ids[:max_length]

        if padding and len(ids) < max_length:
            ids = ids + [self.token_to_idx["<pad>"]] * (max_length - len(ids))


        return ids

    
    def decode(self, ids: List[int], remove_special_tokens: bool = True) -> str:
        tokens = []
        for i in ids:
            tok = self.idx_to_token.get(i,"")
            if remove_special_tokens and tok in self.special_tokens:
                continue
            tokens.append(tok)
        return "".join(tokens)

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_idx)


#######################################
# SMILES augmentation
#######################################

def randomize_smiles(smiles: str, num_tries: int = 10) -> str:
    """
    Generate a randomized SMILES string NOT a list
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles

    for _ in range(num_tries):
        try:
            return Chem.MolToSmiles(
                mol,
                canonical=False,
                doRandom=True
            )
        except Exception:
            continue

    return smiles




#print(randomize_smiles("CCOCN"))
