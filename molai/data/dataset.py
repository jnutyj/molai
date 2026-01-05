# molai/data/dataset.py

import torch
from torch.utils.data import Dataset
from typing import List, Optional
from molai.data.smiles import SmilesTokenizer, randomize_smiles


### bridge raw data to tensors
#holding SMILES + labels

#tokenizing SMILES

#padding sequences

#working with PyTorch DataLoader



class SmilesRegressionDataset(Dataset):
    """
    Dataset for SMILES → regression targets (e.g. pIC50)
    """

    def __init__(
        self,
        smiles: List[str],
        targets: List[float],
        tokenizer: SmilesTokenizer,
        augment: bool = False,
        max_length: Optional[int] = None,
    ):
        assert len(smiles) == len(targets)

        self.smiles = smiles
        self.targets = torch.tensor(targets, dtype=torch.float)
        self.tokenizer = tokenizer
        self.augment = augment
        self.max_length = max_length

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        smi = self.smiles[idx]

        if self.augment:
            smi = randomize_smiles(smi)

        token_ids = self.tokenizer.encode(smi)

        if self.max_length is not None:
            token_ids = token_ids[: self.max_length]

        return {
            "input_ids": torch.tensor(token_ids, dtype=torch.long),
            "target": self.targets[idx],
        }


########################################
# COLLATE FUNCTION (PADDING)
########################################

def collate_smiles(batch, pad_idx: int):
    """
    Pads variable-length SMILES sequences
    """
    input_ids = [item["input_ids"] for item in batch]
    targets = torch.stack([item["target"] for item in batch])

    lengths = torch.tensor([len(x) for x in input_ids])

    max_len = max(lengths)
    padded = torch.full(
        (len(input_ids), max_len),
        pad_idx,
        dtype=torch.long
    )

    for i, seq in enumerate(input_ids):
        padded[i, : len(seq)] = seq

    return {
        "input_ids": padded,
        "lengths": lengths,
        "targets": targets,
    }
