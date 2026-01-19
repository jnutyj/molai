#!/usr/bin/env python3
import sys
from pathlib import Path

# add project root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import pickle
import pandas as pd
from molai.data.smiles import SmilesTokenizer
from molai.data.dataset import SmilesRegressionDataset, collate_smiles
from molai.training.supervised import train_lstm_predictor
from molai.models.lstm import SmilesLSTMPredictor
import torch


df = pd.read_csv("carbonic.csv")
smiles = df["SMILES"].tolist()
targets = df["pIC50"].tolist()

tokenizer = SmilesTokenizer()
tokenizer.build_vocab(smiles)

dataset = SmilesRegressionDataset(
    smiles,
    targets,
    tokenizer,
    augment=True,
    max_length=120
)

# loader = DataLoader(
#     dataset,
#     batch_size = 32,
#     shuffle=True,
#     collate_fn = lambda b: collate_smiles(b,pad_idx=tokenizer.token_to_idx['<pad>'])
# )



device = "cpu"
#model = SmilesLSTMGenerator(vocab_size=tokenizer.vocab_size, embed_dim=256, hidden_dim=512).to(device)
model = SmilesLSTMPredictor(vocab_size=tokenizer.vocab_size, embed_dim=256, hidden_dim=512).to(device)


train_lstm_predictor(model, dataset,epochs=20, device=device)


torch.save(model.state_dict(), "qsar_lstm.pt")
with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)
