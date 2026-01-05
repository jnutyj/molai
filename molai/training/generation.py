import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Optional

from molai.data.dataset import SmilesRegressionDataset, collate_smiles



#############################
# LSTM generator pretraining
#############################

def train_lstm_generator(
    model: nn.Module,
    dataset: SmilesRegressionDataset,
    pad_idx: int = 0,
    batch_size: int = 64,
    lr: float = 3e-4,
    epochs: int = 20,
    device: str = "cpu"
):
    """
    Pretrain LSTM generator with teacher forcing
    """
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_smiles(b, pad_idx)
    )

    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_idx)




    for epoch in range(epochs):
        model.train()
        total_loss=0
        for batch in loader:
            x = batch["input_ids"].to(device)
            y = x.clone() # predict next token
            optimizer.zero_grad()
            logits = model(x)
            loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1:02d} | Generator Loss: {avg_loss:.4f}")
        
    return model


#################################
# VAE pretraining
#################################
def train_vae(
        model: nn.Module,
        dataset: SmilesRegressionDataset,
        pad_idx: int = 0,
        batch_size: int = 64,
        lr: float = 3e-4,
        epochs: int = 20,
        device: str = "cpu",
        beta: float = 1.0

):
    """
    Train SMILES VAE
    """
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_smiles(b, pad_idx)
    )
    
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_idx)


    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in loader:
            x = batch["input_ids"].to(device)
            optimizer.zero_grad()
            logits, mu,logvar = model(x)
            recon_loss = loss_fn(logits.view(-1,logits.size(-1)),x.view(-1))
            kl_loss = -0.5 * torch.mean(1+logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss+ beta* kl_loss
            loss.backward()
            optimizer.step()
            total_loss+=loss.item()
        avg_loss = total_loss/len(loader)
        print(f"Epoch {epoch+1:02d} | VAE Loss: {avg_loss:.4f}")
    return model


