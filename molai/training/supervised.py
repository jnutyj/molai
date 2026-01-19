import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Optional

from molai.data.dataset import SmilesRegressionDataset, collate_smiles
from molai.models.latent import LatentPredictor 


def train_lstm_predictor(
        model: nn.Module,
        dataset: SmilesRegressionDataset,
        val_dataset: Optional[SmilesRegressionDataset] = None,
        pad_idx: int = 0,
        batch_size: int = 32,
        lr: float = 3e-4,
        epochs: int = 20,
        device: str = "cpu"
):
    """
    Train a SMILES LSTM predictor (regression)
    """

    train_loader = DataLoader(
        dataset,
        batch_size = batch_size,
        shuffle = True,
        collate_fn=lambda b: collate_smiles(b,pad_idx)
    )

    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size = batch_size,
            collate_fn = lambda b: collate_smiles(b, pad_idx)
        )

    model.to(device)
    optimizer = optim.Adam(model.parameters(),lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            x = batch["input_ids"].to(device)
            lengths = batch["lengths"].to(device)
            y = batch["targets"].to(device)


            optimizer.zero_grad()
            preds = model(x,lengths)
            loss = loss_fn(preds, y)
            loss.backward()
            optimizer.step()
            total_loss+=loss.item()



        avg_loss = total_loss/len(train_loader)
        print(f"Epoch {epoch+1:02d} | Train MSE: {avg_loss:.4f}")

        if val_dataset is not None:
            model.eval()
            val_loss=0
            with torch.no_grad():
                for batch in val_loader:
                    x = batch["input_ids"].to(device)
                    lengths = batch["lengths"].to(device)
                    y = batch["targets"].to(device)
                    preds = model(x,lengths)
                    val_loss += loss_fn(preds,y).item()
            val_loss /= len(val_loader)
            print(f"             Val MSE: {val_loss:.4f}")



    return model

def train_latent_predictor(
    latent_model: LatentPredictor,
    dataset: SmilesRegressionDataset,
    vae: nn.Module,
    pad_idx: int = 0,
    batch_size: int = 64,
    lr: float = 3e-4,
    epochs: int = 20,
    device: str = "cpu",
        ):

    
    train_loader = DataLoader(
        dataset,
        batch_size = batch_size,
        shuffle = True,
        collate_fn=lambda b: collate_smiles(b,pad_idx)
    )

    vae.to(device).eval()
    latent_model.to(device)
    optimizer = torch.optim.Adam(latent_model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    
    for epoch in range(epochs):
        latent_model.train()
        total_loss = 0.0
        for batch in train_loader:
            x=batch["input_ids"].to(device)
            y=batch["targets"].to(device)
            with torch.no_grad():
                mu, _ = vae.encode(x)
                z = mu  # deterministic latent, avoid noisy data

            preds = latent_model(z)
            loss = loss_fn(preds, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        avg_loss= total_loss/len(train_loader)
        print(f"[Latent] Epoch {epoch+1:03d} | MSE: {avg_loss:.4f}")

    return latent_model
