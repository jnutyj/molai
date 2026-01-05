import torch
import torch.nn as nn
import torch.nn.functional as F


#############################
# Latent predictor
#############################



class LatentPredictor(nn.Module):
    """
    Predict property(pIC50, etc.) from VAE latent vector
    """

    def __init__(self, latent_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(latent_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim,1)

    def forward(self,z):
        x = F.relu(self.fc1(z))
        x = F.relu(self.fc2(x))
        return self.fc_out(x).squeeze(-1)


    
