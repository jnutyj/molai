import torch
import torch.nn as nn
import torch.nn.functional as F


class SmilesVAE(nn.Module):
    """
    SMILES Variational Autoencoder
    """

    def __init__(
            self,
            vocab_size: int,
            embed_dim: int=256,
            hidden_dim: int=512,
            latent_dim: int=128,
            num_layers:int=2
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.encoder_lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, num_layers=num_layers)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        self.latent_dim=latent_dim
        self.decoder_lstm = nn.LSTM(embed_dim+latent_dim, hidden_dim, batch_first=True, num_layers=num_layers)
        self.fc_out = nn.Linear(hidden_dim,vocab_size)

    def encode(self,x):
        """
        Encode SMILES into latent space
        """

        x = self.embedding(x)
        _,(h_n,_) = self.encoder_lstm(x)
        h = h_n[-1] # take last layer hidden state
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu,logvar


    def reparameterize(self,mu,logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu+ eps* std


    def decode(self,z,x):
        """
        Decode latent + input sequence(teacher forcing)
        """

        x = self.embedding(x)
        z_expand = z.unsqueeze(1).expand(-1,x.size(1),-1)
        x_concat = torch.cat([x,z_expand],dim=-1)
        out,_ = self.decoder_lstm(x_concat)
        logits = self.fc_out(out)
        return logits

    def forward(self,x):
        mu,logvar = self.encode(x)
        z = self.reparameterize(mu,logvar)
        logits = self.decode(z,x)
        return logits, mu, logvar
