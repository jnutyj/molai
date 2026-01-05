import torch
import torch.nn as nn
import torch.nn.functional as F



##############################
# LSTM generator/ predictor
##############################


class SmilesLSTMGenerator(nn.Module):
    """
    LSTM Generator for SMILES sequences.
    output: next-token distribution(vocab_size)
    """

    def __init__(
            self,
            vocab_size: int,
            embed_dim: int = 256,
            hidden_dim: int = 512,
            num_layers: int = 2
    ):

        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim,hidden_dim,batch_first=True, num_layers=num_layers)
        self.fc = nn.Linear(hidden_dim,vocab_size)
        

        
    def forward(self, x):
        x = self.embedding(x)
        out,_ = self.lstm(x)
        logits = self.fc(out)
        return logits
        

class SmilesLSTMPredictor(nn.module):
    """
    LSTM for SMILES regression (predict property like pIC50)
    """
    def __init__(
            self,
            vocab_size: int,
            embed_dim: int=256,
            hidden_dim: int=512,
            num_layers: int=2
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, num_layers=num_layers)
        self.fc = nn.Linear(hidden_dim,1)

    def forward(self,x,lengths=None):
        """
        x:[batch_size,seq_len]
        lengths: optional lengths for packing
        """


        x=self.embedding(x)

        if lengths is not None:
            # pack padded sequences
            x = nn.utils.rnn.pack_padded_sequence(x,lengths.cpu,batch_first=True,enforce_sorted=False)
            out,_=self.lstm(x)
            out,_=nn.utils.rnn.pad_packed_sequence(out,batch_first=True)
        else:
            out,_=self.lstm(x)

        # take last timestep (can also do pooling)
        out = out[:,-1,:]
        out=self.fc(out)
        return out.squeeze(-1)
