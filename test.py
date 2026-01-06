
####### test data
from torch.utils.data import DataLoader
from molai.data.smiles import SmilesTokenizer
from molai.data.dataset import SmilesRegressionDataset, collate_smiles

smiles = ["CCO", "CC(=O)O", "c1ccccc1"]
targets = [1.2, 0.8, 2.1]

tokenizer = SmilesTokenizer()
tokenizer.build_vocab(smiles)

dataset = SmilesRegressionDataset(
    smiles,
    targets,
    tokenizer,
    augment=True
)

loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True,
    collate_fn=lambda b: collate_smiles(b, tokenizer.token_to_idx["<pad>"])
)

batch = next(iter(loader))
print(batch["input_ids"].shape)
print(batch["targets"])


#### test features
from molai.features.descriptors import dataframe_to_descriptor_matrix
from molai.features.fingerprints import dataframe_to_fingerprint_matrix

df = pd.read_csv("carbonic.csv")

# Descriptors
desc_df = dataframe_to_descriptor_matrix(df)
print(desc_df.head())

# Morgan fingerprints
fp_df = dataframe_to_fingerprint_matrix(df, fp_type="morgan", radius=2, n_bits=1024)
print(fp_df.head())



########model
from molai.models.lstm import SmilesLSTMGenerator, SmilesLSTMPredictor
from molai.models.vae import SmilesVAE
from molai.models.latent import LatentPredictor

import torch

# Example vocab size
vocab_size = 50
batch_size = 4
seq_len = 20

x = torch.randint(0, vocab_size, (batch_size, seq_len))

# LSTM generator / predictor
gen = SmilesLSTMGenerator(vocab_size)
pred = SmilesLSTMPredictor(vocab_size)
print(gen(x).shape)  # [batch, seq_len, vocab_size]
print(pred(x).shape) # [batch]

# VAE + latent predictor
vae = SmilesVAE(vocab_size)
mu, logvar = vae.encode(x)
z = vae.reparameterize(mu, logvar)

latent_pred = LatentPredictor(latent_dim=z.size(-1))
print(latent_pred(z).shape)  # [batch]


### application

from molai.application import (
    sample_smiles,
    score_smiles,
    filter_smiles,
    apply_constraints,
)

smiles = sample_smiles(generator, tokenizer, 500)

smiles = filter_smiles(
    smiles,
    training_set=train_smiles,
    min_qed=0.4,
    max_sa=6.0,
)

smiles = apply_constraints(
    smiles,
    min_mw=250,
    max_mw=500,
    max_logp=5,
)

scores = score_smiles(smiles, predictor, tokenizer)




from molai.pipelines.generate_filter import generate_and_filter
from molai.models.lstm import SmilesLSTMGenerator
from molai.data.smiles import SmilesTokenizer
import pickle
import torch

# Load tokenizer
tokenizer = SmilesTokenizer()
tokenizer.build_vocab(train_smiles)

# Load generator
generator = SmilesLSTMGenerator(tokenizer.vocab_size)
generator.load_state_dict(torch.load("lstm_generator.pt"))

# Load predictor
with open("rf_qsar.pkl", "rb") as f:
    predictor = pickle.load(f)

# Run pipeline
results = generate_and_filter(
    generator=generator,
    tokenizer=tokenizer,
    predictor=predictor,
    reference_smiles=train_smiles,
    num_samples=2000,
    temperature=0.9,
)

for smi, score in results[:10]:
    print(smi, score)

