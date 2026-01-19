import torch
import numpy as np
from typing import List, Callable
#from molai.application.filter import is_valid_smiles
from molai.models.latent import LatentPredictor 
from molai.models.vae import SmilesVAE
from molai.data.smiles import SmilesTokenizer



########################################
# Latent optimization pipeline
########################################

# def run_vae_latent_optimization(
#     vae,
#     #latent_predictor: LatentPredictor,
#     predictor
#     tokenizer,
#     seed_smiles: List[str],
#     device: str = "cpu",
#     steps: int = 50,
#     step_size: float = 0.1,
#     maximize: bool = True,
# ):
#     """
#     Optimize latent vectors of a SMILES VAE to improve predicted property.

#     Parameters
#     ----------
#     vae : trained VAE model
#     latent_predictor : model 
#     tokenizer : SmilesTokenizer
#     seed_smiles : list of SMILES strings
#     steps : int
#         Number of latent optimization steps
#     step_size : float
#         Latent gradient step size
#     maximize : bool
#         Whether to maximize or minimize property
#     """

#     vae.eval()
#     vae.to(device)

#     # ----------------------------------
#     # Encode SMILES → latent z
#     # ----------------------------------
#     input_ids = [
#         torch.tensor(tokenizer.encode(smi), dtype=torch.long)
#         for smi in seed_smiles
#     ]

#     input_ids = torch.nn.utils.rnn.pad_sequence(
#         input_ids,
#         batch_first=True,
#         padding_value=tokenizer.token_to_idx["<pad>"],
#     ).to(device)

#     with torch.no_grad():
#         z = vae.encode(input_ids)

#     z = z.clone().detach().requires_grad_(True)

#     optimizer = torch.optim.Adam([z], lr=step_size)

#     history = []

#     # ----------------------------------
#     # Latent optimization loop
#     # ----------------------------------
#     for step in range(steps):
#         optimizer.zero_grad()

#         decoded = vae.decode(z)
#         valid_mask = [is_valid_smiles(s) for s in decoded]

#         if not any(valid_mask):
#             print(f"Step {step:02d} | No valid molecules")
#             continue

#         valid_smiles = [s for s, v in zip(decoded, valid_mask) if v]
#         scores = predictor(valid_smiles)

#         scores = torch.tensor(scores, dtype=torch.float, device=device)
#         reward = scores.mean()

#         loss = -reward if maximize else reward
#         loss.backward()
#         optimizer.step()

#         history.append({
#             "step": step,
#             "mean_score": reward.item(),
#             "num_valid": len(valid_smiles),
#         })

#         print(
#             f"Step {step:02d} | "
#             f"Score: {reward.item():.3f} | "
#             f"Valid: {len(valid_smiles)}"
#         )

#     # ----------------------------------
#     # Final decoding
#     # ----------------------------------
#     final_smiles = vae.decode(z.detach())
#     final_smiles = list(set(
#         s for s in final_smiles if is_valid_smiles(s)
#     ))

#     return final_smiles, history


#########################################
# latent optimization
#########################################

def optimize_latent(
    z_init: torch.Tensor,
    predictor: LatentPredictor,
    steps: int = 50,
    lr: float = 0.1,
    maximize: bool = True,
        ):

    """
    Perform gradient ascent/descent in latent space
    """
    z = z_init.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([z],lr=lr)

    for _ in range(steps):
        optimizer.zero_grad()
        score = predictor(z).mean()
        loss = -score if maximize else score
        loss.backward()
        optimizer.step()
    return z.detach()




######################################
# Decode latent to SMILES
######################################

def decode_latent(
        vae: SmilesVAE,
        tokenizer: SmilesTokenizer,
        z: torch.Tensor,
        max_len: int=120,
        temperature: float = 1.0,
        device: str = "cpu",
        ):

    """
    Autoregressive decoding from latent vector
    """

    vae.eval()
    batch_size = z.size(0)

    bos = tokenizer.token_to_idx["<bos>"]
    eos = tokenizer.token_to_idx["<eos>"]

    x = torch.full((batch_size,1),bos,dtype=torch.long,device=device)
    generated = [[] for _ in range(batch_size)]

    hidden = None
    for _ in range(max_len):
        emb = vae.embedding(x)
        z_expand = z.unsqueeze(1)
        decoder_input = torch.cat([emb,z_expand],dim=-1)

        out,hidden = vae.decoder_lstm(decoder_input,hidden)
        logits = vae.fc_out(out[:,-1,:])
        probs = torch.softmax(logits/temperature,dim=-1)

        idx = torch.multinomial(probs,1).squeeze(1)
        x=idx.unsqueeze(1)

        for i, token_id in enumerate(idx.tolist()):
            if token_id == eos:
                continue
                #break
            generated[i].append(token_id)
    smiles = [tokenizer.decode(seq) for seq in generated]
    return smiles

########################################
# End-to-end pipeline
########################################

def vae_latent_optimization_pipeline(
        smiles: List[str],
        vae: SmilesVAE,
        predictor: LatentPredictor,
        tokenizer: SmilesTokenizer,
        device: str = "cpu",
        opt_steps: int = 50,
        opt_lr: float = 0.1,
        ):

    """
    SMILES -> latent -> optimize -> decode
    """

    vae.to(device).eval()
    predictor.to(device).eval()

    # encode smiles to latent
    encoded=[]
    for smi in smiles:
        ids = tokenizer.encode(smi)
        x = torch.tensor(ids, dtype=torch.long,device=device).unsqueeze(0)
        with torch.no_grad():
            mu,_ =vae.encode(x)
        encoded.append(mu)
    z_init = torch.cat(encoded,dim=0)

    # optimize latent vectors
    z_opt = optimize_latent(
            z_init=z_init,
            predictor=predictor,
            steps=opt_steps,
            lr=opt_lr,
            )

    # decode optimized latent vectors

    optimized_smiles=decode_latent(
            vae=vae,
            tokenizer=tokenizer,
            z=z_opt,
            device=device,
            )

    return optimized_smiles



