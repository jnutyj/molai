import torch
from typing import List
from molai.data.smiles import SmilesTokenizer

#############################
# SMILES generation
#############################

@torch.no_grad()
def sample_smiles_lstm(
        model,
        tokenizer: SmilesTokenizer,
        num_samples: int = 100,
        max_length: int = 100,
        temperature: float = 1.0,
        device: str = "cpu",
) -> List[st]:
    """
    Sample SMILES strings from a trained generative model
    """

    model.eval()
    model.to(device)

    bos = tokenizer.token_to_idx["<bos>"]
    eos = tokenizer.token_to_idx["<eos>"]

    smiles_list = []

    for _ in range(num_samples):
        ids = [bos]

        hidden = None

        for _ in range(max_length):
            x = torch.tensor([ids], dtype=torch.long).to(device)

            logits, hidden = model(x, hidden)
            logits = logits[:, -1, :] / temperature

            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, 1).item()

            if next_id == eos:
                break

            ids.append(next_id)

        smi = tokenizer.decode(ids)
        smiles_list.append(smi)

    return smiles_list
