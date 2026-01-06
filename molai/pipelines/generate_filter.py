from typing import List, Dict
import numpy as np
from rdkit import Chem

from molai.application.generate import sample_smiles_lstm
from molai.application.filter import filter_smiles
from molai.application.score import score_smiles



########################################
# Generate -> filter -> score ranking pipeline
########################################


def generate_and_filter(
        generator,
        tokenizer,
        predictor,
        reference_smiles: List[str],
        num_sample: int = 1000,
        max_length: int = 120,
        temperature: float = 1.0,
        device: str = "cpu",
        filter_config: Dict = None,

):
    """
    End-to-end molecule generation pipeline

    Returns ranked list of molecules with scores
    """

    if filter_config is None:
        filter_config = {
            #"valid": True,
            #"unique": True,
            #"novel": True,
            "qed_min": 0.3,
            "sa_max": 6.0,
        }



        ##### 1) generate SMILES#####################

        raw_smiles = sample_smiles_lstm(
            model=generator,
            tokenizer=tokenizer,
            num_samples=num_samples,
            max_length=max_length,
            temperature=temperature,
            device=device,
        )

        ###### 2) filter molecules ##################

        
        filtered_smiles = filter_smiles(
            smiles_list = raw_smiles,
            reference_smiles=reference_smiles,
            **filter_config,
        )
        
        if len(filtered_smiles) ==0:
            return []


        ###### 3) score with predictor #############
        scores = score_smiles(
            smiles_list=filtered_smiles,
            predictor=predictor,
        )

        ###### 4) rank moelcules ###################
        ranked = sorted(
            zip(filtered_smiles, scores),
            key=lambda x: x[1],
            reverse=True,
        )

    return ranked


        
