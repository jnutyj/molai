# molai/pipelines/qsar.py

import pickle
import pandas as pd
from typing import Literal, Dict

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

from molai.features.fingerprints import (
    maccs_fingerprint,
    morgan_fingerprint,
)
from molai.training.supervised import train_sklearn_regressor
from molai.utils.metrics import regression_metrics


########################################
# Feature registry
########################################

FEATURE_EXTRACTORS = {
    "maccs": maccs_fingerprint,
    "morgan": morgan_fingerprint,
}


########################################
# QSAR PIPELINE
########################################

def run_qsar_pipeline(
    csv_path: str,
    smiles_col: str = "SMILES",
    target_col: str = "pIC50",
    feature_type: Literal["maccs", "morgan"] = "maccs",
    model_type: Literal["rf"] = "rf",
    test_size: float = 0.2,
    random_state: int = 42,
    output_model: str = "qsar_model.pkl",
):
    """
    Train and evaluate a QSAR model.

    Parameters
    ----------
    csv_path : str
        CSV file with SMILES and target
    feature_type : str
        'maccs' or 'morgan'
    model_type : str
        Currently only 'rf'
    """

    print("Loading dataset...")
    df = pd.read_csv(csv_path)

    smiles = df[smiles_col].tolist()
    y = df[target_col].values

    # -------------------------
    # Feature extraction
    # -------------------------
    print(f"Extracting features: {feature_type}")
    featurizer = FEATURE_EXTRACTORS[feature_type]
    X = featurizer(smiles)

    # -------------------------
    # Train / test split
    # -------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state
    )

    # -------------------------
    # Model selection
    # -------------------------
    if model_type == "rf":
        model = RandomForestRegressor(
            n_estimators=300,
            n_jobs=-1,
            random_state=random_state,
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # -------------------------
    # Train
    # -------------------------
    print("Training QSAR model...")
    model = train_sklearn_regressor(
        model,
        X_train,
        y_train,
    )

    # -------------------------
    # Evaluate
    # -------------------------
    print("Evaluating...")
    y_pred = model.predict(X_test)
    metrics = regression_metrics(y_test, y_pred)

    for k, v in metrics.items():
        print(f"{k}: {v:.3f}")

    # -------------------------
    # Save model
    # -------------------------
    artifact = {
        "model": model,
        "feature_type": feature_type,
        "metrics": metrics,
    }

    with open(output_model, "wb") as f:
        pickle.dump(artifact, f)

    print(f"QSAR model saved to {output_model}")

    return artifact

