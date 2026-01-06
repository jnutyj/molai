# molai — Molecular AI Toolkit (v0.1) STILL UNDER DEVELOPMENT


**molai** is a modular Python toolkit for **molecular property prediction** and **de novo molecule generation** using **SMILES-based deep learning models**.




It is designed for:
- QSAR modeling (e.g. pIC50 prediction)
- inverse molecular design
- generative chemistry research
- small to medium datasets (∼10²–10⁴ molecules)

> **Design philosophy**  
> Simple building blocks → composable pipelines → research-ready workflows

---

## ✨ Features

### 🔹 Prediction
- SMILES → property regression
- LSTM-based predictors
- Fingerprint-based baselines (MACCS, Morgan)

### 🔹 Generation
- SMILES LSTM language models
- SMILES Variational Autoencoder (VAE)
- Property-guided molecule generation

### 🔹 Optimization
- Filtering-based inverse QSAR
- Reinforcement learning (policy gradient)
- **Latent-space optimization (VAE + latent predictor)**

### 🔹 Chemistry utilities
- RDKit SMILES validation
- QED / SA / novelty filtering
- PhysChem constraints (MW, logP, scaffolds)

---

## 🧠 Core Concepts

molai cleanly separates **modeling**, **training**, and **application logic**:


