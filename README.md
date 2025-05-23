# 🧬 Drug-Likeness Classifier using Machine Learning

This project builds a machine learning pipeline to classify molecules as **drug-like** or **non-drug-like** using chemical fingerprints and optional quantum mechanical (QM) features. It is designed for high-throughput screening and will ultimately be integrated with a protein–ligand binding affinity model (KIBA) for multi-objective drug discovery.

---

## 🚀 Project Goals

- Classify molecules based on drug-likeness using supervised ML
- Benchmark multiple models:
  - ✅ Neural Networks (MLP)
  - ✅ Random Forest
  - ✅ XGBoost
- Integrate QM properties (HOMO/LUMO/dipole)
- Enable scoring of generated molecules (e.g., ZINC, MolGen)
- Lay groundwork for future multi-objective pipelines (drug-likeness + binding)

---

## 🧱 Project Structure

```
drug_likeness/
├── data/
│   ├── pytdc_chembl_valid.smi              # Known drugs (positive samples)
│   ├── zinc_candidates_cleaned.smi         # Candidate or generated molecules
│   ├── zinc_qm_final.csv                   # QM features extracted from structures.tar.gz
│   └── zinc_with_fingerprints.csv          # SMILES + Morgan fingerprints
├── models/
│   ├── model_nn.pkl                        # Trained neural network
│   ├── model_rf.pkl                        # Trained random forest
│   └── model_xgb.pkl                       # Trained XGBoost model
├── scripts/
│   ├── train_nn.py                         # Train neural network
│   ├── train_rf.py                         # Train random forest
│   ├── train_xgb.py                        # Train XGBoost
│   ├── extract_qm_from_sdf_inchikey.py     # Extract QM features from structures.tar.gz
│   └── compare_models.py                   # Evaluate all models: accuracy, ROC, confusion
└── zinc_qm_final.csv                       # Final merged SMILES + QM output
```

---

## 📦 Dependencies

Install everything you need with:

```bash
pip install pandas numpy rdkit scikit-learn xgboost tqdm pyarrow
```

---

## ✅ How to Run

### 1. Prepare Input Data

- Ensure your ZINC candidate SMILES are cleaned and saved as:
  ```
  data/zinc_candidates_cleaned.smi
  ```
- Extract QM features from `structures.tar.gz` using:
  ```bash
  python scripts/extract_qm_from_sdf_inchikey.py
  ```

This will output:
```
data/zinc_qm_final.csv
```

---

### 2. Train Models

```bash
python scripts/train_nn.py
python scripts/train_rf.py
python scripts/train_xgb.py
```

Each model will be saved in `models/`.

---

### 3. Evaluate Models

Run the comparison and generate ROC curves and scores:

```bash
python scripts/compare_models.py
```

---

## 🔬 Features Used

- **Morgan Fingerprints** (radius=2, nBits=2048)
- Optional QM descriptors:
  - `DFT:HOMO_ENERGY`
  - `DFT:LUMO_ENERGY`
  - `DFT:HOMO_LUMO_GAP`
  - `DFT:DIPOLE`

---

## 🧪 Work in Progress: KIBA Integration

We are currently developing a **multi-objective scoring pipeline** by merging this classifier with a **QM-augmented KIBA binding affinity predictor** (R² ≈ 0.62).

**Planned final scoring function:**

```python
final_score = w1 * drug_likeness(smiles) + w2 * binding_affinity(smiles, protein, qm_features)
```

This will enable virtual screening pipelines that optimize for:
- Drug-likeness
- Target affinity
- QM-based electronic descriptors

This component is under active development and will be documented in future releases.

---

## 📄 License

MIT License – you are free to use, modify, and distribute this project.

---

## 📬 Contact

**Aniket Mandal**  
For questions or collaborations: aniket.kmandal@gmail.com
