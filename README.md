# SIF Incident-Report Screening System
**Supervised Learning (INT395) / Smart India Hackathon (SIH 2026 — SIH26165)**

An empirical, decision-support machine learning system to triage workplace incident and near-miss narratives, surfacing hidden Serious Injury & Fatality (SIF) precursors for prioritized safety committee inspection.

---

## 1. Problem Definition & Real-World Context

In industrial environments (mining, metallurgical processing, chemical manufacturing), thousands of near-miss and minor incident reports are logged monthly. Modern safety science demonstrates that routine minor accidents and fatal accidents do not share identical root causes: fatal incidents originate from **high-energy hazards with compromised controls (SIF Precursors)**. 

Because safety teams lack the capacity to manually audit every low-injury narrative, dangerous precursor events (e.g., an ungrounded 1200V busbar or a snapped crane hoist that narrowly missed a worker) are frequently archived as routine first-aid logs. This system automates the text triage process using regularized supervised NLP calibrated for high-recall screening.

---

## 2. System Architecture

```text
supervisedproject/
├── .streamlit/
│   └── config.toml          # Corporate dark-navy/slate theme
├── config.py                # Centralized paths, thresholds, and seeds
├── requirements.txt         # Pinned production dependencies
├── data/
│   └── industrial_safety.csv# 411 deduplicated industrial incident records
├── src/
│   ├── preprocessing.py     # Canonical regex unit normalizer & data loader
│   ├── train.py             # Repeated 5-Fold CV, ROC/PR-AUC calculation & artifact export
│   └── predict.py           # Single-text highlighter and high-throughput batch scoring
├── models/
│   ├── sif_vectorizer.pkl   # Fitted TF-IDF feature extractor
│   ├── sif_lr_model.pkl     # Calibrated Logistic Regression model
│   └── sif_rf_model.pkl     # Random Forest benchmark model
├── reports/
│   ├── metrics.json         # CV benchmark results, confusion matrices, ROC/PR points
│   ├── error_analysis.csv   # 10 False Positives & 10 False Negatives with root-cause notes
│   ├── flagged_top10.csv    # Top 10 highest-risk production review queue
│   └── cv_threshold_curve.csv # Full threshold operating trade-off points
└── app.py                   # Streamlit interactive decision-support UI
```

---

## 3. Empirical Results (Repeated 5-Fold Cross-Validation)

All models were evaluated using **5-Fold Stratified Cross-Validation repeated across 3 independent random seeds (15 total folds)**, with strict zero-leakage vectorization:

| Approach | Architecture | ROC-AUC (Mean ± Std) | PR-AUC (Mean ± Std) | Baseline PR Floor |
| :--- | :--- | :---: | :---: | :---: |
| **Model 1: Logistic Regression (Text Only)** | TF-IDF (1-2 ngrams) + L2 penalty | **0.724 ± 0.031** | **0.642 ± 0.045** | 0.412 |
| **Model 2: Random Forest (Text Only)** | 100 Ensembled Decision Trees | 0.704 ± 0.036 | 0.601 ± 0.048 | 0.412 |
| **Model 3: LR (Text + Sector/Risk Metadata)** | TF-IDF + One-Hot Encoded Categoricals | **0.734 ± 0.029** | **0.655 ± 0.042** | 0.412 |

### Operating Threshold Calibration
Standard models default to an operational threshold of $\tau = 0.50$. In life-critical safety triage, missing a precursor (False Negative) is catastrophic, whereas reviewing a harmless incident (False Positive) is merely a minor time expenditure.

* **Default Threshold ($\tau = 0.50$):** Recall on SIF Precursors is only **57%** (43% of fatal precursors missed).
* **Calibrated High-Recall Threshold ($\tau = 0.38$):** Recall increases to **90%+**, ensuring that life-threatening conditions reliably enter the manual review queue.

---

## 4. Key Engineering Improvements

1. **Semantic Unit Normalization:** Regular expressions convert unobserved domain quantities (e.g. `1200v`, `50 tons`, `15m`) into unified hazard tokens (`__voltage_hazard__`, `__heavy_load__`, `__height_hazard__`). This resolves the classic Out-Of-Vocabulary (OOV) failure of standard Bag-of-Words models.
2. **Decoupled Metric Ingestion:** The Streamlit application (`app.py`) dynamically reads from `reports/metrics.json` and `reports/cv_threshold_curve.csv`. No evaluation numbers are hardcoded in the user interface.
3. **In-Text Span Explainability:** Rather than outputting abstract mathematical weights, the screening interface renders the incident narrative directly with detected hazardous tokens highlighted in place.
4. **Batch CSV Pipeline:** Supports drag-and-drop ingestion of entire multi-incident shift logs, outputting a prioritized review queue with CSV export.

---

## 5. Execution Guide

### Installation
Ensure Python 3.10+ is installed:
```bash
pip install -r requirements.txt
```

### Model Training & Cross-Validation
Run the training script to execute 5-fold cross-validation and generate all metrics and model artifacts:
```bash
python src/train.py
```

### Launch Interactive Dashboard
Launch the Streamlit triage dashboard:
```bash
streamlit run app.py
```

---

## 6. Qualitative Error Analysis & Limitations

* **False Negatives:** Analysis indicates missed precursors occur primarily when narratives describe subtle machinery malfunctions using mundane narrative phrases without overt danger keywords.
* **False Positives:** Non-hazardous reports containing high-energy vocabulary (e.g., *"worker was using cut-resistant gloves while opening boxes"*) trigger false alarms. In safety screening, this bias toward caution is operationally preferred over missing real hazards.
* **Governance Notice:** This system is scoped strictly as an advisory decision-support screening tool for Environmental Health & Safety (EHS) committees and does not replace human safety investigations.
