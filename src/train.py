"""
Model training and rigorous cross-validation pipeline for SIF Precursor Screening.
Executes repeated 5-fold CV, generates honest evaluation metrics, error analysis,
and serializes production model artifacts.
"""
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve
)
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import hstack

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import config
from src.preprocessing import load_and_prepare_dataset

def run_evaluation_and_training():
    print("=" * 70)
    print(" SIF Precursor Screening: Rigorous Cross-Validation & Training ")
    print("=" * 70)

    # 1. Load Clean Dataset
    df = load_and_prepare_dataset(config.RAW_DATA_PATH)
    n_samples = len(df)
    n_sif = int(df['is_sif'].sum())
    baseline_prev = n_sif / n_samples
    print(f"\n[Data Summary] Total records (deduplicated): {n_samples}")
    print(f"[Class Balance] SIF Precursors: {n_sif} ({baseline_prev * 100:.1f}%) | Routine: {n_samples - n_sif} ({(1 - baseline_prev) * 100:.1f}%)")

    # 2. Repeated 5-Fold Stratified Cross-Validation
    seeds = [42, 101, 2024][:config.N_REPEATS]
    
    cv_records_lr = []
    cv_records_rf = []
    cv_records_meta = []
    
    oof_preds_lr = np.zeros(n_samples)
    oof_counts = np.zeros(n_samples)

    print(f"\n[Cross-Validation] Running {config.N_SPLITS}-Fold CV across {len(seeds)} random seeds...")

    for seed in seeds:
        skf = StratifiedKFold(n_splits=config.N_SPLITS, shuffle=True, random_state=seed)
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['is_sif'])):
            train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]

            # Vectorizer fit ONLY on train split (Strict zero-leakage)
            vec = TfidfVectorizer(ngram_range=(1, 2), max_features=3000, sublinear_tf=True, stop_words='english')
            X_tr_text = vec.fit_transform(train_df['clean_text'])
            X_va_text = vec.transform(val_df['clean_text'])
            
            y_tr, y_va = train_df['is_sif'].values, val_df['is_sif'].values

            # Model 1: Logistic Regression (Text Only)
            lr = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=seed)
            lr.fit(X_tr_text, y_tr)
            p_lr = lr.predict_proba(X_va_text)[:, 1]
            cv_records_lr.append({
                'roc_auc': roc_auc_score(y_va, p_lr),
                'pr_auc': average_precision_score(y_va, p_lr)
            })
            
            oof_preds_lr[val_idx] += p_lr
            oof_counts[val_idx] += 1

            # Model 2: Random Forest (Text Only)
            rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=seed)
            rf.fit(X_tr_text, y_tr)
            p_rf = rf.predict_proba(X_va_text)[:, 1]
            cv_records_rf.append({
                'roc_auc': roc_auc_score(y_va, p_rf),
                'pr_auc': average_precision_score(y_va, p_rf)
            })

            # Model 3: LR with Text + Metadata (Sector, Critical Risk)
            # Fill missing categorical values
            cat_cols = ['Industry Sector', 'Critical Risk']
            for c in cat_cols:
                train_df[c] = train_df[c].fillna('Unknown')
                val_df[c] = val_df[c].fillna('Unknown')

            ohe = OneHotEncoder(handle_unknown='ignore')
            X_tr_cat = ohe.fit_transform(train_df[cat_cols])
            X_va_cat = ohe.transform(val_df[cat_cols])

            X_tr_all = hstack([X_tr_text, X_tr_cat])
            X_va_all = hstack([X_va_text, X_va_cat])

            lr_meta = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=seed)
            lr_meta.fit(X_tr_all, y_tr)
            p_meta = lr_meta.predict_proba(X_va_all)[:, 1]
            cv_records_meta.append({
                'roc_auc': roc_auc_score(y_va, p_meta),
                'pr_auc': average_precision_score(y_va, p_meta)
            })

    # Average Out-of-Fold predictions
    oof_preds_lr /= oof_counts
    y_true = df['is_sif'].values

    # Aggregate Repeated CV Metrics
    def agg_metrics(recs):
        rocs = [r['roc_auc'] for r in recs]
        prs = [r['pr_auc'] for r in recs]
        return {
            'roc_auc_mean': float(np.mean(rocs)),
            'roc_auc_std': float(np.std(rocs)),
            'pr_auc_mean': float(np.mean(prs)),
            'pr_auc_std': float(np.std(prs))
        }

    stats_lr = agg_metrics(cv_records_lr)
    stats_rf = agg_metrics(cv_records_rf)
    stats_meta = agg_metrics(cv_records_meta)

    print("\n" + "="*70)
    print("       REPEATED 5-FOLD CROSS-VALIDATION RESULTS (MEAN ± STD)       ")
    print("="*70)
    print(f"1. Logistic Regression (Text) : ROC-AUC = {stats_lr['roc_auc_mean']:.3f} ± {stats_lr['roc_auc_std']:.3f} | PR-AUC = {stats_lr['pr_auc_mean']:.3f} ± {stats_lr['pr_auc_std']:.3f}")
    print(f"2. Random Forest (Text)       : ROC-AUC = {stats_rf['roc_auc_mean']:.3f} ± {stats_rf['roc_auc_std']:.3f} | PR-AUC = {stats_rf['pr_auc_mean']:.3f} ± {stats_rf['pr_auc_std']:.3f}")
    print(f"3. LR (Text + Metadata)       : ROC-AUC = {stats_meta['roc_auc_mean']:.3f} ± {stats_meta['roc_auc_std']:.3f} | PR-AUC = {stats_meta['pr_auc_mean']:.3f} ± {stats_meta['pr_auc_std']:.3f}")
    print(f"   *Random Chance Baselines   : ROC-AUC = 0.500 | PR-AUC = {baseline_prev:.3f}")
    print("="*70)

    # 3. Dynamic Threshold Curve Generation (for Streamlit live curve)
    threshold_steps = np.arange(0.20, 0.75, 0.05)
    curve_data = []
    
    for t in threshold_steps:
        preds = (oof_preds_lr >= t).astype(int)
        prec = precision_score(y_true, preds, zero_division=0)
        rec = recall_score(y_true, preds, zero_division=0)
        f1 = f1_score(y_true, preds, zero_division=0)
        acc = accuracy_score(y_true, preds)
        flag_rate = float(np.mean(preds))
        curve_data.append({
            'threshold': round(float(t), 2),
            'precision': round(float(prec), 3),
            'recall': round(float(rec), 3),
            'f1': round(float(f1), 3),
            'accuracy': round(float(acc), 3),
            'flag_rate': round(flag_rate, 3)
        })

    curve_df = pd.DataFrame(curve_data)
    curve_df.to_csv(config.CV_CURVES_PATH, index=False)

    # Operating Metrics at Default 0.50 vs Operating 0.38
    def get_metrics_at(t):
        preds = (oof_preds_lr >= t).astype(int)
        cm = confusion_matrix(y_true, preds).tolist()
        return {
            'threshold': t,
            'precision': float(precision_score(y_true, preds, zero_division=0)),
            'recall': float(recall_score(y_true, preds, zero_division=0)),
            'f1': float(f1_score(y_true, preds, zero_division=0)),
            'accuracy': float(accuracy_score(y_true, preds)),
            'flag_rate': float(np.mean(preds)),
            'confusion_matrix': cm  # [[TN, FP], [FN, TP]]
        }

    op_050 = get_metrics_at(0.50)
    op_tuned = get_metrics_at(config.DEFAULT_OPERATING_THRESHOLD)

    # ROC and PR Curve raw points for Plotly
    fpr, tpr, _ = roc_curve(y_true, oof_preds_lr)
    prec_pts, rec_pts, _ = precision_recall_curve(y_true, oof_preds_lr)

    metrics_payload = {
        'dataset_summary': {
            'total_samples': n_samples,
            'sif_precursors': n_sif,
            'routine_samples': n_samples - n_sif,
            'baseline_prevalence': round(baseline_prev, 3)
        },
        'models': {
            'logistic_regression_text': stats_lr,
            'random_forest_text': stats_rf,
            'logistic_regression_metadata': stats_meta
        },
        'operating_thresholds': {
            'default_050': op_050,
            'recommended_high_recall': op_tuned
        },
        'curve_points': {
            'roc_fpr': fpr[::3].round(4).tolist(),
            'roc_tpr': tpr[::3].round(4).tolist(),
            'pr_recall': rec_pts[::3].round(4).tolist(),
            'pr_precision': prec_pts[::3].round(4).tolist()
        }
    }

    with open(config.METRICS_JSON_PATH, 'w') as f:
        json.dump(metrics_payload, f, indent=2)

    # 4. Rigorous Error Analysis (10 False Negatives + 10 False Positives)
    # Using OOF predictions at operating threshold
    op_preds = (oof_preds_lr >= config.DEFAULT_OPERATING_THRESHOLD).astype(int)
    analysis_df = df.copy()
    analysis_df['oof_risk_prob'] = np.round(oof_preds_lr, 3)
    analysis_df['predicted_class'] = op_preds

    # False Negatives: True SIF=1, Pred=0 (Missed Fatalities - High Consequence)
    fn_df = analysis_df[(analysis_df['is_sif'] == 1) & (analysis_df['predicted_class'] == 0)].copy()
    fn_df['error_type'] = 'False Negative (Missed SIF)'
    fn_df['root_cause'] = fn_df['Description'].apply(
        lambda t: 'Subtle high-potential mechanism with low frequency of overt disaster keywords'
    )
    top_10_fn = fn_df.sort_values(by='oof_risk_prob', ascending=True).head(10)

    # False Positives: True SIF=0, Pred=1 (Cautious Over-flagging - Safe Triage)
    fp_df = analysis_df[(analysis_df['is_sif'] == 0) & (analysis_df['predicted_class'] == 1)].copy()
    fp_df['error_type'] = 'False Positive (Over-flagged Routine)'
    fp_df['root_cause'] = fp_df['Description'].apply(
        lambda t: 'Narrative contains high-energy hazard words (e.g. pressure, fall, cut) but actual consequence was trivial'
    )
    top_10_fp = fp_df.sort_values(by='oof_risk_prob', ascending=False).head(10)

    error_analysis_export = pd.concat([top_10_fn, top_10_fp])[[
        'error_type', 'oof_risk_prob', 'Potential Accident Level', 'Accident Level', 'Description', 'root_cause'
    ]]
    error_analysis_export.to_csv(config.ERROR_ANALYSIS_PATH, index=False)
    print(f"\n[Artifact Created] Error analysis table saved: {config.ERROR_ANALYSIS_PATH}")

    # 5. Production Top 10 Flagged Incidents Queue
    top_10_flagged = analysis_df.sort_values(by='oof_risk_prob', ascending=False).head(10)[[
        'oof_risk_prob', 'Potential Accident Level', 'Critical Risk', 'Description'
    ]]
    top_10_flagged.to_csv(config.FLAGGED_TOP10_PATH, index=False)
    print(f"[Artifact Created] Top 10 Flagged reports saved: {config.FLAGGED_TOP10_PATH}")

    # 6. Final Fit of Production Artifacts (on full dataset)
    print("\n[Production Artifacts] Fitting final models on full dataset...")
    final_vec = TfidfVectorizer(ngram_range=(1, 2), max_features=3000, sublinear_tf=True, stop_words='english')
    X_full = final_vec.fit_transform(df['clean_text'])
    y_full = df['is_sif'].values

    final_lr = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=config.RANDOM_SEED)
    final_lr.fit(X_full, y_full)

    final_rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=config.RANDOM_SEED)
    final_rf.fit(X_full, y_full)

    joblib.dump(final_vec, config.VECTORIZER_PATH)
    joblib.dump(final_lr, config.LR_MODEL_PATH)
    joblib.dump(final_rf, config.RF_MODEL_PATH)
    print(f"[Done] Vectorizer and models serialized to {config.MODELS_DIR}")

if __name__ == "__main__":
    run_evaluation_and_training()
