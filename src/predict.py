"""
Inference and explainability engine for SIF Precursor Screening.
Provides single-text inference with in-text span highlighting and high-throughput batch scoring.
"""
import re
import html
import numpy as np
import pandas as pd
import joblib
from typing import Dict, Any, List, Tuple
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from src.preprocessing import clean_incident_text

def get_models():
    """Load production model artifacts from models directory."""
    vec = joblib.load(config.VECTORIZER_PATH)
    lr = joblib.load(config.LR_MODEL_PATH)
    rf = joblib.load(config.RF_MODEL_PATH)
    return vec, lr, rf

def get_priority_tier(prob: float, threshold: float) -> Tuple[str, str, str]:
    """
    Map probability to color-coded operational safety priority.
    Returns: (Label, CSS Color, Hex Background)
    """
    if prob >= 0.70:
        return "CRITICAL SIF RISK", "#EF4444", "rgba(239, 68, 68, 0.15)"
    elif prob >= 0.50:
        return "HIGH SIF RISK", "#F97316", "rgba(249, 115, 22, 0.15)"
    elif prob >= threshold:
        return "ELEVATED PRECURSOR RISK", "#EAB308", "rgba(234, 179, 8, 0.15)"
    else:
        return "ROUTINE / LOW PRECURSOR RISK", "#22C55E", "rgba(34, 197, 94, 0.15)"

def generate_highlighted_narrative(raw_text: str, top_words: List[str]) -> str:
    """
    Injects high-contrast warning badges directly into the narrative text
    for explainability without altering original wording.
    """
    safe_text = html.escape(raw_text)
    
    # Add domain quantities to highlight list
    regex_terms = [r'\b\d+\s*(?:v|volt|volts|kv)\b', r'\b\d+\s*(?:kg|ton|tons)\b', r'\b\d+\s*(?:m|meters|ft)\b']
    
    # Highlight specific detected top words
    for word in top_words:
        if len(word) > 2 and not word.startswith('__'):
            pattern = re.compile(rf'\b({re.escape(word)})\b', re.IGNORECASE)
            safe_text = pattern.sub(
                r'<span style="background-color: rgba(239, 68, 68, 0.25); color: #f87171; padding: 2px 5px; border-radius: 4px; font-weight: 600;">\1</span>',
                safe_text
            )
            
    # Highlight raw physical quantity indicators
    for reg in regex_terms:
        pattern = re.compile(reg, re.IGNORECASE)
        safe_text = pattern.sub(
            r'<span style="background-color: rgba(249, 115, 22, 0.25); color: #fb923c; padding: 2px 5px; border-radius: 4px; font-weight: 600;">\g<0></span>',
            safe_text
        )

    return safe_text

def score_narrative(
    text: str,
    model_name: str = "Logistic Regression",
    threshold: float = config.DEFAULT_OPERATING_THRESHOLD
) -> Dict[str, Any]:
    """Score a single incident report text."""
    vec, lr, rf = get_models()
    model = lr if "Logistic" in model_name else rf

    cleaned = clean_incident_text(text)
    if not cleaned:
        return {
            'risk_probability': 0.0,
            'is_flagged': False,
            'priority_tier': "UNKNOWN",
            'badge_color': "#94A3B8",
            'top_phrases': [],
            'highlighted_html': html.escape(text)
        }

    X = vec.transform([cleaned])
    prob = float(model.predict_proba(X)[0][1])
    is_flagged = prob >= threshold
    tier, text_color, bg_color = get_priority_tier(prob, threshold)

    # Extract top contributing phrases
    top_phrases = []
    detected_words_for_highlight = []
    
    if "Logistic" in model_name:
        feature_names = np.array(vec.get_feature_names_out())
        coefs = lr.coef_[0]
        # Find active features in this input
        indices = X.nonzero()[1]
        word_contributions = []
        for idx in indices:
            weight = coefs[idx]
            feat = feature_names[idx]
            if weight > 0:
                word_contributions.append((feat, float(weight)))
                detected_words_for_highlight.extend(feat.split())

        word_contributions.sort(key=lambda x: x[1], reverse=True)
        top_phrases = word_contributions[:6]

    highlighted_html = generate_highlighted_narrative(text, detected_words_for_highlight)

    return {
        'risk_probability': round(prob, 4),
        'is_flagged': is_flagged,
        'priority_tier': tier,
        'badge_color': text_color,
        'badge_bg': bg_color,
        'top_phrases': top_phrases,
        'highlighted_html': highlighted_html
    }

def score_batch_dataframe(
    df: pd.DataFrame,
    text_column: str,
    model_name: str = "Logistic Regression",
    threshold: float = config.DEFAULT_OPERATING_THRESHOLD
) -> pd.DataFrame:
    """High-throughput scoring for an entire batch of incident logs."""
    vec, lr, rf = get_models()
    model = lr if "Logistic" in model_name else rf

    clean_series = df[text_column].astype(str).apply(clean_incident_text)
    X = vec.transform(clean_series)
    probs = model.predict_proba(X)[:, 1]

    out_df = df.copy()
    out_df['SIF_Risk_Score'] = np.round(probs, 4)
    out_df['Flagged_For_Review'] = (probs >= threshold).astype(int)
    
    tiers = []
    for p in probs:
        label, _, _ = get_priority_tier(p, threshold)
        tiers.append(label)
    out_df['Priority_Tier'] = tiers

    # Sort descending by risk score to present actionable queue
    return out_df.sort_values(by='SIF_Risk_Score', ascending=False).reset_index(drop=True)
