"""
Canonical text preprocessing and dataset preparation module.
Guarantees zero train-test leakage by providing a single transformation pipeline.
"""
import re
import pandas as pd
from typing import Tuple

def clean_incident_text(text: str) -> str:
    """
    Standardize text and replace domain-specific high-energy physical quantities
    with semantic hazard tokens so unseen values (e.g., 1200v, 50 tons) are captured.
    """
    if not isinstance(text, str):
        return ""
    
    text = text.lower()
    text = re.sub(r'[\r\n\t]+', ' ', text)

    # 1. Electrical Hazard Tokens
    text = re.sub(r'\b\d+\s*(?:v|volt|volts|kv)\b', ' __voltage_hazard__ ', text)
    
    # 2. Heavy Load / Mass Tokens
    text = re.sub(r'\b\d+\s*(?:kg|ton|tons|tn|lbs)\b', ' __heavy_load__ ', text)
    
    # 3. Working at Height Tokens
    text = re.sub(r'\b\d+\s*(?:m|meters|metres|ft|feet)\b', ' __height_hazard__ ', text)
    
    # 4. Pressurized Systems Tokens
    text = re.sub(r'\b\d+\s*(?:bar|psi|kpa|mpa)\b', ' __pressure_hazard__ ', text)

    # 5. Clean punctuation while preserving meaningful hyphens (e.g. lockout-tagout)
    text = re.sub(r'[^a-zA-Z0-9_\s-]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_and_prepare_dataset(csv_path: str) -> pd.DataFrame:
    """
    Loads raw CSV, strips unneeded index columns, deduplicates records,
    creates ground truth binary SIF target, and prepares clean text.
    """
    df = pd.read_csv(csv_path)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df.columns = df.columns.str.strip()

    # Drop null descriptions and filter noise
    df = df.dropna(subset=['Description']).copy()
    df['Description'] = df['Description'].astype(str).str.strip()
    df = df[df['Description'].str.len() > 15].copy()

    # Deduplicate based on description narrative
    initial_count = len(df)
    df = df.drop_duplicates(subset=['Description']).copy()
    dedup_count = len(df)

    # Create Binary Ground Truth Label: Potential Level IV, V, VI
    high_potential = ['IV', 'V', 'VI']
    df['is_sif'] = df['Potential Accident Level'].astype(str).str.strip().isin(high_potential).astype(int)

    # Apply Canonical Text Cleaning
    df['clean_text'] = df['Description'].apply(clean_incident_text)
    
    return df
