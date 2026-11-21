import pandas as pd
import numpy as np
import re
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, recall_score, precision_score, accuracy_score

print("==========================================================")
print(" SIF Precursor Classification: Model Comparison Pipeline  ")
print("==========================================================")

# ------------------------------------------------------------------
# STEP 1: LOAD DATASET
# ------------------------------------------------------------------
# We load the verified dataset from the data folder
df = pd.read_csv("data/industrial_safety.csv")
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
df.columns = df.columns.str.strip()

# ------------------------------------------------------------------
# STEP 2: DEFINE SIF PRECURSOR TARGET (y)
# ------------------------------------------------------------------
# Potential Levels IV, V, VI represent severe/fatal potential
df['is_sif'] = df['Potential Accident Level'].astype(str).str.strip().isin(['IV', 'V', 'VI']).astype(int)
df = df.dropna(subset=['Description'])

# ------------------------------------------------------------------
# STEP 3: DOMAIN HAZARD NORMALIZATION (Solves the 1200v Issue!)
# ------------------------------------------------------------------
# Normalizing units allows the model to recognize any voltage or weight
def advanced_clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[\r\n\t]+', ' ', text)
    # Replace any voltage (e.g. 1200v, 440v, 50kv) with a unified hazard token
    text = re.sub(r'\b\d+\s*(?:v|volt|volts|kv)\b', ' __voltage_hazard__ ', text)
    # Replace any heavy load (e.g. 200kg, 5 tons) with heavy load token
    text = re.sub(r'\b\d+\s*(?:kg|ton|tons|tn)\b', ' __heavy_load__ ', text)
    # Replace heights (e.g. 4m, 10 meters) with height hazard token
    text = re.sub(r'\b\d+\s*(?:m|meters|metres|ft)\b', ' __height_hazard__ ', text)
    # Remove remaining punctuation
    text = re.sub(r'[^a-zA-Z0-9_\s-]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['clean_text'] = df['Description'].apply(advanced_clean_text)

# ------------------------------------------------------------------
# STEP 4: STRATIFIED TRAIN-TEST SPLIT (80% Train, 20% Test)
# ------------------------------------------------------------------
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    df['clean_text'], 
    df['is_sif'], 
    test_size=0.20, 
    stratify=df['is_sif'], 
    random_state=42
)

# ------------------------------------------------------------------
# STEP 5: TF-IDF VECTORIZATION
# ------------------------------------------------------------------
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=3000,
    sublinear_tf=True,
    stop_words='english'
)
X_train = vectorizer.fit_transform(X_train_raw)
X_test = vectorizer.transform(X_test_raw)

# ------------------------------------------------------------------
# STEP 6: TRAIN MODEL 1 (Logistic Regression) & MODEL 2 (Random Forest)
# ------------------------------------------------------------------
print("\n>>> Training Model 1: Logistic Regression (Baseline)...")
m1_lr = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
m1_lr.fit(X_train, y_train)

print(">>> Training Model 2: Random Forest (100 Decision Trees)...")
m2_rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
m2_rf.fit(X_train, y_train)

# ------------------------------------------------------------------
# STEP 7: EVALUATE & COMPARE MODELS AT BOTH THRESHOLDS
# ------------------------------------------------------------------
thresholds = [0.50, 0.35]
comparison_records = []

for thresh in thresholds:
    # Model 1 Predictions
    p1 = m1_lr.predict_proba(X_test)[:, 1]
    pred1 = (p1 >= thresh).astype(int)
    comparison_records.append({
        'Model': '1. Logistic Regression',
        'Threshold': thresh,
        'Accuracy': f"{accuracy_score(y_test, pred1):.2f}",
        'Precision (SIF)': f"{precision_score(y_test, pred1):.2f}",
        'Recall (SIF)': f"{recall_score(y_test, pred1):.2f}",
        'F1-Score': f"{f1_score(y_test, pred1):.2f}"
    })
    
    # Model 2 Predictions
    p2 = m2_rf.predict_proba(X_test)[:, 1]
    pred2 = (p2 >= thresh).astype(int)
    comparison_records.append({
        'Model': '2. Random Forest',
        'Threshold': thresh,
        'Accuracy': f"{accuracy_score(y_test, pred2):.2f}",
        'Precision (SIF)': f"{precision_score(y_test, pred2):.2f}",
        'Recall (SIF)': f"{recall_score(y_test, pred2):.2f}",
        'F1-Score': f"{f1_score(y_test, pred2):.2f}"
    })

# Display Comparison Table
results_table = pd.DataFrame(comparison_records)
print("\n" + "="*70)
print("           OFFICIAL MODEL COMPARISON TABLE (CA1 DELIVERABLE)")
print("="*70)
print(results_table.to_string(index=False))
print("="*70)

# ------------------------------------------------------------------
# STEP 8: SAVE ARTIFACTS
# ------------------------------------------------------------------
# Save the enhanced vectorizer and Random Forest model
joblib.dump(m2_rf, 'sif_rf_model.pkl')
joblib.dump(m1_lr, 'sif_lr_model.pkl')
joblib.dump(vectorizer, 'sif_vectorizer.pkl')
print("\nModels and vectorizer successfully saved for Streamlit!")