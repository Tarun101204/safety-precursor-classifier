import pandas as pd
import numpy as np
import re
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

print("==========================================================")
print(" SIF Incident-Report Classification (Topic 18 / SIH26165)")
print("==========================================================")

# 1. LOAD DATASET
data_path = "data/industrial_safety.csv"
df = pd.read_csv(data_path)

# Drop any unneeded index column and strip whitespace from column names
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
df.columns = df.columns.str.strip()

# 2. DEFINE SIF TARGET (y)
# Potential Level IV, V, VI represent high-energy precursor events
high_potential_levels = ['IV', 'V', 'VI']
df['is_sif'] = df['Potential Accident Level'].astype(str).str.strip().isin(high_potential_levels).astype(int)

# 3. TEXT PREPROCESSING
df = df.dropna(subset=['Description'])

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[\r\n\t]+', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9\s-]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['clean_text'] = df['Description'].apply(clean_text)

# 4. STRATIFIED TRAIN-TEST SPLIT (80% Train, 20% Test)
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    df['clean_text'], 
    df['is_sif'], 
    test_size=0.20, 
    stratify=df['is_sif'], 
    random_state=42
)

# 5. TF-IDF VECTORIZATION
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=3000,
    sublinear_tf=True,
    stop_words='english'
)

X_train = vectorizer.fit_transform(X_train_raw)
X_test = vectorizer.transform(X_test_raw)

# 6. TRAIN BASELINE (Logistic Regression with Balanced Weights)
model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# 7. HIGH-RECALL THRESHOLD CALIBRATION
y_probs = model.predict_proba(X_test)[:, 1]

# Default 0.50 threshold
print("\n--- Standard Threshold (0.50) Evaluation ---")
print(classification_report(y_test, (y_probs >= 0.50).astype(int), target_names=["Routine (0)", "SIF Precursor (1)"]))

# Tuned High-Recall 0.35 threshold
threshold = 0.35
y_pred_tuned = (y_probs >= threshold).astype(int)

print(f"\n--- High-Recall Threshold ({threshold}) Evaluation ---")
print(classification_report(y_test, y_pred_tuned, target_names=["Routine (0)", "SIF Precursor (1)"]))

# 8. SAVE TRAINED ARTIFACTS (FOR STREAMLIT WEB APP)
joblib.dump(model, 'sif_model.pkl')
joblib.dump(vectorizer, 'sif_vectorizer.pkl')
print("\nModel saved to 'sif_model.pkl' and vectorizer to 'sif_vectorizer.pkl'")

# 9. OUTPUT TOP 10 FLAGGED EXAMPLES (Syllabus Deliverable)
results_df = pd.DataFrame({
    'Narrative': X_test_raw.values,
    'True_SIF': y_test.values,
    'Risk_Score': np.round(y_probs, 3),
    'Flagged': y_pred_tuned
})

top_10 = results_df.sort_values(by='Risk_Score', ascending=False).head(10)

print("\n--- Top 10 Flagged Incidents for Safety Review Queue ---")
for idx, row in top_10.reset_index(drop=True).iterrows():
    print(f"\n[{idx+1}] Score: {row['Risk_Score']} | Ground Truth SIF: {row['True_SIF']}")
    print(f"Narrative: {row['Narrative'][:120]}...")