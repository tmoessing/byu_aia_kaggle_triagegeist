import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer

# Include unigrams + two-word phrases (bigrams) in both pipelines.
NGRAM_RANGE = (1, 2)
MAX_FEATURES = 50

print("Loading data...")
base_dir = Path(__file__).resolve().parent
input_csv = base_dir / 'chief_complaints.csv'

df = pd.read_csv(input_csv)
df['clean_text'] = df['chief_complaint_raw'].str.lower()


def bigram_feature_count(feature_names):
    return sum(1 for feature in feature_names if ' ' in feature)

# --- DEFINING OUR EXPERIMENT ---
# These are the subjective words we hypothesized introduce human bias
subjective_words = [
    # 1. Pain/Intensity Scales (Highly inter-rater variable)
    'mild', 'moderate', 'severe', 'extreme', 'minor', 'major',
    
    # 2. Vague Quantifiers
    'small', 'large', 'medium', 'tiny', 'massive', 'heavy', 'light',
    
    # 3. Vague Sentiments
    'general', 'feeling', 'unwell', 'abnormal'
]

# Create the "Objective-Only" text column by removing those specific words
# This uses a quick lambda function to rebuild the sentence without the subjective words
df['objective_text'] = df['clean_text'].apply(
    lambda text: ' '.join([word for word in str(text).split() if word not in subjective_words])
)

# --- PIPELINE A: FULL CONTEXT ---
print("Running Pipeline A (Full Context)...")
tfidf_full = TfidfVectorizer(ngram_range=NGRAM_RANGE, stop_words='english', max_features=MAX_FEATURES)
matrix_full = tfidf_full.fit_transform(df['clean_text'].fillna(''))
full_feature_names = tfidf_full.get_feature_names_out()

df_full_features = pd.DataFrame(
    matrix_full.toarray(), 
    columns=[f"nlp_full_{word.replace(' ', '_')}" for word in full_feature_names]
)
df_full_features['patient_id'] = df['patient_id']
df_full_features.to_csv(base_dir / 'features_pipeline_A_full_2_phrase.csv', index=False)
print(f"Pipeline A features: {len(full_feature_names)} total, {bigram_feature_count(full_feature_names)} two-word phrases")


# --- PIPELINE B: OBJECTIVE ONLY ---
print("Running Pipeline B (Objective Only)...")
tfidf_obj = TfidfVectorizer(ngram_range=NGRAM_RANGE, stop_words='english', max_features=MAX_FEATURES)
matrix_obj = tfidf_obj.fit_transform(df['objective_text'].fillna(''))
obj_feature_names = tfidf_obj.get_feature_names_out()

df_obj_features = pd.DataFrame(
    matrix_obj.toarray(), 
    columns=[f"nlp_obj_{word.replace(' ', '_')}" for word in obj_feature_names]
)
df_obj_features['patient_id'] = df['patient_id']
df_obj_features.to_csv(base_dir / 'features_pipeline_B_objective_2_phrase.csv', index=False)
print(f"Pipeline B features: {len(obj_feature_names)} total, {bigram_feature_count(obj_feature_names)} two-word phrases")

print("Done! Both feature sets are ready for modeling.")