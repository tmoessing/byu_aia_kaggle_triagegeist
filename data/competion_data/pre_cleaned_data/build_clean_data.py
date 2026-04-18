import pandas as pd

train = pd.read_csv("data/competion_data/pre_cleaned_data/train.csv")
test = pd.read_csv("data/competion_data/pre_cleaned_data/test.csv")
patient_history = pd.read_csv("data/competion_data/pre_cleaned_data/patient_history.csv")
chief_complaints = pd.read_csv("data/competion_data/pre_cleaned_data/chief_complaints.csv")

# chief_complaints has both chief_complaint_raw and chief_complaint_system
# train/test already have chief_complaint_system, so only join chief_complaint_raw
cc_raw = chief_complaints[["patient_id", "chief_complaint_raw"]]

def add_missing_indicators(df):
    # Conclusion from missing data analysis (see data/pre_cleaned_data/MISSING_DATA/MISSING_DATA.md):
    # Vitals are MNAR — they are only missing for acuity 4 and 5 patients.
    # The missingness itself is a strong predictor of the target, so we encode it
    # explicitly as binary flags rather than imputing.
    #
    # bp_missing covers: systolic_bp, diastolic_bp, mean_arterial_pressure,
    #                    pulse_pressure, and shock_index (all missing together).
    #
    # news2_score is kept as-is — it was pre-calculated independently and should
    # not be recomputed from the raw vitals (see MISSING_DATA.md for details).
    df = df.copy()
    df["bp_missing"]               = df["systolic_bp"].isna().astype(int)
    df["respiratory_rate_missing"] = df["respiratory_rate"].isna().astype(int)
    df["temperature_missing"]      = df["temperature_c"].isna().astype(int)
    return df

def build_clean(df):
    df = df.merge(patient_history, on="patient_id", how="left")
    df = df.merge(cc_raw, on="patient_id", how="left")
    # move chief_complaint_raw next to chief_complaint_system
    cols = df.columns.tolist()
    cols.remove("chief_complaint_raw")
    idx = cols.index("chief_complaint_system")
    cols.insert(idx, "chief_complaint_raw")
    df = df[cols]
    df = add_missing_indicators(df)
    return df

clean_train = build_clean(train)
clean_test = build_clean(test)

clean_train.to_csv("data/competion_data/clean_train.csv", index=False)
clean_test.to_csv("data/competion_data/clean_test.csv", index=False)

print(f"clean_train: {clean_train.shape}")
print(f"clean_test:  {clean_test.shape}")
print(f"\nColumns added: {[c for c in clean_train.columns if c not in train.columns]}")
