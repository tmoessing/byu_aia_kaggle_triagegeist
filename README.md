# BYU AIA Kaggle — Triageeist

**Question:** Can AI meaningfully support triage decisions in the emergency department?

**Problem:** Find a model that predicts triage acuity level from structured patient intake data (vitals, demographics, chief complaint).

**Team:** Tyler · Nathan · Carson

**Links:** [GitHub](https://github.com/tmoessing/byu_aia_kaggle_triagegeist) · [Kaggle Competition](https://www.kaggle.com/competitions/triagegeist/overview)

---

## File Structure

```
byu_aia_kaggle_triagegeist/
│
├── main.py                        # Model training and evaluation
│
├── data/
│   ├── clean_train.csv            # Final training data (ready for model)
│   ├── clean_test.csv             # Final test data (ready for model)
│   │
│   ├── pre_cleaned_data/          # Raw inputs and cleaning pipeline
│   │   ├── train.csv
│   │   ├── test.csv
│   │   ├── patient_history.csv
│   │   ├── chief_complaints.csv
│   │   ├── build_clean_data.py    # Runs the full cleaning pipeline
│   │   └── MISSING_DATA/
│   │       ├── MISSING_DATA.md            # Missing data analysis write-up
│   │       └── missing_data_analysis.py   # Code used to produce the analysis
│   │
│   └── additional_data/           # External datasets (see Tasks below)
│       ├── ed2022.csv
│       ├── ed2022-stata/
│       └── stata_to_csv.py        # Converts .dta to .csv
```

---

## Data Cleaning

The cleaning pipeline lives in `data/pre_cleaned_data/build_clean_data.py`.

Run it from the project root:
```bash
python data/pre_cleaned_data/build_clean_data.py
```

This produces `data/clean_train.csv` and `data/clean_test.csv`.

**What it does:**

1. **Joins patient history** — merges `patient_history.csv` onto train/test by `patient_id`
2. **Joins chief complaint raw text** — merges `chief_complaint_raw` from `chief_complaints.csv` and places it next to `chief_complaint_system`
3. **Adds missing data indicators** — see Missing Data section below

---

## Missing Data

Full write-up: `data/pre_cleaned_data/MISSING_DATA/MISSING_DATA.md`

**Short version:** Seven vital sign columns have missing values — all exclusively in acuity 4 and 5 patients (less urgent). Acuity 1, 2, and 3 rows have zero missing values. This is MNAR (Missing Not At Random): the vitals were simply never recorded for lower-priority patients, which is standard ED practice.

Because the missingness itself predicts the target, we encode it as binary flags rather than imputing:

| New column | Captures |
|---|---|
| `bp_missing` | systolic_bp, diastolic_bp, MAP, pulse_pressure, shock_index |
| `respiratory_rate_missing` | respiratory_rate |
| `temperature_missing` | temperature_c |

These three columns are added automatically by `build_clean_data.py`.

**Note on NEWS2:** `news2_score` is a standard clinical scoring system (National Early Warning Score 2) computed from BP, RR, temperature, SpO2, HR, and consciousness. It should be non-zero when BP is missing — yet it has valid values for all missing-BP rows. This means it was pre-calculated independently and should be treated as its own feature, not recomputed from the raw vitals.

---

## main.py

Our Model

---

## Tasks

### Baseline

- [ ] Create a Decision Tree and determine feature importance *(Owner: Carson)*
- [ ] Check class imbalance
- [ ] Confirm no leakage

### Modeling

- [ ] Random Forest
- [ ] XGBoost
- [ ] LightGBM
- [ ] CatBoost
- [ ] Logistic Regression
- [ ] Cross-validation
- [ ] Ensemble / stacking

### Feature Engineering

- [ ] Chief complaint features
- [ ] Vital sign ratios
- [ ] Acuity-group vitals analysis
- [ ] Missing as a category
- [ ] NLP on chief complaints

### Additional Data *(Owner: Tyler)*

- [ ] Explore ed2022.csv
- [ ] Identify overlap with existing features
- [ ] Join or extract aggregates into pipeline
- [ ] Document new missing value patterns
