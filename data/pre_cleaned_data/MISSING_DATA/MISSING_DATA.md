# Missing Data Analysis

## Summary

Seven features have missing values, all vital signs. Everything else is complete.

| Feature | Train Missing | Train % | Test Missing | Test % |
|---|---|---|---|---|
| systolic_bp | 4,146 | 5.2% | 962 | 4.8% |
| diastolic_bp | 4,146 | 5.2% | 962 | 4.8% |
| mean_arterial_pressure | 4,146 | 5.2% | 962 | 4.8% |
| pulse_pressure | 4,146 | 5.2% | 962 | 4.8% |
| shock_index | 4,146 | 5.2% | 962 | 4.8% |
| respiratory_rate | 3,067 | 3.8% | 752 | 3.8% |
| temperature_c | 574 | 0.7% | 106 | 0.5% |

The BP group (systolic, diastolic, MAP, pulse_pressure, shock_index) always goes missing together — they're derived from the same measurement.

---

## Key Finding: MNAR (Missing Not At Random)

Missingness is **entirely determined by the target variable** (`triage_acuity`):

| Acuity | Severity | BP Missing % | RR Missing % | Temp Missing % |
|---|---|---|---|---|
| 1 | Most urgent | 0% | 0% | 0% |
| 2 | Urgent | 0% | 0% | 0% |
| 3 | Less urgent | 0% | 0% | 0% |
| 4 | Non-urgent | 12.1% | 8.9% | 0% |
| 5 | Minor | 11.8% | 8.9% | 5.0% |

**Vitals are only ever missing for acuity 4 and 5 patients.** This reflects real ED practice: low-priority patients may not have full vitals taken at triage.

This means the missingness itself is a strong predictor of the target — a missing BP tells you the patient is likely acuity 4 or 5.

---

## NEWS2 Inconsistency

**What is NEWS2?**
NEWS2 (National Early Warning Score 2) is a standardized clinical scoring system used in emergency departments to detect patient deterioration. It is computed by summing weighted scores from six vital signs:

| Input | Source column |
|---|---|
| Respiratory rate | `respiratory_rate` |
| Oxygen saturation | `spo2` |
| Systolic blood pressure | `systolic_bp` |
| Heart rate | `heart_rate` |
| Temperature | `temperature_c` |
| Consciousness level | `mental_status_triage` |

Higher scores indicate sicker patients. A score of 0–4 is low risk; 7+ is high risk and typically triggers immediate escalation.

**The inconsistency:**
Because `systolic_bp` is one of the six required inputs, NEWS2 *cannot be fully computed* for patients where BP is missing. Yet the dataset contains valid `news2_score` values for all 4,146 BP-missing rows:

| Group | NEWS2 Mean | NEWS2 Max |
|---|---|---|
| BP missing rows | 0.42 | 5 |
| BP present rows | 3.59 | 17 |

The values are low (mean 0.42), consistent with the acuity 4/5 patients these rows belong to, but they should not exist at all if NEWS2 were computed correctly from the raw vitals.

**What this means:**
NEWS2 was likely pre-calculated before some vitals were recorded, computed from a partial set of inputs, or generated independently of the raw vital columns in this dataset. Either way, `news2_score` should be treated as its own independent feature — do not try to recompute or validate it from the other columns.

---

## Approach: Option A — Binary Indicator Features

Add a `_missing` flag for each affected feature group. These flags directly encode the MNAR signal.

**Features added:**
- `bp_missing` — 1 if systolic_bp (and the whole BP group) is NaN
- `respiratory_rate_missing` — 1 if respiratory_rate is NaN
- `temperature_missing` — 1 if temperature_c is NaN

**Why not simple imputation?**
Global mean/median imputation would assign "normal-looking" vital values to ~5% of rows that are genuinely low-acuity, actively corrupting the signal. The indicator flags preserve the information while allowing the model to also use real values where available.
