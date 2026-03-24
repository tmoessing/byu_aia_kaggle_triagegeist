import pandas as pd

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

# ── 1. Overall missing counts ─────────────────────────────────────────────────
print("--- Train Missing Values ---")
m = train.isna().sum()
print(m[m > 0].sort_values(ascending=False))
print(f"Total Rows: {len(train)}\n")

print("--- Test Missing Values ---")
m2 = test.isna().sum()
print(m2[m2 > 0].sort_values(ascending=False))
print(f"Total Rows: {len(test)}\n")

# ── 2. Co-occurrence of missingness ───────────────────────────────────────────
bp_missing   = train["systolic_bp"].isna()
rr_missing   = train["respiratory_rate"].isna()
temp_missing = train["temperature_c"].isna()

print("--- Missing co-occurrence (train) ---")
print(f"BP missing rows      : {bp_missing.sum()}")
print(f"RR missing rows      : {rr_missing.sum()}")
print(f"Temp missing rows    : {temp_missing.sum()}")
print(f"BP AND RR missing    : {(bp_missing & rr_missing).sum()}")
print(f"BP AND Temp missing  : {(bp_missing & temp_missing).sum()}")
print(f"RR AND Temp (not BP) : {(~bp_missing & rr_missing & temp_missing).sum()}")
print(f"All 3 missing        : {(bp_missing & rr_missing & temp_missing).sum()}")
print(f"Only BP missing      : {(bp_missing & ~rr_missing & ~temp_missing).sum()}")
print(f"Only RR missing      : {(~bp_missing & rr_missing & ~temp_missing).sum()}")
print(f"Only Temp missing    : {(~bp_missing & ~rr_missing & temp_missing).sum()}\n")

# ── 3. Missing % by triage acuity (MNAR check) ───────────────────────────────
for label, col in [("BP", "systolic_bp"), ("RR", "respiratory_rate"), ("Temp", "temperature_c")]:
    print(f"--- {label} missing % by acuity ---")
    for acuity in sorted(train["triage_acuity"].unique()):
        subset = train[train["triage_acuity"] == acuity]
        pct = subset[col].isna().mean() * 100
        n   = subset[col].isna().sum()
        print(f"  Acuity {acuity}: {pct:.1f}% missing ({n}/{len(subset)})")
    print()

# ── 4. NEWS2 consistency check ────────────────────────────────────────────────
print("--- NEWS2 stats for BP-MISSING rows ---")
print(train[bp_missing]["news2_score"].describe())
print()
print("--- NEWS2 stats for BP-PRESENT rows ---")
print(train[~bp_missing]["news2_score"].describe())
