# Additional Data: NHAMCS Emergency Department 2022 (ed2022.csv)

## Source

**National Hospital Ambulatory Medical Care Survey (NHAMCS) — Emergency Department, 2022**
Published by the CDC/NCHS. Each row is a single ED visit, sampled from a nationally representative set of US hospital EDs.

- Rows: 16,025 visits
- Columns: 914 features
- Also available as Stata file: `ed2022-stata/ed2022-stata.dta`

---

## Feature Groups

### Visit Timing
| Column | Description |
|--------|-------------|
| `VMONTH` | Month of visit |
| `VDAYR` | Day of week |
| `ARRTIME` | Arrival time (HHMM) |
| `WAITTIME` | Wait time to see provider (minutes) |
| `LOV` | Length of visit (minutes) |

### Patient Demographics
| Column | Description |
|--------|-------------|
| `AGE` | Age in years |
| `AGER` | Age recode (grouped) |
| `AGEDAYS` | Age in days (for infants) |
| `SEX` | Sex |
| `ETHUN` / `ETHIM` | Hispanic/Latino ethnicity |
| `RACEUN` / `RACER` / `RACERETH` | Race / race+ethnicity |
| `RESIDNCE` | Residence type |

### Arrival & Transport
| Column | Description |
|--------|-------------|
| `ARREMS` | Arrived by ambulance/EMS (Y/N) |
| `AMBTRANSFER` | Transferred by ambulance from another facility |

### Payment / Insurance
| Column | Description |
|--------|-------------|
| `PAYTYPER` | Primary expected payer (recode) |
| `PAYPRIV` / `PAYMCARE` / `PAYMCAID` / `PAYWKCMP` / `PAYSELF` / `PAYNOCHG` / `PAYOTH` / `PAYDK` | Individual payer flags |
| `NOPAY` | No expected payment |

### Triage / Acuity
| Column | Description |
|--------|-------------|
| `IMMEDR` | Immediacy — triage level (1=immediate … 5=non-urgent) |
| `PAINSCALE` | Pain scale (0–10) |
| `SEEN72` | Seen within 72 hours for same problem |

### Vital Signs
| Column | Description |
|--------|-------------|
| `TEMPF` | Temperature (°F) |
| `PULSE` | Heart rate (bpm) |
| `RESPR` | Respiratory rate |
| `BPSYS` / `BPDIAS` | Systolic / diastolic blood pressure |
| `POPCT` | Pulse oximetry (SpO2 %) |
| `VITALSD` | Vitals documented flag |
| `TEMPDF` / `PULSED` / `RESPRD` / `BPSYSD` / `BPDIASD` | Vital sign missing/default flags |

### Reason for Visit (Chief Complaint)
| Column | Description |
|--------|-------------|
| `RFV1`–`RFV5` | Up to 5 reason-for-visit codes |
| `RFV13D`–`RFV53D` | Verbatim reason-for-visit text (up to 5) |
| `EPISODE` | Visit episode type |
| `INJURY` / `INJPOISAD` / `INJURY72` / `INTENT15` / `INJURY_ENC` | Injury-related flags |
| `CAUSE1`–`CAUSE3` | External cause of injury codes |

### Diagnoses
| Column | Description |
|--------|-------------|
| `DIAG1`–`DIAG5` | ICD-10-CM ED diagnosis codes (up to 5) |
| `PRDIAG1`–`PRDIAG5` | Primary diagnosis flags |
| `TOTDIAG` | Total number of diagnoses |
| `HDDIAG1`–`HDDIAG5` | Hospital discharge diagnosis codes |
| `HDSTAT` | Hospital discharge status |

### Chronic Conditions / Comorbidities
| Column | Description |
|--------|-------------|
| `HTN` | Hypertension |
| `DIABTYP1` / `DIABTYP2` / `DIABTYP0` | Diabetes type 1 / type 2 / unspecified |
| `ASTHMA` | Asthma |
| `COPD` | COPD |
| `CHF` | Congestive heart failure |
| `CAD` | Coronary artery disease |
| `CKD` | Chronic kidney disease |
| `ESRD` | End-stage renal disease |
| `OBESITY` | Obesity |
| `DEPRN` | Depression |
| `ALZHD` | Alzheimer's / dementia |
| `CANCER` | Malignancy |
| `CEBVD` | Cerebrovascular disease (stroke history) |
| `EDHIV` | HIV |
| `SUBSTAB` | Substance abuse |
| `ETOHAB` | Alcohol abuse |
| `HPE` | Hemiplegia/paraplegia |
| `HYPLIPID` | Hyperlipidemia |
| `OSA` | Obstructive sleep apnea |
| `OSTPRSIS` | Osteoporosis |
| `NOCHRON` / `TOTCHRON` | No chronic conditions / total count |

### Labs Ordered
| Column | Description |
|--------|-------------|
| `CBC`, `BMP`, `CMP`, `BNP`, `BUNCREAT`, `CARDENZ` | Common lab panels |
| `GLUCOSE`, `LACTATE`, `LFT`, `PTTINR`, `ELECTROL`, `DDIMER`, `OTHERBLD` | Individual labs |
| `BLOODCX`, `URINECX`, `TRTCX`, `WOUNDCX`, `OTHCX` | Culture types |
| `ABG`, `BAC` | ABG / blood alcohol |
| `HIVTEST`, `FLUTEST`, `COVIDTEST`, `COVIDANTIBODY`, `PREGTEST`, `TOXSCREN`, `URINE`, `OTHRTEST` | Rapid/point-of-care tests |
| `DIAGSCRN` | Diagnostic screen ordered |
| `CARDMON`, `EKG` | Cardiac monitoring / EKG |

### Imaging
| Column | Description |
|--------|-------------|
| `ANYIMAGE` | Any imaging ordered |
| `XRAY` | X-ray |
| `CATSCAN` | CT scan (any) |
| `CTAB`, `CTCHEST`, `CTHEAD`, `CTOTHER`, `CTUNK` | CT by body region |
| `CTCONTRAST` | CT with contrast |
| `MRI` / `MRICONTRAST` | MRI |
| `ULTRASND` | Ultrasound |
| `OTHIMAGE` | Other imaging |

### Procedures
| Column | Description |
|--------|-------------|
| `PROC` | Any procedure performed |
| `IVFLUIDS`, `NEBUTHER`, `BLADCATH`, `CASTSPLINT`, `CENTLINE`, `CPR`, `ENDOINT` | Common procedures |
| `INCDRAIN`, `LUMBAR`, `PELVIC`, `SKINADH`, `SUTURE`, `BPAP`, `OTHPROC` | Additional procedures |
| `TOTPROC` | Total procedures |

### Medications
| Column | Description |
|--------|-------------|
| `MED` | Any medication given |
| `MED1`–`MED30` | Medication given flags (up to 30) |
| `DRUGID1`–`DRUGID30` | Drug identifier (Multum code) for each med |
| `GPMED1`–`GPMED30` | Generic medication name |
| `RXnCAT1`–`RXnCAT4` | Drug therapeutic category (up to 4 per drug) |
| `RXnV1C1`–`RXnV3C4` | Detailed drug classification hierarchy |
| `PRESCR1`–`PRESCR30` | Prescription at discharge flags |
| `CONTSUB1`–`CONTSUB30` | Controlled substance flags |
| `COMSTAT1`–`COMSTAT30` | Medication completion status |
| `NUMGIV` / `NUMDIS` / `NUMMED` | Number of meds given / prescribed / total |

### Providers Seen
| Column | Description |
|--------|-------------|
| `ATTPHYS`, `RESINT`, `CONSULT`, `RNLPN`, `NURSEPR`, `PHYSASST`, `EMT`, `MHPROV`, `OTHPROV` | Provider types present |
| `NOPROVID` | No provider documented |

### Disposition
| Column | Description |
|--------|-------------|
| `ADMIT` | Admitted to hospital |
| `ADMITHOS` | Admitted to this hospital |
| `OBSHOS` / `OBSDIS` | Observation stay |
| `TRANNH` / `TRANPSYC` / `TRANOTH` | Transferred (nursing home / psych / other) |
| `NODISP` / `NOFU` / `RETRNED` / `RETREFFU` | No disposition / no follow-up / returned |
| `LWBS` / `LBTC` / `LEFTAMA` | Left without being seen / left before triage / left AMA |
| `DOA` / `DIEDED` | Dead on arrival / died in ED |
| `OTHDISP` | Other disposition |
| `LOS` | Length of stay (if admitted, in days) |
| `ADISP` / `ADMTPHYS` / `OBSSTAY` / `STAY24` | Admission details |

### Hospital Characteristics
| Column | Description |
|--------|-------------|
| `REGION` | US census region |
| `MSA` | Metropolitan statistical area |
| `EMEDRES` | ED residency program |
| `EMRED`, `EHRINSE`, `HHSMUE` | HIT/EHR adoption flags |
| `EDPRIM`, `EDINFO`, `OBSCLIN`, `OBSSEP`, `OBSPHYSED`, `OBSHOSP`, `OBSPHYSOT`, `OBSPHYSUN` | ED services |
| `BOARD`, `BOARDHOS`, `AMBDIV`, `TOTHRDIVR`, `REGDIV`, `ADMDIV` | Boarding / diversion metrics |
| `BEDREG`, `IMBED`, `ADVTRIAG`, `PHYSPRACTRIA`, `FASTTRAK`, `EDPTOR`, `KIOSELCHK`, `CATRIAGE` | ED flow processes |
| `DASHBORD`, `RFID`, `WIRELESS`, `ZONENURS`, `POOLNURS`, `SURGDAY`, `BEDCZAR`, `BEDDATA` | ED technology / staffing |
| `HLIST`, `HLISTED` | Hospital list flags |
| `HOSPCODE` / `PATCODE` | De-identified hospital / patient codes |

### Survey Weights
| Column | Description |
|--------|-------------|
| `SETTYPE` | Setting type |
| `YEAR` | Survey year (2022) |
| `CSTRATM` / `CPSUM` | Survey strata / PSU |
| `PATWT` / `EDWT` | Patient / ED visit weight |
| `BOARDED` | Patient boarded flag |

---

## Feature Alignment with Competition Dataset

The competition dataset (`clean_train.csv` / `clean_test.csv`) maps to ed2022 as follows:

### Direct / Near-Direct Matches

| Competition Feature | ed2022 Column | Notes |
|---------------------|---------------|-------|
| `age` | `AGE` | Same concept; ed2022 also has `AGER` (grouped) and `AGEDAYS` |
| `sex` | `SEX` | Same |
| `arrival_month` | `VMONTH` | Same |
| `arrival_day` | `VDAYR` | Day of week |
| `arrival_hour` | `ARRTIME` | ed2022 is HHMM integer; competition is hour |
| `arrival_mode` | `ARREMS` | ed2022 is EMS flag only; competition is richer mode field |
| `insurance_type` | `PAYTYPER` | ed2022 recoded payer type; competition is categorical |
| `systolic_bp` | `BPSYS` | Same; check `BPSYSD` for missing flag |
| `diastolic_bp` | `BPDIAS` | Same; check `BPDIASD` for missing flag |
| `heart_rate` | `PULSE` | Same; check `PULSED` for missing flag |
| `respiratory_rate` | `RESPR` | Same; check `RESPRD` for missing flag |
| `temperature_c` | `TEMPF` | ed2022 is Fahrenheit — convert: C = (F-32)*5/9; check `TEMPDF` |
| `spo2` | `POPCT` | Pulse oximetry |
| `pain_score` | `PAINSCALE` | Same 0–10 scale |
| `triage_acuity` | `IMMEDR` | Immediacy / ESI triage level |
| `ed_los_hours` | `LOV` | ed2022 in minutes; divide by 60 |
| `chief_complaint_raw` | `RFV13D`–`RFV53D` | ed2022 has up to 5 verbatim RFV text fields |
| `chief_complaint_system` | `RFV1`–`RFV5` | Coded reason-for-visit (NAMCS reason code) |
| `hx_hypertension` | `HTN` | Same |
| `hx_diabetes_type1` | `DIABTYP1` | Same |
| `hx_diabetes_type2` | `DIABTYP2` | Same |
| `hx_asthma` | `ASTHMA` | Same |
| `hx_copd` | `COPD` | Same |
| `hx_heart_failure` | `CHF` | Same |
| `hx_coronary_artery_disease` | `CAD` | Same |
| `hx_ckd` | `CKD` | Same |
| `hx_obesity` | `OBESITY` | Same |
| `hx_depression` | `DEPRN` | Same |
| `hx_dementia` | `ALZHD` | Alzheimer's + dementia in ed2022 |
| `hx_malignancy` | `CANCER` | Same |
| `hx_stroke_prior` | `CEBVD` | Cerebrovascular disease (stroke included) |
| `hx_hiv` | `EDHIV` | Same |
| `hx_substance_use_disorder` | `SUBSTAB` | Substance abuse flag |
| `disposition` | `ADMIT`, `TRANNH`, `LWBS`, `LEFTAMA`, etc. | ed2022 splits disposition into many binary columns |

### Partial / Derived Matches

| Competition Feature | ed2022 Equivalent | Notes |
|---------------------|-------------------|-------|
| `num_comorbidities` | `TOTCHRON` | Total chronic conditions count — very close proxy |
| `shock_index` | Derived: `PULSE / BPSYS` | Compute from ed2022 vitals |
| `mean_arterial_pressure` | Derived: `BPDIAS + (BPSYS-BPDIAS)/3` | Compute from ed2022 vitals |
| `pulse_pressure` | Derived: `BPSYS - BPDIAS` | Compute from ed2022 vitals |
| `news2_score` | Partially derivable | Need RR, SpO2, temp, HR, SBP, consciousness — most present |
| `arrival_season` | Derived from `VMONTH` | Map months to season |
| `num_prior_ed_visits_12m` | `SEEN72` | ed2022 only captures 72-hour return; not 12-month count |
| `hx_anxiety` | No direct column | Not in ed2022 comorbidity list |
| `hx_atrial_fibrillation` | No direct column | Not explicitly in ed2022 |
| `hx_liver_disease` | No direct column | Not in ed2022 comorbidity list |
| `hx_epilepsy` | No direct column | Not in ed2022 |
| `hx_hypothyroidism` / `hx_hyperthyroidism` | No direct column | Not in ed2022 |
| `hx_coagulopathy` | No direct column | Not in ed2022 |
| `hx_immunosuppressed` | No direct column | Not in ed2022 |
| `hx_peripheral_vascular_disease` | No direct column | Not in ed2022 |
| `hx_pregnant` | `PREGTEST` | Only a test flag, not a confirmed diagnosis |
| `transport_origin` | `AMBTRANSFER` | ed2022 only has transfer flag, not full origin |
| `language` | No direct column | Not captured in ed2022 |
| `mental_status_triage` | No direct column | Not in ed2022 |
| `weight_kg` / `height_cm` / `bmi` | No direct columns | Not captured in ed2022 |
| `gcs_total` | No direct column | Not in ed2022 |
| `num_active_medications` | `NUMMED` | Medications given/prescribed in this visit, not active home meds |
| `num_prior_admissions_12m` | No direct column | Not in ed2022 |
| `site_id` / `triage_nurse_id` | `HOSPCODE` | Only hospital-level de-id; no nurse ID |
| `shift` | Derived from `ARRTIME` | Map ARRTIME to shift bins |

### Rich Features in ed2022 Not in Competition Data

These are available only in ed2022 and could serve as useful supplementary signal:

- **Labs ordered**: CBC, BMP, CMP, lactate, troponin/cardiac enzymes, D-dimer, etc.
- **Imaging ordered**: X-ray, CT (by body region), MRI, ultrasound
- **Procedures**: IV fluids, intubation, nebulizer, catheter, etc.
- **Medications given**: Up to 30 drugs with full therapeutic classification hierarchy
- **Diagnoses**: Up to 5 ICD-10 codes; also hospital discharge diagnoses
- **Wait time**: `WAITTIME` — time from arrival to provider (not in competition data)
- **Hospital characteristics**: Region, MSA, ED flow processes, boarding metrics
- **Provider types**: Which provider types were involved in care
- **Survey weights**: `PATWT`/`EDWT` for nationally representative analysis

---

## Key Notes for Modeling

1. **No join key** — ed2022 and the competition dataset cannot be row-matched. ed2022 is a separate real-world survey; the competition data is synthetic or from a different source. Use ed2022 for **distributional reference, imputation priors, or pretraining**.

2. **Temperature units** — ed2022 `TEMPF` is Fahrenheit; competition `temperature_c` is Celsius.

3. **LOV vs ed_los_hours** — ed2022 `LOV` is in minutes; divide by 60 to compare.

4. **Disposition encoding** — competition uses a single categorical `disposition`; ed2022 uses multiple binary columns. You will need to construct a mapping.

5. **Comorbidity flags** — ed2022 comorbidity flags are binary (1=yes, 2=no); competition uses 0/1. Recode accordingly.

6. **Survey weights** — ed2022 is a complex survey sample. Use `PATWT` for patient-level estimates or `EDWT` for visit-level national estimates if computing reference statistics.
