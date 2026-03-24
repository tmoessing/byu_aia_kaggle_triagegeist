# Chief Complaint Word Frequency by Acuity Level

## Overview

This folder contains word frequency analyses of the `chief_complaint_raw` field from the triage dataset, broken down by ESI acuity level (1–5). Each file lists every word found across all complaints for that acuity level, ranked by frequency.

Common English function words (e.g., "with", "and", "the", "for") are excluded to keep the focus on clinically relevant terms.

## Files

| File | Acuity | Description | Complaints |
|------|--------|-------------|------------|
| `acuity_1_top_words.txt` | 1 – Immediate | Life-threatening, requires immediate intervention | 3,222 |
| `acuity_2_top_words.txt` | 2 – Emergent | High risk, should be seen within 15 minutes | 13,439 |
| `acuity_3_top_words.txt` | 3 – Urgent | Stable but needs timely evaluation | 28,921 |
| `acuity_4_top_words.txt` | 4 – Less Urgent | Non-urgent, can wait | 23,020 |
| `acuity_5_top_words.txt` | 5 – Non-Urgent | Minor, routine, or administrative | 11,398 |

