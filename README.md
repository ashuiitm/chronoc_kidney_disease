# CKD Point-of-Care Risk Screening — a portfolio project for Renalyx

**Why this project:** Renalyx's Point-of-Care Solution screens diabetic and hypertensive patients
for early-stage Chronic Kidney Disease, and its RxT 21 hemodialysis platform emphasizes real-time,
cloud-connected clinical decision support. This project simulates the first piece of that pipeline:
a model that flags CKD risk from the kind of routine vitals and basic labs collected at a
screening visit — before a patient ever needs a nephrology referral or dialysis.

## Data

[UCI Chronic Kidney Disease dataset](https://archive.ics.uci.edu/dataset/336/chronic+kidney+disease)
— 400 patients, 24 clinical features (vitals, urinalysis, blood chemistry, comorbidities), binary
CKD/not-CKD label. Genuinely messy, in the way real clinical exports are:

- Numeric labs (`wc`, `rc`, `pcv`) stored as strings
- Categorical fields with inconsistent formatting (`'\tyes'`, `' yes'`, `'yes'` all meaning the same thing)
- A typo in the target label itself (`'ckd\t'`)
- Missingness up to ~40% in several lab columns (`sod`, `pot`, `rc`)

See `src/01_clean.py` for the cleaning pipeline and `data/missingness_report.csv` for the
per-column missingness audit.

## Feature engineering

On top of the raw labs, I added five domain-informed features (`src/02_train.py`):

| Feature | Clinical rationale |
|---|---|
| `bun_creatinine_ratio` | Classic nephrology signal distinguishing prerenal from intrinsic renal dysfunction |
| `comorbidity_count` | Sum of hypertension + diabetes + CAD — the exact population Renalyx's screening tool targets |
| `anemia_flag` | Hemoglobin < 11 g/dL — CKD-associated anemia |
| `significant_albuminuria` | Albumin grade ≥ 2 — stronger CKD signal than trace albuminuria |
| `low_specific_gravity` | Urine specific gravity ≤ 1.010 — impaired renal concentrating ability |

## Models & results

Two models, trained on a stratified 75/25 split (n=300 train / 100 test):

| Model | ROC-AUC | F1 | Precision | Recall |
|---|---|---|---|---|
| Logistic Regression (baseline, explainable by design) | 1.00 | 1.00 | 1.00 | 1.00 |
| XGBoost | 1.00 | 0.99 | 1.00 | 0.98 |

5-fold CV on the training set: mean AUC 0.9998 (± 0.0005).

**On the near-perfect scores — read this before assuming it's a bug or overclaiming:**
This dataset was collected and published specifically as a CKD *classification* benchmark, and
several of its features (hemoglobin, specific gravity, red cell counts) are close to diagnostic
markers for CKD, not just risk factors — so very high separability is expected and shows up
consistently in prior work on this exact dataset. At n=400 from what appears to be a single site,
this result is a sanity check that the pipeline works end-to-end, not evidence of real-world
screening performance. I would not present this as production-ready accuracy.

## Explainability

`figures/shap_summary.png` shows SHAP values for the XGBoost model. The top drivers — low
hemoglobin, low urine specific gravity, low RBC count, high serum creatinine — line up with
standard nephrology teaching, which is the check I'd want before trusting any clinical model:
does the model's reasoning match a clinician's reasoning, not just the accuracy number.

## What I'd change before this touches a real patient

This is the part that maps directly to the JD's "SaMD standards, FDA AI/ML guidance" ask, and
it's deliberately the most detailed section:

1. **Data**: multi-site cohort, thousands of patients not hundreds, prospective collection
   matching the actual screening workflow (age/BP/urinalysis at a PHC, not a full nephrology panel)
2. **Validation**: temporal or geographic holdout instead of a random split, calibration curves
   (not just AUC — a screening tool needs trustworthy predicted probabilities), subgroup
   performance checks across age/sex/comorbidity strata
3. **Monitoring**: drift detection on incoming screening data vs. training distribution, with
   retraining triggers — this is the piece I built end-to-end at Mediassist (MLflow + Evidently AI)
   and would carry over directly
4. **Documentation**: model card + validation report structured against FDA AI/ML guidance
   and IEC 62304 traceability requirements, versioned alongside the model artifact

## Repo structure

```
data/           raw + cleaned CSVs, missingness report
src/01_clean.py cleaning pipeline
src/02_train.py feature engineering, training, evaluation, SHAP
figures/        confusion matrix, ROC curve, SHAP summary
results_metrics.json, feature_importance.csv, classification_report.txt
```

## Stack

Python · pandas · scikit-learn · XGBoost · SHAP · matplotlib/seaborn
