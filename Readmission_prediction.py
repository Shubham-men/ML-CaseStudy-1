# ------------------------------------------------------------------------
# 0. IMPORTS
# ------------------------------------------------------------------------
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Newer scikit-learn versions (>=1.8) emit a FutureWarning recommending
# 'l1_ratio' instead of the 'penalty' argument. We keep penalty="l2"
# below because the assignment explicitly asks for L2 regularization to
# be set explicitly and visibly, and this call remains fully functional
# in current scikit-learn - we just silence the cosmetic warning here.
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, roc_curve, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, classification_report
)

# Reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Only edit this if your CSV lives somewhere else
DATA_PATH = "diabetic_data.csv"


# ==========================================================================
# 1. LOAD THE DATASET
# ==========================================================================
print("=" * 78)
print("STEP 1: LOADING DATA")
print("=" * 78)

df = pd.read_csv(DATA_PATH)

print(f"\nDataset shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
print("\nFirst 5 rows:")
print(df.head())
print("\nColumn names:")
print(list(df.columns))
print("\nBasic info:")
df.info()


# ==========================================================================
# 2. DATA CLEANING / PREPROCESSING (part 1: row-level and column-level)
# ==========================================================================
print("\n" + "=" * 78)
print("STEP 2: DATA CLEANING")
print("=" * 78)

# --------------------------------------------------------------------
# 2.1 Treat '?' as missing (NaN). This dataset uses '?' as its own
#     placeholder for missing values instead of a blank cell.
# --------------------------------------------------------------------
df = df.replace("?", np.nan)

# --------------------------------------------------------------------
# 2.2 Remove rows where the patient died or was discharged to hospice.
#     discharge_disposition_id codes 11, 13, 14, 19, 20, 21 correspond to
#     "Expired" or "Hospice" outcomes. These patients cannot be
#     readmitted in any meaningful clinical sense, so keeping them would
#     add noisy/invalid labels to the "not readmitted" (0) class.
# --------------------------------------------------------------------
expired_or_hospice_codes = [11, 13, 14, 19, 20, 21]
before_rows = len(df)
df = df[~df["discharge_disposition_id"].isin(expired_or_hospice_codes)].copy()
print(f"Removed {before_rows - len(df):,} rows for expired/hospice discharge "
      f"(readmission is not clinically meaningful for these patients).")

# --------------------------------------------------------------------
# 2.3 Drop the 3 rows with gender == 'Unknown/Invalid'. This category is
#     essentially a data-entry artifact with only 3 records and carries
#     no reliable clinical meaning.
# --------------------------------------------------------------------
before_rows = len(df)
df = df[df["gender"] != "Unknown/Invalid"].copy()
print(f"Removed {before_rows - len(df):,} rows with gender == 'Unknown/Invalid'.")

# --------------------------------------------------------------------
# 2.4 Drop columns that are identifiers or otherwise unsuitable as
#     predictive features:
#       - encounter_id, patient_nbr : unique identifiers, carry no
#         clinical signal and would only let the model "memorize" rows.
#       - weight                    : ~97% missing in this dataset,
#         far too sparse to impute reliably.
#       - payer_code                : ~40% missing and reflects
#         insurance/billing information rather than clinical status.
# --------------------------------------------------------------------
columns_to_drop = ["encounter_id", "patient_nbr", "weight", "payer_code"]
df = df.drop(columns=columns_to_drop)
print(f"Dropped identifier/high-missingness columns: {columns_to_drop}")

# --------------------------------------------------------------------
# 2.5 Drop zero-variance columns (columns with only a single unique
#     value, e.g. medication columns that are "No" for every patient).
#     A feature that never changes cannot help the model discriminate
#     between classes, so we remove it automatically rather than
#     hard-coding column names.
# --------------------------------------------------------------------
zero_variance_cols = [c for c in df.columns if df[c].nunique(dropna=True) <= 1]
df = df.drop(columns=zero_variance_cols)
print(f"Dropped zero-variance columns: {zero_variance_cols}")

# --------------------------------------------------------------------
# 2.6 The three ID columns below are stored as integers but are really
#     CATEGORY CODES (e.g. admission_type_id 1 = "Emergency", 2 =
#     "Urgent" ...). Treating them as plain numbers would wrongly imply
#     an order/magnitude relationship (e.g. that code 6 is "more" than
#     code 3), so we cast them to strings and one-hot encode them later.
# --------------------------------------------------------------------
id_like_categoricals = ["admission_type_id", "discharge_disposition_id",
                         "admission_source_id"]
for col in id_like_categoricals:
    df[col] = df[col].astype(str)

# --------------------------------------------------------------------
# 2.7 Simplify the diagnosis codes (diag_1, diag_2, diag_3).
#     These are raw ICD-9 codes with 700+ unique values each. One-hot
#     encoding every individual code would create thousands of very
#     sparse columns and hurt interpretability. Instead, we map each
#     code to its clinical ICD-9 chapter (a fixed, rule-based lookup
#     defined by the ICD-9 standard - not derived from this dataset,
#     so this introduces no data leakage).
# --------------------------------------------------------------------
def map_diagnosis_to_category(code):
    """Map a raw ICD-9 diagnosis code to a broad clinical category."""
    if pd.isna(code):
        return "Missing"
    code = str(code)
    # Codes starting with V (supplemental) or E (external causes)
    if code.startswith("V") or code.startswith("E"):
        return "Other"
    try:
        value = float(code)
    except ValueError:
        return "Other"

    if 250 <= value < 251:
        return "Diabetes"
    elif 390 <= value <= 459 or value == 785:
        return "Circulatory"
    elif 460 <= value <= 519 or value == 786:
        return "Respiratory"
    elif 520 <= value <= 579 or value == 787:
        return "Digestive"
    elif 800 <= value <= 999:
        return "Injury"
    elif 710 <= value <= 739:
        return "Musculoskeletal"
    elif 580 <= value <= 629 or value == 788:
        return "Genitourinary"
    elif 140 <= value <= 239:
        return "Neoplasms"
    else:
        return "Other"

for diag_col in ["diag_1", "diag_2", "diag_3"]:
    df[diag_col + "_category"] = df[diag_col].apply(map_diagnosis_to_category)
df = df.drop(columns=["diag_1", "diag_2", "diag_3"])
print("Converted diag_1/diag_2/diag_3 into clinical categories "
      "(Diabetes, Circulatory, Respiratory, Digestive, Injury, "
      "Musculoskeletal, Genitourinary, Neoplasms, Other, Missing).")

# --------------------------------------------------------------------
# 2.8 Build the binary target variable.
#     <30 -> 1 (the event we want to predict)
#     >30 or NO -> 0
# --------------------------------------------------------------------
df["readmitted_binary"] = (df["readmitted"] == "<30").astype(int)
df = df.drop(columns=["readmitted"])

print(f"\nRemaining shape after cleaning: {df.shape[0]:,} rows x {df.shape[1]} columns")


# ==========================================================================
# 3. FEATURE SELECTION - split into numeric vs. categorical feature lists
# ==========================================================================
print("\n" + "=" * 78)
print("STEP 3: FEATURE SELECTION")
print("=" * 78)

target_col = "readmitted_binary"

# Numeric, clinically meaningful counts/measurements.
numeric_features = [
    "time_in_hospital",       # length of stay
    "num_lab_procedures",     # number of lab tests performed
    "num_procedures",         # number of non-lab procedures performed
    "num_medications",        # number of distinct medications administered
    "number_outpatient",      # outpatient visits in the year before this encounter
    "number_emergency",       # emergency visits in the year before this encounter
    "number_inpatient",       # inpatient visits in the year before this encounter
    "number_diagnoses",       # number of diagnoses entered for this encounter
]

# Everything else (besides the target) is treated as categorical: patient
# demographics, admission/discharge circumstances, diagnosis categories,
# lab result flags, and the 20+ medication-dosage-change columns.
categorical_features = [c for c in df.columns
                         if c not in numeric_features + [target_col]]

print(f"Numeric features ({len(numeric_features)}): {numeric_features}")
print(f"\nCategorical features ({len(categorical_features)}):")
print(categorical_features)

X = df[numeric_features + categorical_features]
y = df[target_col]


# ==========================================================================
# 4. CLASS IMBALANCE CHECK
# ==========================================================================
print("\n" + "=" * 78)
print("STEP 4: CLASS DISTRIBUTION / IMBALANCE CHECK")
print("=" * 78)

class_counts = y.value_counts().sort_index()
class_pct = y.value_counts(normalize=True).sort_index() * 100
print("\nClass counts (0 = not readmitted <30 days, 1 = readmitted <30 days):")
for cls in class_counts.index:
    print(f"  Class {cls}: {class_counts[cls]:,} samples "
          f"({class_pct[cls]:.2f}%)")

print(
    "\nThe positive class (readmitted within 30 days) is a clear minority "
    "(~11% of patients). If left unaddressed, a classifier can reach high "
    "'accuracy' simply by predicting 'not readmitted' for almost everyone, "
    "while completely failing at the clinically important task of catching "
    "high-risk patients. This motivates evaluating with ROC-AUC / recall / "
    "F1 (not accuracy alone) and considering class_weight='balanced'."
)

# Plot: class distribution
plt.figure(figsize=(6, 4))
sns.barplot(x=class_counts.index.astype(str), y=class_counts.values,
            hue=class_counts.index.astype(str), palette="viridis", legend=False)
plt.xticks([0, 1], ["Not readmitted <30d (0)", "Readmitted <30d (1)"])
plt.ylabel("Number of patients")
plt.title("Class Distribution: 30-Day Readmission")
for i, v in enumerate(class_counts.values):
    plt.text(i, v + 500, f"{v:,}", ha="center")
plt.tight_layout()
plt.savefig("class_distribution.png", dpi=150)
plt.show()
print("\nSaved plot: class_distribution.png")


# ==========================================================================
# 5. TRAIN-TEST SPLIT (stratified, to preserve class ratio in both sets)
# ==========================================================================
print("\n" + "=" * 78)
print("STEP 5: TRAIN-TEST SPLIT")
print("=" * 78)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    stratify=y,            # preserves the ~89/11 class ratio in both splits
    random_state=RANDOM_STATE
)

print(f"Training set: {X_train.shape[0]:,} samples")
print(f"Test set:     {X_test.shape[0]:,} samples")
print(f"Training set positive rate: {y_train.mean():.4f}")
print(f"Test set positive rate:     {y_test.mean():.4f}")


# ==========================================================================
# 6. PREPROCESSING PIPELINE (fit ONLY on training data to avoid leakage)
# ==========================================================================
print("\n" + "=" * 78)
print("STEP 6: BUILDING THE PREPROCESSING + MODEL PIPELINE")
print("=" * 78)

# Numeric pipeline: impute any missing numeric values with the median,
# then standardize (mean 0, std 1) - important for regularized logistic
# regression, since L2 penalizes large coefficients and features must be
# on comparable scales for that penalty to be fair across features.
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

# Categorical pipeline: fill missing values with the literal string
# "Missing" (treating "missingness" itself as informative, which is
# reasonable here since '?' was not random - e.g. certain specialties
# are simply not recorded for certain admission types), then one-hot
# encode. handle_unknown="ignore" ensures that if the test set contains
# a category never seen during training, it is encoded as all-zeros
# instead of crashing the pipeline.
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

# Combine into a single ColumnTransformer. Because this whole pipeline
# is fit with pipeline.fit(X_train, y_train) below, ALL statistics used
# for imputing/scaling/encoding (medians, means, stds, category lists)
# come exclusively from the training data - the test set is only ever
# *transformed*, never used to compute these statistics. This is what
# prevents data leakage.
preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features),
])


# ==========================================================================
# 7. MODEL TRAINING - Logistic Regression with L2 regularization
# ==========================================================================
print("\n" + "=" * 78)
print("STEP 7: TRAINING LOGISTIC REGRESSION MODELS")
print("=" * 78)

# We train two versions to study the effect of class weighting:
#   (a) a baseline model (no class weighting)
#   (b) a model with class_weight="balanced", which re-weights the loss
#       function so that misclassifying the minority (readmitted) class
#       is penalized more heavily than misclassifying the majority class,
#       roughly in inverse proportion to class frequency. This nudges the
#       decision boundary to catch more true positives (higher recall for
#       the clinically important class), typically at some cost to
#       precision.
models = {
    "baseline (no class weighting)": LogisticRegression(
        penalty="l2", C=1.0, solver="lbfgs",
        max_iter=2000, random_state=RANDOM_STATE
    ),
    "balanced (class_weight='balanced')": LogisticRegression(
        penalty="l2", C=1.0, solver="lbfgs",
        max_iter=2000, random_state=RANDOM_STATE, class_weight="balanced"
    ),
}

fitted_pipelines = {}
results = {}

for name, clf in models.items():
    pipe = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", clf),
    ])
    pipe.fit(X_train, y_train)
    fitted_pipelines[name] = pipe

    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    results[name] = {
        "y_pred": y_pred,
        "y_proba": y_proba,
        "roc_auc": roc_auc_score(y_test, y_proba),
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
    }

    print(f"\n--- {name} ---")
    print(f"ROC-AUC:   {results[name]['roc_auc']:.4f}")
    print(f"Accuracy:  {results[name]['accuracy']:.4f}")
    print(f"Precision: {results[name]['precision']:.4f}")
    print(f"Recall:    {results[name]['recall']:.4f}")
    print(f"F1-score:  {results[name]['f1']:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=["Not <30d", "<30d"],
                                 zero_division=0))


# --------------------------------------------------------------------
# 7.1 Side-by-side comparison table (baseline vs. balanced)
# --------------------------------------------------------------------
print("\n" + "-" * 78)
print("COMPARISON: baseline vs. class_weight='balanced'")
print("-" * 78)
comparison_df = pd.DataFrame({
    name: {
        "ROC-AUC": r["roc_auc"], "Accuracy": r["accuracy"],
        "Precision": r["precision"], "Recall": r["recall"], "F1": r["f1"],
    } for name, r in results.items()
}).T
print(comparison_df.round(4))
print(
    "\nNote: ROC-AUC is threshold-independent, so it barely changes between "
    "the two models - class_weight mainly shifts WHERE the default 0.5 "
    "threshold falls relative to the predicted probabilities, trading "
    "precision for recall. The balanced model typically achieves much "
    "higher recall (catches more true 30-day readmissions) at the cost of "
    "lower precision (more false alarms)."
)

# --------------------------------------------------------------------
# 7.2 Select the FINAL model used for the rest of the analysis.
#     We choose the class_weight="balanced" model as the primary model:
#     in a hospital-readmission setting, failing to flag a patient who
#     will in fact be readmitted (a false negative) is generally more
#     costly than flagging a patient who turns out fine (a false
#     positive) - see Section 9 for a full clinical discussion. Recall
#     on the minority class is therefore weighted heavily in this choice.
# --------------------------------------------------------------------
FINAL_MODEL_NAME = "balanced (class_weight='balanced')"
final_pipeline = fitted_pipelines[FINAL_MODEL_NAME]
final_result = results[FINAL_MODEL_NAME]
y_pred_final = final_result["y_pred"]
y_proba_final = final_result["y_proba"]

print(f"\n>>> Final model selected for detailed evaluation: {FINAL_MODEL_NAME} <<<")


# ==========================================================================
# 8. EVALUATION (detailed) - ROC curve + confusion matrix for final model
# ==========================================================================
print("\n" + "=" * 78)
print("STEP 8: DETAILED EVALUATION OF THE FINAL MODEL")
print("=" * 78)

# --- Confusion matrix ---
cm = confusion_matrix(y_test, y_pred_final)
tn, fp, fn, tp = cm.ravel()
print("\nConfusion matrix (rows = actual, columns = predicted):")
print(cm)
print(f"\nTrue Negatives  (correctly predicted NOT readmitted): {tn:,}")
print(f"False Positives (predicted readmitted, actually NOT):  {fp:,}")
print(f"False Negatives (predicted NOT readmitted, actually WAS): {fn:,}")
print(f"True Positives  (correctly predicted readmitted):      {tp:,}")

plt.figure(figsize=(5.5, 4.5))
sns.heatmap(cm, annot=True, fmt=",d", cmap="Blues",
            xticklabels=["Pred: Not <30d", "Pred: <30d"],
            yticklabels=["Actual: Not <30d", "Actual: <30d"])
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.title(f"Confusion Matrix - {FINAL_MODEL_NAME}")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.show()
print("\nSaved plot: confusion_matrix.png")

# --- ROC curve ---
fpr, tpr, thresholds = roc_curve(y_test, y_proba_final)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color="darkorange", lw=2,
         label=f"ROC curve (AUC = {final_result['roc_auc']:.4f})")
plt.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--",
         label="Random guessing (AUC = 0.50)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate (Recall)")
plt.title(f"ROC Curve - {FINAL_MODEL_NAME}")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("roc_curve.png", dpi=150)
plt.show()
print("Saved plot: roc_curve.png")


# ==========================================================================
# 9. CLINICAL INTERPRETATION
# ==========================================================================
print("\n" + "=" * 78)
print("STEP 9: CLINICAL INTERPRETATION")
print("=" * 78)
print(f"""
WHAT ROC-AUC MEANS HERE
------------------------
ROC-AUC ({final_result['roc_auc']:.4f}) is the probability that, if we pick one
randomly chosen readmitted patient and one randomly chosen non-readmitted
patient, the model assigns a higher predicted risk score to the readmitted
patient. A value of 0.5 means the model is no better than random guessing;
1.0 means perfect separation. Unlike accuracy, ROC-AUC does not depend on
choosing a specific probability threshold, which makes it a good summary
metric for an imbalanced classification problem like this one.

FALSE NEGATIVES (FN = {fn:,} patients in the test set)
------------------------------------------------------
A false negative is a patient the model predicts will NOT be readmitted
within 30 days, but who actually IS readmitted. This is the most
clinically dangerous type of error: the patient leaves the hospital
without being flagged for extra follow-up (e.g. a post-discharge phone
call, closer medication monitoring, an earlier follow-up appointment),
and may deteriorate at home until an emergency readmission becomes
necessary - worse health outcomes, and often a more expensive and urgent
episode of care than a planned intervention would have been.

FALSE POSITIVES (FP = {fp:,} patients in the test set)
------------------------------------------------------
A false positive is a patient predicted to be at high risk of readmission
who, in reality, does fine and is not readmitted. These errors are not
free either: they can trigger unnecessary follow-up calls or visits,
extra diagnostic testing, added workload for already-stretched clinical
staff, and higher healthcare spending on interventions the patient did
not actually need. At scale, too many false positives can also cause
"alert fatigue," where staff start to distrust or ignore the model's
warnings altogether.

WHY 0.5 IS NOT NECESSARILY THE RIGHT THRESHOLD
------------------------------------------------
By default, LogisticRegression.predict() classifies a patient as
"high risk" only if its predicted probability exceeds 0.5. But 0.5 is
an arbitrary cutoff with no inherent clinical meaning - it simply treats
false negatives and false positives as equally costly. In this setting
they are not: missing a true readmission (FN) can mean a preventable
health crisis, while a false alarm (FP) mainly costs staff time and
money. In practice, hospitals typically LOWER the threshold below 0.5
(e.g. to 0.2-0.3) so that the model flags more patients as high-risk,
accepting more false positives in exchange for catching more true
positives - the right trade-off point should be chosen using the ROC
curve above together with the real-world costs of each error type, not
by defaulting to 0.5.
""")


# ==========================================================================
# 10. MODEL INTERPRETATION - most influential coefficients
# ==========================================================================
print("\n" + "=" * 78)
print("STEP 10: MODEL INTERPRETATION (LOGISTIC REGRESSION COEFFICIENTS)")
print("=" * 78)

# Recover the human-readable feature names produced by the
# ColumnTransformer (numeric features keep their name; one-hot encoded
# categorical features become "column_value").
feature_names = final_pipeline.named_steps["preprocessor"].get_feature_names_out()
coefficients = final_pipeline.named_steps["classifier"].coef_[0]

coef_df = pd.DataFrame({
    "feature": feature_names,
    "coefficient": coefficients
}).sort_values("coefficient", ascending=False)

n_top = 15
print(f"\nTop {n_top} features associated with HIGHER predicted readmission risk "
      "(largest positive coefficients):")
print(coef_df.head(n_top).to_string(index=False))

print(f"\nTop {n_top} features associated with LOWER predicted readmission risk "
      "(largest negative coefficients):")
print(coef_df.tail(n_top).sort_values("coefficient").to_string(index=False))

print("""
IMPORTANT CAVEAT: ASSOCIATION IS NOT CAUSATION
------------------------------------------------
These coefficients describe statistical ASSOCIATIONS learned from
historical hospital data, not proven causal effects. For example, if
"number_inpatient" (prior inpatient visits) has a strong positive
coefficient, it means patients with more past inpatient stays tend to
have higher predicted readmission risk in this dataset - it does NOT
mean that having more inpatient visits directly CAUSES readmission.
The true underlying cause is more likely an unmeasured factor such as
disease severity or chronic illness burden, which independently drives
both past hospitalizations and future readmission risk (i.e.
confounding). These coefficients should guide further clinical
investigation and hypothesis generation, not be treated as proven
causal mechanisms or used in isolation for high-stakes clinical
decisions.
""")


# ==========================================================================
# 11. FINAL SUMMARY
# ==========================================================================
n_features_after_preprocessing = final_pipeline.named_steps["preprocessor"].transform(
    X_train.iloc[:1]
).shape[1]

print("\n" + "=" * 78)
print("FINAL SUMMARY")
print("=" * 78)
print(f"Model used for summary:        {FINAL_MODEL_NAME}")
print(f"Total samples (after cleaning): {len(df):,}")
print(f"  Training samples:             {X_train.shape[0]:,}")
print(f"  Test samples:                 {X_test.shape[0]:,}")
print(f"Number of raw input features:   {len(numeric_features) + len(categorical_features)} "
      f"({len(numeric_features)} numeric + {len(categorical_features)} categorical)")
print(f"Number of features after preprocessing (one-hot expanded): "
      f"{n_features_after_preprocessing}")
print(f"ROC-AUC:                        {final_result['roc_auc']:.4f}")
print(f"Accuracy:                       {final_result['accuracy']:.4f}")
print(f"Precision:                      {final_result['precision']:.4f}")
print(f"Recall:                         {final_result['recall']:.4f}")
print(f"F1-score:                       {final_result['f1']:.4f}")
print("=" * 78)
print("Plots saved to: class_distribution.png, confusion_matrix.png, roc_curve.png")
print("=" * 78)