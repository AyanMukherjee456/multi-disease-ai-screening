import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)

import joblib

# ==========================================
# LOAD DATASET
# ==========================================

DATA_PATH = "../../data/raw/diabetes.csv"

df = pd.read_csv(DATA_PATH)

print("\nDATASET LOADED SUCCESSFULLY")
print(df.head())

# ==========================================
# BASIC DATA CLEANING
# ==========================================

# Replace impossible zeros with NaN
columns_with_zero_issue = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]

for col in columns_with_zero_issue:
    df[col] = df[col].replace(0, np.nan)

print("\nMISSING VALUES:")
print(df.isnull().sum())

# ==========================================
# FEATURES & TARGET
# ==========================================

FEATURES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age"
]

TARGET = "Outcome"

X = df[FEATURES]
y = df[TARGET]

# ==========================================
# TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# ==========================================
# PREPROCESSING PIPELINE
# ==========================================

numeric_features = FEATURES

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features)
])

# ==========================================
# MODEL 1 — DIABETES DETECTION
# ==========================================

print("\nTRAINING DIABETES DETECTION MODEL...")

model_1 = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    class_weight="balanced",
    random_state=42
)

pipeline_1 = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", model_1)
])

pipeline_1.fit(X_train, y_train)

# ==========================================
# EVALUATION
# ==========================================

y_pred = pipeline_1.predict(X_test)

y_prob = pipeline_1.predict_proba(X_test)[:, 1]

print("\nCONFUSION MATRIX")
print(confusion_matrix(y_test, y_pred))

print("\nCLASSIFICATION REPORT")
print(classification_report(y_test, y_pred))

print("\nACCURACY:", accuracy_score(y_test, y_pred))

print("\nROC AUC SCORE:", roc_auc_score(y_test, y_prob))

# ==========================================
# SAVE MODEL 1
# ==========================================

joblib.dump(
    pipeline_1,
    "../../models/DIABETES_M1_randomforest.joblib"
)

print("\nMODEL 1 SAVED SUCCESSFULLY")

# ==========================================
# MODEL 2 — FUTURE DIABETES RISK
# ==========================================

print("\nTRAINING FUTURE DIABETES RISK MODEL...")

# Create future risk label
risk_label = (
    (df["Glucose"] > 140).astype(int) +
    (df["BMI"] > 30).astype(int) +
    (df["Age"] > 45).astype(int) +
    (df["DiabetesPedigreeFunction"] > 0.5).astype(int)
)

risk_label = (risk_label >= 2).astype(int)

X_risk = X.copy()
y_risk = risk_label

Xr_train, Xr_test, yr_train, yr_test = train_test_split(
    X_risk,
    y_risk,
    test_size=0.2,
    random_state=42
)

model_2 = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.05,
    random_state=42
)

pipeline_2 = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", model_2)
])

pipeline_2.fit(Xr_train, yr_train)

# ==========================================
# EVALUATE MODEL 2
# ==========================================

yr_pred = pipeline_2.predict(Xr_test)

yr_prob = pipeline_2.predict_proba(Xr_test)[:, 1]

print("\nFUTURE RISK MODEL ACCURACY:")
print(accuracy_score(yr_test, yr_pred))

print("\nFUTURE RISK ROC AUC:")
print(roc_auc_score(yr_test, yr_prob))

# ==========================================
# SAVE MODEL 2
# ==========================================

joblib.dump(
    pipeline_2,
    "../../models/DIABETES_M2_risk_model.joblib"
)

print("\nMODEL 2 SAVED SUCCESSFULLY")

# ==========================================
# FEATURE IMPORTANCE
# ==========================================

feature_importance = pipeline_1.named_steps[
    "classifier"
].feature_importances_

importance_df = pd.DataFrame({
    "Feature": FEATURES,
    "Importance": feature_importance
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print("\nFEATURE IMPORTANCE")
print(importance_df)

# ==========================================
# SAVE FEATURE IMPORTANCE
# ==========================================

importance_df.to_csv(
    "../../reports/diabetes_feature_importance.csv",
    index=False
)

print("\nPIPELINE COMPLETED SUCCESSFULLY")