import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline

from sklearn.compose import ColumnTransformer

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder
)

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)

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

DATA_PATH = "../../data/raw/heart disease Uci.csv"

df = pd.read_csv(DATA_PATH)

print("\nDATASET LOADED SUCCESSFULLY")
print(df.head())

# ==========================================
# BASIC CLEANING
# ==========================================

if "id" in df.columns:
    df.drop(columns=["id"], inplace=True)

# Convert target to binary
df["num"] = (df["num"] > 0).astype(int)

print("\nTARGET DISTRIBUTION")
print(df["num"].value_counts())

# ==========================================
# FEATURES & TARGET
# ==========================================

TARGET = "num"

FEATURES = [
    col for col in df.columns
    if col != TARGET
]

X = df[FEATURES]
y = df[TARGET]

# ==========================================
# FEATURE TYPES
# ==========================================

numeric_features = [
    "age",
    "trestbps",
    "chol",
    "thalch",
    "oldpeak",
    "ca"
]

categorical_features = [
    "sex",
    "dataset",
    "cp",
    "fbs",
    "restecg",
    "exang",
    "slope",
    "thal"
]

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
# PREPROCESSOR
# ==========================================

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# ==========================================
# MODEL 1 - HEART DISEASE DETECTION
# ==========================================

print("\nTRAINING HEART DISEASE DETECTION MODELS...")

models = {

    "RandomForest": RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        class_weight="balanced",
        random_state=42
    ),

    "GradientBoosting": GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        random_state=42
    )
}

best_model = None
best_pipeline = None
best_auc = 0
best_name = None

for name, model in models.items():

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", model)
        ]
    )

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    y_prob = pipeline.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_prob)

    print(f"\n{name}")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("ROC AUC:", auc)

    if auc > best_auc:
        best_auc = auc
        best_pipeline = pipeline
        best_model = model
        best_name = name

print("\nBEST MODEL:", best_name)
print("BEST ROC AUC:", best_auc)

# ==========================================
# FINAL EVALUATION
# ==========================================

y_pred = best_pipeline.predict(X_test)

y_prob = best_pipeline.predict_proba(X_test)[:, 1]

print("\nCONFUSION MATRIX")
print(confusion_matrix(y_test, y_pred))

print("\nCLASSIFICATION REPORT")
print(classification_report(y_test, y_pred))

print("\nFINAL ACCURACY")
print(accuracy_score(y_test, y_pred))

print("\nFINAL ROC AUC")
print(roc_auc_score(y_test, y_prob))

# ==========================================
# SAVE MODEL 1
# ==========================================

joblib.dump(
    best_pipeline,
    "../../models/heart/HEART_M1_detection.joblib"
)

print("\nMODEL 1 SAVED SUCCESSFULLY")

# ==========================================
# MODEL 2 - FUTURE HEART RISK
# ==========================================

print("\nTRAINING FUTURE HEART RISK MODEL...")

risk_score = (

    (df["age"] > 55).astype(int)

    +

    (df["trestbps"] > 140).astype(int)

    +

    (df["chol"] > 240).astype(int)

    +

    (df["oldpeak"] > 2).astype(int)

    +

    (df["exang"].astype(str) == "True").astype(int)
)

risk_label = (risk_score >= 2).astype(int)

X_risk = X.copy()
y_risk = risk_label

Xr_train, Xr_test, yr_train, yr_test = train_test_split(
    X_risk,
    y_risk,
    test_size=0.2,
    random_state=42
)

risk_model = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.05,
    random_state=42
)

risk_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", risk_model)
    ]
)

risk_pipeline.fit(Xr_train, yr_train)

yr_pred = risk_pipeline.predict(Xr_test)

yr_prob = risk_pipeline.predict_proba(Xr_test)[:, 1]

print("\nFUTURE RISK MODEL ACCURACY")
print(accuracy_score(yr_test, yr_pred))

print("\nFUTURE RISK ROC AUC")
print(roc_auc_score(yr_test, yr_prob))

# ==========================================
# SAVE MODEL 2
# ==========================================

joblib.dump(
    risk_pipeline,
    "../../models/heart/HEART_M2_risk_model.joblib"
)

print("\nMODEL 2 SAVED SUCCESSFULLY")

# ==========================================
# FEATURE IMPORTANCE
# ==========================================

if hasattr(
    best_pipeline.named_steps["classifier"],
    "feature_importances_"
):

    feature_names = (
        best_pipeline.named_steps[
            "preprocessor"
        ].get_feature_names_out()
    )

    importance = (
        best_pipeline.named_steps[
            "classifier"
        ].feature_importances_
    )

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importance
    })

    importance_df = importance_df.sort_values(
        by="Importance",
        ascending=False
    )

    print("\nTOP FEATURES")
    print(importance_df.head(20))

    importance_df.to_csv(
        "../../reports/heart_feature_importance.csv",
        index=False
    )

print("\nPIPELINE COMPLETED SUCCESSFULLY")