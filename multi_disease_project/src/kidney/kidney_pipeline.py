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
# LOAD ARFF DATASET
# ==========================================

DATA_PATH = "../../data/raw/chronic_kidney_disease.arff"

print("\nLOADING CKD DATASET...")

with open(DATA_PATH, "r") as f:
    lines = f.readlines()

data_start = 0

for i, line in enumerate(lines):
    if line.strip().lower() == "@data":
        data_start = i + 1
        break

data_lines = lines[data_start:]

columns = [
    "age",
    "bp",
    "sg",
    "al",
    "su",
    "rbc",
    "pc",
    "pcc",
    "ba",
    "bgr",
    "bu",
    "sc",
    "sod",
    "pot",
    "hemo",
    "pcv",
    "wbcc",
    "rbcc",
    "htn",
    "dm",
    "cad",
    "appet",
    "pe",
    "ane",
    "class"
]

rows = []

for line in data_lines:

    line = line.strip()

    if line == "":
        continue

    parts = line.split(",")

    if len(parts) == 25:
        rows.append(parts)

df = pd.DataFrame(rows, columns=columns)

print("\nDATASET LOADED SUCCESSFULLY")
print(df.head())

# ==========================================
# BASIC CLEANING
# ==========================================

print("\nCLEANING DATA...")

df = df.replace("?", np.nan)

df = df.apply(
    lambda col: col.map(
        lambda x: x.strip()
        if isinstance(x, str)
        else x
    )
)

df = df.replace({
    " yes": "yes",
    " no": "no",
    "\tyes": "yes",
    "\tno": "no",
    "ckd,": "ckd",
    "notckd,": "notckd"
})

# ==========================================
# FEATURE SELECTION
# ==========================================

FEATURES = [
    "age",
    "bp",
    "bgr",
    "bu",
    "sc",
    "hemo",
    "htn",
    "dm",
    "cad",
    "appet",
    "pe",
    "ane"
]

TARGET = "class"

df = df[FEATURES + [TARGET]]

# ==========================================
# TARGET CLEANING
# ==========================================

df[TARGET] = (
    df[TARGET]
    .astype(str)
    .str.strip()
)

df[TARGET] = (
    df[TARGET] == "ckd"
).astype(int)

# ==========================================
# NUMERIC CONVERSION
# ==========================================

numeric_features = [
    "age",
    "bp",
    "bgr",
    "bu",
    "sc",
    "hemo"
]

categorical_features = [
    "htn",
    "dm",
    "cad",
    "appet",
    "pe",
    "ane"
]

for col in numeric_features:

    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )

print("\nTARGET DISTRIBUTION")
print(df[TARGET].value_counts())

# ==========================================
# FEATURES & TARGET
# ==========================================

X = df[FEATURES]
y = df[TARGET]

# ==========================================
# TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42
)

# ==========================================
# PREPROCESSOR
# ==========================================

numeric_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)

categorical_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            numeric_transformer,
            numeric_features
        ),
        (
            "cat",
            categorical_transformer,
            categorical_features
        )
    ]
)

# ==========================================
# MODEL 1
# CKD DETECTION
# ==========================================

print("\nTRAINING CKD DETECTION MODELS...")

models = {

    "RandomForest":

    RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        class_weight="balanced",
        random_state=42
    ),

    "GradientBoosting":

    GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        random_state=42
    )
}

best_pipeline = None
best_name = None
best_auc = 0

for name, model in models.items():

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "classifier",
                model
            )
        ]
    )

    pipeline.fit(
        X_train,
        y_train
    )

    y_pred = pipeline.predict(
        X_test
    )

    y_prob = pipeline.predict_proba(
        X_test
    )[:, 1]

    auc = roc_auc_score(
        y_test,
        y_prob
    )

    print(f"\n{name}")
    print(
        "Accuracy:",
        accuracy_score(
            y_test,
            y_pred
        )
    )

    print(
        "ROC AUC:",
        auc
    )

    if auc > best_auc:

        best_auc = auc
        best_pipeline = pipeline
        best_name = name

print("\nBEST MODEL:", best_name)
print("BEST ROC AUC:", best_auc)

# ==========================================
# FINAL EVALUATION
# ==========================================

y_pred = best_pipeline.predict(
    X_test
)

y_prob = best_pipeline.predict_proba(
    X_test
)[:, 1]

print("\nCONFUSION MATRIX")
print(
    confusion_matrix(
        y_test,
        y_pred
    )
)

print("\nCLASSIFICATION REPORT")
print(
    classification_report(
        y_test,
        y_pred
    )
)

print(
    "\nFINAL ACCURACY:",
    accuracy_score(
        y_test,
        y_pred
    )
)

print(
    "\nFINAL ROC AUC:",
    roc_auc_score(
        y_test,
        y_prob
    )
)

# ==========================================
# SAVE MODEL 1
# ==========================================

joblib.dump(
    best_pipeline,
    "../../models/kidney/KIDNEY_M1_detection.joblib"
)

print("\nMODEL 1 SAVED SUCCESSFULLY")

# ==========================================
# MODEL 2
# FUTURE CKD RISK
# ==========================================

print("\nTRAINING FUTURE CKD RISK MODEL...")

risk_score = (

    (df["age"] > 60).astype(int)

    +

    (df["bp"] > 140).astype(int)

    +

    (df["bgr"] > 140).astype(int)

    +

    (df["bu"] > 50).astype(int)

    +

    (df["sc"] > 1.5).astype(int)

    +

    (df["hemo"] < 12).astype(int)

    +

    (df["htn"] == "yes").astype(int)

    +

    (df["dm"] == "yes").astype(int)

)

risk_label = (
    risk_score >= 3
).astype(int)

X_risk = X.copy()
y_risk = risk_label

Xr_train, Xr_test, yr_train, yr_test = train_test_split(
    X_risk,
    y_risk,
    test_size=0.20,
    random_state=42
)

risk_model = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.05,
    random_state=42
)

risk_pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            risk_model
        )
    ]
)

risk_pipeline.fit(
    Xr_train,
    yr_train
)

yr_pred = risk_pipeline.predict(
    Xr_test
)

yr_prob = risk_pipeline.predict_proba(
    Xr_test
)[:, 1]

print(
    "\nFUTURE RISK ACCURACY:"
)

print(
    accuracy_score(
        yr_test,
        yr_pred
    )
)

print(
    "\nFUTURE RISK ROC AUC:"
)

print(
    roc_auc_score(
        yr_test,
        yr_prob
    )
)

# ==========================================
# SAVE MODEL 2
# ==========================================

joblib.dump(
    risk_pipeline,
    "../../models/kidney/KIDNEY_M2_risk_model.joblib"
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
        best_pipeline
        .named_steps["preprocessor"]
        .get_feature_names_out()
    )

    importance = (
        best_pipeline
        .named_steps["classifier"]
        .feature_importances_
    )

    importance_df = pd.DataFrame({

        "Feature": feature_names,

        "Importance": importance

    })

    importance_df = (
        importance_df
        .sort_values(
            by="Importance",
            ascending=False
        )
    )

    print("\nTOP FEATURES")
    print(
        importance_df.head(20)
    )

    importance_df.to_csv(
        "../../reports/kidney_feature_importance.csv",
        index=False
    )

print("\nPIPELINE COMPLETED SUCCESSFULLY")