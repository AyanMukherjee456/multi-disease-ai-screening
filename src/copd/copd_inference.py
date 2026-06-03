import joblib
import pandas as pd

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
# ==========================================
# LOAD MODELS
# ==========================================

MODEL_1 = joblib.load(
    BASE_DIR / "models/copd/COPD_M1_randomforest.joblib"
)

MODEL_2 = joblib.load(
    BASE_DIR / "models/copd/COPD_M2_gb_regressor.joblib"
)

MODEL_3 = joblib.load(
    BASE_DIR / "models/copd/COPD_M3_risk_model.joblib"
)

# ==========================================
# FEATURE ORDER
# ==========================================

FEATURE_ORDER = [
    "Age",
    "Gender",
    "SmokingStatus",
    "SmokingYears",
    "BMI",
    "FEV1_pct",
    "FVC_pct",
    "FEV1_FVC",
    "ChronicCough",
    "CRP_mg_L",
    "SpO2"
]

# ==========================================
# MAIN FUNCTION
# ==========================================

def predict_copd(patient_data):

    input_df = pd.DataFrame([patient_data])

    input_df = input_df[FEATURE_ORDER]

    # ------------------------------
    # MODEL 1
    # ------------------------------

    prediction = MODEL_1.predict(
        input_df
    )[0]

    probability = (
        MODEL_1.predict_proba(
            input_df
        )[0][1] * 100
    )

    # ------------------------------
    # MODEL 2
    # ------------------------------

    severity = None

    if prediction == 1:

        severity = float(
            MODEL_2.predict(
                input_df
            )[0]
        )

    # ------------------------------
    # MODEL 3
    # ------------------------------

    future_risk = float(

        MODEL_3.predict_proba(
            input_df
        )[0][1] * 100

    )

    # ------------------------------
    # RESULT
    # ------------------------------

    if prediction == 1:

        disease_result = (
            "COPD DETECTED"
        )

        if probability >= 85:
            risk_category = "HIGH RISK"

        elif probability >= 65:
            risk_category = "MODERATE RISK"

        else:
            risk_category = "MILD RISK"

    else:

        disease_result = (
            "NO COPD DETECTED"
        )

        risk_category = "LOW"

    result = {

        "prediction":
            disease_result,

        "probability":
            round(
                float(probability),
                2
            ),

        "severity_percentage":
            round(
                severity,
                2
            ) if severity is not None else None,

        "future_risk":
            round(
                float(future_risk),
                2
            ),

        "risk_category":
            risk_category

    }

    return result