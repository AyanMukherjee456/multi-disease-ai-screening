import joblib
import pandas as pd

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
# ==========================================
# LOAD MODELS ONCE
# ==========================================

MODEL_1 = joblib.load(
    BASE_DIR / "models/diabetes/DIABETES_M1_randomforest.joblib"
)

MODEL_2 = joblib.load(
    BASE_DIR / "models/diabetes/DIABETES_M2_risk_model.joblib"
)

# ==========================================
# FEATURE ORDER
# ==========================================

FEATURE_ORDER = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age"
]

# ==========================================
# MAIN INFERENCE FUNCTION
# ==========================================

def predict_diabetes(patient_data):

    # --------------------------------------
    # Convert input to dataframe
    # --------------------------------------

    input_df = pd.DataFrame([patient_data])

    input_df = input_df[FEATURE_ORDER]

    # --------------------------------------
    # MODEL 1
    # Diabetes Detection
    # --------------------------------------

    prediction = MODEL_1.predict(input_df)[0]

    probability = (
        MODEL_1.predict_proba(input_df)[0][1] * 100
    )

    # --------------------------------------
    # Feature Importance
    # --------------------------------------

    importance = MODEL_1.named_steps[
        "classifier"
    ].feature_importances_

    importance_df = pd.DataFrame({
        "Feature": FEATURE_ORDER,
        "Importance": importance
    })

    importance_df = importance_df.sort_values(
        by="Importance",
        ascending=False
    )

    top_features = (
        importance_df["Feature"]
        .head(3)
        .tolist()
    )

    # --------------------------------------
    # If Diabetes Detected
    # --------------------------------------

    if prediction == 1:

        disease_result = "DIABETES DETECTED"

        if probability >= 85:
            risk_category = "HIGH RISK"

        elif probability >= 65:
            risk_category = "MODERATE RISK"

        else:
            risk_category = "MILD RISK"

        future_risk = "N/A"

    # --------------------------------------
    # If No Diabetes
    # --------------------------------------

    else:

        disease_result = "NO DIABETES DETECTED"

        risk_category = "LOW"

        future_risk_prob = (
            MODEL_2.predict_proba(input_df)[0][1] * 100
        )

        future_risk = round(
            future_risk_prob,
            2
        )

    # --------------------------------------
    # Final Response
    # --------------------------------------

    result = {

        "prediction": disease_result,

        "probability": round(
            float(probability),
            2
        ),

        "risk_category": risk_category,

        "future_risk": future_risk,

        "top_features": top_features

    }

    return result