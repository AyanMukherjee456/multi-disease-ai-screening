import joblib
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
# ==========================================
# LOAD MODELS ONCE
# ==========================================

MODEL_1 = joblib.load(
    BASE_DIR / "models/kidney/KIDNEY_M1_detection.joblib"
)

MODEL_2 = joblib.load(
    BASE_DIR / "models/kidney/KIDNEY_M2_risk_model.joblib"
)

# ==========================================
# FEATURE ORDER
# ==========================================

FEATURE_ORDER = [
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

# ==========================================
# MAIN INFERENCE FUNCTION
# ==========================================

def predict_kidney(patient_data):

    # --------------------------------------
    # Convert input to dataframe
    # --------------------------------------

    input_df = pd.DataFrame([patient_data])

    input_df = input_df[FEATURE_ORDER]

    # --------------------------------------
    # MODEL 1
    # CKD Detection
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

    feature_names = (
        MODEL_1.named_steps[
            "preprocessor"
        ].get_feature_names_out()
    )

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importance
    })

    importance_df = importance_df.sort_values(
        by="Importance",
        ascending=False
    )

    top_features = (
        importance_df["Feature"]
        .head(5)
        .tolist()
    )

    # --------------------------------------
    # If CKD Detected
    # --------------------------------------

    if prediction == 1:

        disease_result = (
            "KIDNEY DISEASE DETECTED"
        )

        if probability >= 85:
            risk_category = "HIGH RISK"

        elif probability >= 65:
            risk_category = "MODERATE RISK"

        else:
            risk_category = "MILD RISK"

        future_risk = "N/A"

    # --------------------------------------
    # If No CKD
    # --------------------------------------

    else:

        disease_result = (
            "NO KIDNEY DISEASE DETECTED"
        )

        risk_category = "LOW"

        future_risk_prob = (
            MODEL_2.predict_proba(
                input_df
            )[0][1] * 100
        )

        future_risk = float(
            round(
                future_risk_prob,
                2
            )
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