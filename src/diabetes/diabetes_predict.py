import joblib
import pandas as pd

# ==========================================
# LOAD MODELS
# ==========================================

model_1 = joblib.load("../../models/diabetes/DIABETES_M1_randomforest.joblib")

model_2 = joblib.load("../../models/diabetes/DIABETES_M2_risk_model.joblib")

print("\nMODELS LOADED SUCCESSFULLY")

# ==========================================
# SAMPLE PATIENT INPUT
# ==========================================

sample_input = {
    "Pregnancies": 4,
    "Glucose": 160,
    "BloodPressure": 78,
    "SkinThickness": 30,
    "Insulin": 150,
    "BMI": 35.2,
    "DiabetesPedigreeFunction": 0.8,
    "Age": 52
}

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
# CONVERT TO DATAFRAME
# ==========================================

input_df = pd.DataFrame([sample_input])

input_df = input_df[FEATURE_ORDER]

print("\nINPUT DATA")
print(input_df)

# ==========================================
# MODEL 1 PREDICTION
# ==========================================

prediction = model_1.predict(input_df)[0]

probability = model_1.predict_proba(input_df)[0][1] * 100

# ==========================================
# RESULT LOGIC
# ==========================================

if prediction == 1:

    disease_result = "DIABETES DETECTED"

    # ======================================
    # RISK CATEGORY
    # ======================================

    if probability >= 85:
        risk_category = "HIGH RISK"

    elif probability >= 65:
        risk_category = "MODERATE RISK"

    else:
        risk_category = "MILD RISK"

    future_risk = "N/A"

else:

    disease_result = "NO DIABETES DETECTED"

    risk_category = "LOW"

    # ======================================
    # FUTURE RISK PREDICTION
    # ======================================

    future_risk_prob = model_2.predict_proba(input_df)[0][1] * 100

    future_risk = f"{future_risk_prob:.2f}%"

# ==========================================
# FEATURE IMPORTANCE
# ==========================================

feature_importance = model_1.named_steps[
    "classifier"
].feature_importances_

importance_df = pd.DataFrame({
    "Feature": FEATURE_ORDER,
    "Importance": feature_importance
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

top_features = importance_df.head(3)

# ==========================================
# FINAL OUTPUT
# ==========================================

print("\n===================================")
print("DIABETES PREDICTION RESULT")
print("===================================")

print(f"\nPrediction: {disease_result}")

print(f"Diabetes Probability: {probability:.2f}%")

print(f"Risk Category: {risk_category}")

print(f"Future Diabetes Risk: {future_risk}")

print("\nTOP CONTRIBUTING FEATURES:")

for _, row in top_features.iterrows():

    print(f"- {row['Feature']}")

print("\n===================================")