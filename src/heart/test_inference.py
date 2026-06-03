from heart_inference import predict_heart

# ==========================================
# SAMPLE PATIENT
# ==========================================

patient = {

    "age": 63,
    "sex": "Male",
    "dataset": "Cleveland",
    "cp": "asymptomatic",
    "trestbps": 145,
    "chol": 233,
    "fbs": False,
    "restecg": "normal",
    "thalch": 150,
    "exang": False,
    "oldpeak": 2.3,
    "slope": "downsloping",
    "ca": 0.0,
    "thal": "fixed defect"

}

# ==========================================
# RUN PREDICTION
# ==========================================

result = predict_heart(patient)

print("\nHEART PREDICTION RESULT")
print("-" * 40)

for key, value in result.items():
    print(f"{key}: {value}")