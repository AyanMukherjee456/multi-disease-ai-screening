from diabetes_inference import predict_diabetes

sample_patient = {

    "Pregnancies": 4,
    "Glucose": 160,
    "BloodPressure": 78,
    "SkinThickness": 30,
    "Insulin": 150,
    "BMI": 35.2,
    "DiabetesPedigreeFunction": 0.8,
    "Age": 52

}

result = predict_diabetes(
    sample_patient
)

print(result)