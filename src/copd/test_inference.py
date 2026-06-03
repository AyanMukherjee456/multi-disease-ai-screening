from copd_inference import predict_copd

sample = {

    "Age": 45,
    "Gender": "Male",
    "SmokingStatus": "never",
    "SmokingYears": 0,
    "BMI": 24.5,
    "FEV1_pct": 95,
    "FVC_pct": 98,
    "FEV1_FVC": 0.82,
    "ChronicCough": 0,
    "CRP_mg_L": 1.2,
    "SpO2": 98

}

result = predict_copd(sample)

print(result)