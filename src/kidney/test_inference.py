from kidney_inference import predict_kidney

sample_patient = {

    "age": 45,
    "bp": 80,
    "bgr": 110,
    "bu": 25,
    "sc": 1.0,
    "hemo": 14.5,

    "htn": "no",
    "dm": "no",
    "cad": "no",
    "appet": "good",
    "pe": "no",
    "ane": "no"
}

result = predict_kidney(sample_patient)

print("\nRESULT")
print(result)
print("--------------------------")

sample2 ={
    "age": 55,
    "bp": 80,
    "bgr": 110,
    "bu": 25,
    "sc": 1.0,
    "hemo": 14.5,

    "htn": "yes",
    "dm": "yes",
    "cad": "no",
    "appet": "good",
    "pe": "no",
    "ane": "no"

}
r=predict_kidney(sample2)
print(r)

print("_______________________________________")
sample3 ={
    "age": 70,
    "bp": 160,
    "bgr": 110,
    "bu": 70,
    "sc": 5.0,
    "hemo": 9.9,

    "htn": "no",
    "dm": "yes",
    "cad": "no",
    "appet": "good",
    "pe": "no",
    "ane": "no"

}
r3=predict_kidney(sample3)
print(r3)

print("_______________________________________")