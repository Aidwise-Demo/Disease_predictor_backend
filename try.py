from HeartAPI import model

data_values = {
    "age": 70,
    "sex": 1,
    "chest_pain_type": 4,
    "blood_pressure": 130,
    "cholesterol": 322,
    "fbs_over_120": 0,
    "ekg_results": 2,
    "max_heart_rate": 109,
    "exercise_angina": 0,
    "st_depression": 2.4,
    "slope_of_st": 2,
    "num_vessels_fluro": 3,
    "thallium": 3
}

# print(data_values.values())
result = model.predict_heart_disease([data_values.values()])
print(f"The predicted result is: {result}")
