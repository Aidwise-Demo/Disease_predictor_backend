import pandas as pd
import pickle


def predict_heart_disease(features: list):
    # Load the model using pickle
    with open('HeartAPI/model.pkl', 'rb') as model_file:
        loaded_model = pickle.load(model_file)
        
    prediction = loaded_model.predict([features])

    print(prediction)
    # per = np.argmax(prediction, axis=1)
    if prediction[0] == 0:
        return 'Heart Disease Not Detected'
    else:
        return 'Heart Disease Detected'