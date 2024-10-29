from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pathlib import Path
import os
from dotenv import load_dotenv
import DRAPI.model as DRmodel
import CovidAPI.model as Covidmodel
import ParserScript.script as ParserScript
from HeartAPI import model as HeartModel
from pydantic import BaseModel
import cloudinary
import cloudinary.uploader
import cloudinary.api
import requests
import numpy as np
import cv2
import time

app = FastAPI()

# Allow all origins for demonstration purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with your client's URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load environment variables from .env file
load_dotenv()

# Configure Cloudinary
cloudinary.config(
    cloud_name=os.getenv('CLOUDINARY_CLOUD_NAME_AIDWISE_DEMO'),
    api_key=os.getenv('CLOUDINARY_API_KEY_AIDWISE_DEMO'),
    api_secret=os.getenv('CLOUDINARY_API_SECRET_AIDWISE_DEMO'),
    secure=True
)

# Helper function to download an image from a URL and convert it to OpenCV format
async def download_image(image_url):
    response = requests.get(image_url)
    if response.status_code == 200:
        image_array = np.frombuffer(response.content, np.uint8)
        return cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    else:
        raise HTTPException(status_code=500, detail="Unable to download the image")

@app.get("/", response_class=JSONResponse)
async def read_form():
    return {"message": "Server is running"}

# Diabetic Retinopathy Detection API
from fastapi import HTTPException, UploadFile, File
import cloudinary.uploader
import cv2
import numpy as np
import time

from fastapi import HTTPException, UploadFile, File
import cloudinary.uploader
import cv2
import numpy as np
import time
import os
from datetime import datetime
import uuid

@app.post("/DRuploadfile")
async def create_upload_file(file: UploadFile = File(...)):
    try:
        # Generate a unique filename based on timestamp and UUID
        
        unique_filename = f"superimposed_image_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4()}.png"

        # Upload the original file to Cloudinary
        upload_result = cloudinary.uploader.upload(file.file, folder="DRAPI")
        image_url = upload_result["secure_url"]

        # Wait for 2 seconds to allow the image to be fully processed
        time.sleep(2)

        # Download the image for OpenCV processing
        image = await download_image(image_url)

        # Run prediction with the downloaded image
        prediction_result, superimposed_image = await DRmodel.predict_class_with_heatmap(image)

        # Save superimposed image locally with the unique filename
        superimposed_path = unique_filename  # Save in the current working directory
        if isinstance(superimposed_image, np.ndarray):
            cv2.imwrite(superimposed_path, superimposed_image)

        # Upload superimposed image to Cloudinary
        heatmap_upload = cloudinary.uploader.upload(superimposed_path, folder="DRAPI")
        heatmap_url = heatmap_upload["secure_url"]

        return {
            "filename": unique_filename,
            "Detection": prediction_result,
            "image": heatmap_url
        }

    except Exception as e:
        # Improved error logging
        print(f"Error with file: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# COVID Detection API
@app.post("/Coviduploadfile")
async def create_upload_file_fn(file: UploadFile = File(...)):
    try:
        # Upload the file to Cloudinary
        upload_result = cloudinary.uploader.upload(file.file, folder="CovidAPI")
        image_url = upload_result["secure_url"]

        # Wait for 2 seconds to allow the image to be fully processed
        time.sleep(2)

        # Download the image for OpenCV processing
        image = await download_image(image_url)

        # Run prediction with the downloaded image
        prediction_result, superimposed_image = await Covidmodel.predict_class_with_heatmap(image)

        # Save superimposed image locally if returned as an array
        superimposed_path = "superimposed_image.png"
        if isinstance(superimposed_image, np.ndarray):
            cv2.imwrite(superimposed_path, superimposed_image)
            superimposed_image = superimposed_path  # Update variable to path

        # Upload superimposed image to Cloudinary
        heatmap_upload = cloudinary.uploader.upload(superimposed_image, folder="CovidAPI")
        heatmap_url = heatmap_upload["secure_url"]

        return {"filename": file.filename, "Detection": prediction_result, "image": heatmap_url}
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))

# Heart Disease Prediction API
class InputFeatures(BaseModel):
    age: int
    sex: str
    chest_pain_type: str
    blood_pressure: int
    cholesterol: int
    fbs_over_120: str
    ekg_results: str
    max_heart_rate: int
    exercise_angina: str
    st_depression: float
    slope_of_st: str
    num_vessels_fluro: int
    thallium: str

@app.post("/predict_heart_disease")
async def predict_heart_disease(features: InputFeatures):
    try:
        Values = validate_and_map_input(**(features.dict()))
        prediction = HeartModel.predict_heart_disease(Values)
        return {"prediction": prediction}
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))

# Helper function to validate and map input features for heart disease prediction
def validate_and_map_input(age, sex, chest_pain_type, blood_pressure, cholesterol, fbs_over_120, ekg_results,
                           max_heart_rate, exercise_angina, st_depression, slope_of_st, num_vessels_fluro, thallium):
    # Define mappings for text values to numerical values
    mappings = {
        "sex": {"female": 0, "male": 1},
        "chest_pain_type": {"typical angina": 1, "atypical angina": 2, "non-anginal pain": 3, "asymptomatic": 4},
        "ekg_results": {"normal": 0, "ST-T wave abnormality": 1, "left ventricular hypertrophy": 2},
        "fbs": {"true": 1, "false": 0},
        "exercise_angina": {"no": 0, "yes": 1},
        "slope_of_st": {"upsloping": 1, "flat": 2, "downsloping": 3},
        "thallium": {"normal": 3, "fixed defect": 6, "reversible defect": 7}
    }

    # Validate and map the input
    mapped_values = [
        age,
        mappings["sex"].get(sex, None),
        mappings["chest_pain_type"].get(chest_pain_type, None),
        blood_pressure,
        cholesterol,
        mappings["fbs"].get(fbs_over_120, None),
        mappings["ekg_results"].get(ekg_results, None),
        max_heart_rate,
        mappings["exercise_angina"].get(exercise_angina, None),
        st_depression,
        mappings["slope_of_st"].get(slope_of_st, None),
        num_vessels_fluro,
        mappings["thallium"].get(thallium, None)
    ]

    return mapped_values

# Parser File Upload API
@app.post("/Parserupload")
async def upload(file: UploadFile = File(...)):
    try:
        # Upload the file to Cloudinary
        upload_result = cloudinary.uploader.upload(file.file, folder="ParserAPI")
        file_url = upload_result["secure_url"]

        # Process the file using ParserScript
        output_file = ParserScript.process_837_file(file_url)

        # Return the processed file
        return FileResponse(f"{output_file}", filename=output_file, media_type='application/octet-stream')
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))
