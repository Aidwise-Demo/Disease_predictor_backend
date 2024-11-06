from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.responses import FileResponse
from pathlib import Path
import shutil
import os
import DRAPI.model as DRmodel
import CovidAPI.model as Covidmodel
import ParserScript.script as ParserScript
from HeartAPI import model as HeartModel
from pydantic import BaseModel
import base64
from removing_files import remove_old_files
from fastapi import WebSocket
import asyncio
 
 

# Call the function to remove old files
remove_old_files()


app = FastAPI()

# Allow all origins for demonstration purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with your client's URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_FOLDER = "uploads"
Path(UPLOAD_FOLDER).mkdir(parents=True, exist_ok=True)
UPLOAD_FOLDERCovid = "uploadsCovid"
Path(UPLOAD_FOLDERCovid).mkdir(parents=True, exist_ok=True)


@app.get("/", response_class=JSONResponse)
async def read_form():
    return {"message": "Server is running"}


@app.post("/DRuploadfile")
async def create_upload_file(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    image_path = Path("DRAPI/superimposed_image.png")
    if image_path.is_file():
        os.remove(image_path)

    async def run_prediction():
        return await DRmodel.predict_class_with_heatmap(f"{UPLOAD_FOLDER}/{file.filename}")

    prediction_result = await run_prediction()

    image_path = Path("DRAPI/superimposed_image.png")
    if not image_path.is_file():
        return {"error": "Image not found on the server"}

    byte_string = image_to_byte_string("DRAPI/superimposed_image.png")

    print(file.filename)
    return {"filename": file.filename, "Detection": prediction_result, "image": byte_string}


# Allow all origins for demonstration purposes


@app.post("/Coviduploadfile")
async def create_upload_file_fn(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_FOLDERCovid, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    image_path = Path("CovidAPI/superimposed_image.png")
    if image_path.is_file():
        os.remove(image_path)

    async def run_prediction():
        return await Covidmodel.predict_class_with_heatmap(f"{UPLOAD_FOLDERCovid}/{file.filename}")

    prediction_result = await run_prediction()

    image_path = Path("CovidAPI/superimposed_image.png")
    if not image_path.is_file():
        return {"error": "Image not found on the server"}

    byte_string = image_to_byte_string("CovidAPI/superimposed_image.png")

    print(file.filename)
    return {"filename": file.filename, "Detection": prediction_result, "image": byte_string}
    # return {"filename": files.filename}


def image_to_byte_string(image_path):
    # Read the image file
    with open(image_path, "rb") as image_file:
        image_data = image_file.read()

    # Encode the image data in Base64
    encoded_string = base64.b64encode(image_data).decode("utf-8")

    print(f"Base64 encoded image string: {encoded_string}")
    return encoded_string


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


# Allow all origins for demonstration purposes



@app.post("/predict_heart_disease")
async def predict_heart_disease(features: InputFeatures):
    try:
        # print(features)
        Values = validate_and_map_input(**(features.model_dump()))
        # print(Values)

        prediction = HeartModel.predict_heart_disease(Values)
        # prediction = HeartModel.predict_heart_disease([features.model_dump().values()])
        # print(prediction)
        return {"prediction": prediction}

    except Exception as e:
        print(e)



def validate_and_map_input(age, sex, chest_pain_type, blood_pressure, cholesterol, fbs_over_120, ekg_results,
                            max_heart_rate, exercise_angina, st_depression, slope_of_st, num_vessels_fluro, thallium):
    # Define mappings for text values to numerical values
    mappings = {
        "sex": {"female": 0,"male" : 1},
        "chest_pain_type": {"typical angina": 1, "atypical angina" : 2, "non-anginal pain": 3, "asymptomatic": 4},
        "ekg_results": {"normal": 0, "ST-T wave abnormality": 1, "left ventricular hypertrophy": 2},
        "fbs": {"true": 1, "false": 0},
        "exercise_angina": {"no": 0, "yes": 1},
        "slope_of_st": {"upsloping" : 1, "flat": 2, "downsloping": 3},
        "thallium": {"normal": 3, "fixed defect": 6, "reversible defect": 7}
        # Add mappings for other attributes as needed
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

# Allow all origins for demonstration purposes


@app.post("/Parserupload")
async def upload(file: UploadFile = File(...)):
    try:
        filename = file.filename
        save_location = f"input/{filename}"
        with open(save_location, "wb") as file_object:
            file_object.write(file.file.read())

        output_file = ParserScript.process_837_file(save_location)


        return FileResponse(f"{output_file}", filename=output_file, media_type='application/octet-stream')
    except Exception as e:
        # Handle exceptions as needed
        print(e)
        
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # Simulate sending live data every 5 seconds
            await websocket.send_text("Live update: Hello, this message is sent every 5 seconds!")
            await asyncio.sleep(5)  # Wait for 5 seconds
    except Exception as e:
        print(f"Error: {e}")
    finally:
        await websocket.close()
