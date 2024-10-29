from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
# from fastapi.staticfiles import StaticFiles
# from fastapi.templating import Jinja2Templates
from pathlib import Path
import shutil
import os
import model
import concurrent.futures
import cv2
import io
from PIL import Image
import asyncio
import base64
app = FastAPI()

# Allow all origins for demonstration purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with your client's URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# app.mount("/static", StaticFiles(directory="static"), name="static")

# templates = Jinja2Templates(directory="templates")

UPLOAD_FOLDER = "uploads"
Path(UPLOAD_FOLDER).mkdir(parents=True, exist_ok=True)


@app.get("/", response_class=JSONResponse)
async def read_form():
    return {"message": "Server is running"}

@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    image_path = Path("superimposed_image.png")
    if image_path.is_file():
        os.remove(image_path)

    async def run_prediction():
        return await model.predict_class_with_heatmap(f"{UPLOAD_FOLDER}/{file.filename}")

    prediction_result = await run_prediction()

    image_path = Path("superimposed_image.png")
    if not image_path.is_file():
        return {"error": "Image not found on the server"}

    byte_string = image_to_byte_string("superimposed_image.png")

    print(file.filename)
    return {"filename": file.filename, "Detection": prediction_result, "image": byte_string}


# convert image to binary string 

def image_to_byte_string(image_path):

    # Read the image file
    # image_path = "path/to/image.jpg"
    with open(image_path, "rb") as image_file:
        image_data = image_file.read()

    # Encode the image data in Base64
    encoded_string = base64.b64encode(image_data).decode("utf-8")

    print(f"Base64 encoded image string: {encoded_string}")
    return encoded_string
