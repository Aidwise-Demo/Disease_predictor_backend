from typing import Union
from fastapi import FastAPI, File, UploadFile, HTTPException
import uuid
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from requests import request
import model
import shutil
import base64
from pydantic import BaseModel

from fastapi import HTTPException, Depends
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

app = FastAPI()
IMAGEDIR = "images/"

# Allow all origins for simplicity. You can restrict it to your specific frontend domain.
origins = [
    "*",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["POST", "GET", "OPTIONS", "DELETE"],
    allow_headers=["*"],
)

# class ImageUpload(BaseModel):
#     file: UploadFile


async def predict_and_process_image(file_path):
    return await model.predict_class_with_heatmap(file_path)
@app.post('/UploadImages')
async def upload_images(file: UploadFile = File(...)):
    try:
        print("Start")
        file.filename = f"{uuid.uuid4()}.png"
        contents = await file.read()

        with open(f"{IMAGEDIR}{file.filename}", "wb") as f:
            f.write(contents)

        # Message = model.predict_class_with_heatmap(f"{IMAGEDIR}{file.filename}")

        image_path = Path("superimposed_image.png")
        if not image_path.is_file():
            return {"error": "Image not found on the server"}

        print("Done sent")
        return {"Detection": "Message"}
    except Exception as e:
        print(e)
        return {"error": str(e)}


# def image_to_base64(image_path: str) -> str:
#     with open(image_path, "rb") as image_file:
#         encoded_image = base64.b64encode(image_file.read())
#         return encoded_image.decode("utf-8")

# def image_to_base64(image_path: str) -> str:
#     with open(image_path, "rb") as image_file:
#         encoded_image = base64.b64encode(image_file.read())
#         return encoded_image

@app.exception_handler(HTTPException)
async def request_validation_exception_handler(request, exc):
    detail = None
    if exc.detail:
        detail = exc.detail
    elif exc.errors():
        detail = jsonable_encoder(exc.errors())
    elif exc.response and exc.response.body:
        return JSONResponse(content=exc.response.body, status_code=exc.status_code)
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": detail},
    )
