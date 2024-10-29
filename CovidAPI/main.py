
from typing import Union
from fastapi import FastAPI, File, UploadFile
import uuid
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from requests import request
import model

app = FastAPI(max_file_size=10000000)
IMAGEDIR = "images/"

# Allow all origins for simplicity. You can restrict it to your specific frontend domain.
origins = [
    "http://localhost:5500",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}


@app.post('/UploadImages')
async def upload_file(file: UploadFile = File(...)):
    file.filename = f"{uuid.uuid4()}.png"
    contents = await file.read()

    print(file.filename)
    with open(f"{IMAGEDIR}{file.filename}", "wb") as f:
        f.write(contents)

    Message = model.predict_class_with_heatmap(f"{IMAGEDIR}{file.filename}")

    image_path = Path("superimposed_image.png")
    if not image_path.is_file():
        return {"error": "Image not found on the server"}

    return FileResponse(image_path, media_type="image/png")

    # return {"filename": file.filename, "Detection": Message}
