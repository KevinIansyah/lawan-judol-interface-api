from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from app.model.classifier import GamblingClassifier
from app.schemas.input import InputText
from app.utils.text_cleaner import clean_text
from app.utils.file_predicor import process_file
from pathlib import Path
import uuid
import os


app = FastAPI()
model = GamblingClassifier("app/model")

@app.get("/")
def hello():
    return {"message": "Hello, world!"}

@app.post("/predict")
def predict(data: InputText):
    cleaned = clean_text(data.text)
    return model.predict(cleaned)

@app.post("/predict-file")
def predict_file(file: UploadFile = File(...)):
    base_dir = Path("storage")
    temp_dir = base_dir / "temp"
    judol_dir = base_dir / "judol"
    non_judol_dir = base_dir / "non_judol"

    temp_dir.mkdir(parents=True, exist_ok=True)
    judol_dir.mkdir(parents=True, exist_ok=True)
    non_judol_dir.mkdir(parents=True, exist_ok=True)

    temp_input = temp_dir / f"temp_{uuid.uuid4().hex}.json"
    temp_judol = judol_dir / f"judol_{uuid.uuid4().hex}.json"
    temp_non_judol = non_judol_dir / f"non_judol_{uuid.uuid4().hex}.json"

    with open(temp_input, "wb") as f:
        f.write(file.file.read())

    process_file(temp_input, temp_judol, temp_non_judol)

    os.remove(temp_input)

    return {
        "judol_result": temp_judol,
        "non_judol_result": temp_non_judol
    }

@app.get("/download/{filename}")
def download_file(filename: str):
    return FileResponse(path=filename, filename=filename, media_type='application/json')
