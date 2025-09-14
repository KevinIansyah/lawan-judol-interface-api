from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, Security
from fastapi.responses import FileResponse
from fastapi.security.api_key import APIKeyHeader
from starlette.status import HTTP_403_FORBIDDEN
from app.model.classifier import GamblingClassifier
from app.model.ner import KeywordExtractor
from app.schemas.input import InputText
from app.utils.file_predicor import process_file
from app.utils.text_cleaner import clean_text_classifier
from app.utils.keyword_processor import extract_keywords_from_judol_data, save_keywords_to_file
from pathlib import Path
import uuid
import os
import json
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("API_KEY")
API_KEY_NAME = os.getenv("API_KEY_NAME", "X-API-Key")
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

async def get_api_key(api_key_header: str = Security(api_key_header)):
    if api_key_header == API_KEY:
        return api_key_header
    raise HTTPException(
        status_code=HTTP_403_FORBIDDEN, detail="Invalid or missing API Key"
    )

app = FastAPI()

classifier_model = GamblingClassifier("app/model/classifier")
ner_model = KeywordExtractor("app/model/ner")

@app.get("/")
def hello():
    return {"message": "API SIAP DIGUNAKAN"}


@app.post("/extract-keywords")
def extract_keywords(data: InputText, api_key: str = Depends(get_api_key)):
    entities = ner_model.extract_entities_from_text(data.text)
    keywords = [entity['text'] for entity in entities]
    return {
        "text": data.text,
        "entities": entities,
        "keywords": keywords
    }


@app.post("/predict")
def predict(data: InputText, api_key: str = Depends(get_api_key)):
    cleaned = clean_text_classifier(data.text)
    return classifier_model.predict(cleaned)


@app.post("/predict-file")
def predict_file(file: UploadFile = File(...), api_key: str = Depends(get_api_key)):
    base_dir = Path("storage")
    temp_dir = base_dir / "temp"
    judol_dir = base_dir / "judol"
    non_judol_dir = base_dir / "non_judol"
    keywords_dir = base_dir / "keywords"
    
    temp_dir.mkdir(parents=True, exist_ok=True)
    judol_dir.mkdir(parents=True, exist_ok=True)
    non_judol_dir.mkdir(parents=True, exist_ok=True)
    keywords_dir.mkdir(parents=True, exist_ok=True)
    
    temp_input = temp_dir / f"temp_{uuid.uuid4().hex}.json"
    temp_judol = judol_dir / f"judol_{uuid.uuid4().hex}.json"
    temp_non_judol = non_judol_dir / f"non_judol_{uuid.uuid4().hex}.json"
    temp_keywords = keywords_dir / f"keywords_{uuid.uuid4().hex}.json"
    
    with open(temp_input, "wb") as f:
        f.write(file.file.read())
    
    process_file(temp_input, temp_judol, temp_non_judol)
    
    with open(temp_judol, "r", encoding="utf-8") as f:
        judol_data = json.load(f)
    
    keywords = extract_keywords_from_judol_data(judol_data)
    save_keywords_to_file(keywords, temp_keywords)
    
    os.remove(temp_input)
    
    return {
        "judol_result": str(temp_judol),
        "non_judol_result": str(temp_non_judol),
        "keyword_result": str(temp_keywords),
    }


@app.get("/download/{filename}")
def download_file(filename: str, api_key: str = Depends(get_api_key)):
    return FileResponse(path=filename, filename=filename, media_type='application/json')