from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
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

app = FastAPI()

classifier_model = GamblingClassifier("app/model/classifier")
ner_model = KeywordExtractor("app/model/ner")

@app.get("/")
def hello():
    return {"message": "Hello, world!"}

@app.post("/extract-keywords")
def extract_keywords(data: InputText):
    """Extract keywords from single text"""
    entities = ner_model.extract_entities_from_text(data.text)
    keywords = [entity['text'] for entity in entities]
    return {
        "text": data.text,
        "entities": entities,
        "keywords": keywords
    }

@app.post("/predict")
def predict(data: InputText):
    cleaned = clean_text_classifier(data.text)
    return classifier_model.predict(cleaned)

@app.post("/predict-file")
def predict_file(file: UploadFile = File(...)):
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
    
    print("judol_result:", str(temp_judol))
    print("non_judol_result:", str(temp_non_judol))
    print("keyword_result:", str(temp_keywords))
    
    return {
        "judol_result": str(temp_judol),
        "non_judol_result": str(temp_non_judol),
        "keyword_result": str(temp_keywords),
        # "summary": {
        #     "total_judol_comments": judol_data["total_comments"],
        #     "total_keywords_found": len(keywords),
        #     "top_keywords": keywords[:10]
        # }
    }

@app.get("/download/{filename}")
def download_file(filename: str):
    return FileResponse(path=filename, filename=filename, media_type='application/json')