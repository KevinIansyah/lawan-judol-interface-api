import json
from pathlib import Path
from app.utils.text_cleaner import clean_text
from app.model.classifier import GamblingClassifier

model = GamblingClassifier("app/model")

def process_file(input_path: str, output_judol_path: str, output_non_judol_path: str):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    judol_data = {
        "total_comments": 0,
        "total_chunks": 0,
        "chunks": []
    }

    non_judol_data = {
        "total_comments": 0,
        "total_chunks": 0,
        "chunks": []
    }

    for chunk in data["chunks"]:
        comments = chunk["comments"]
        texts = [clean_text(comment["text"]) for comment in comments]
        preds = model.predict_batch(texts)

        judol_chunk = {"chunk_id": chunk["chunk_id"], "comments": []}
        non_judol_chunk = {"chunk_id": chunk["chunk_id"], "comments": []}

        for comment, pred in zip(comments, preds):
            comment["label"] = pred
            if pred == 1:
                judol_chunk["comments"].append(comment)
            else:
                non_judol_chunk["comments"].append(comment)

        if judol_chunk["comments"]:
            judol_data["chunks"].append(judol_chunk)
            judol_data["total_comments"] += len(judol_chunk["comments"])

        if non_judol_chunk["comments"]:
            non_judol_data["chunks"].append(non_judol_chunk)
            non_judol_data["total_comments"] += len(non_judol_chunk["comments"])

    judol_data["total_chunks"] = len(judol_data["chunks"])
    non_judol_data["total_chunks"] = len(non_judol_data["chunks"])

    with open(output_judol_path, "w", encoding="utf-8") as f:
        json.dump(judol_data, f, ensure_ascii=False, indent=2)

    with open(output_non_judol_path, "w", encoding="utf-8") as f:
        json.dump(non_judol_data, f, ensure_ascii=False, indent=2)

    print("✅ Selesai. Hasil disimpan di:")
    print(f"   - Judol: {output_judol_path}")
    print(f"   - Non-Judol: {output_non_judol_path}")
