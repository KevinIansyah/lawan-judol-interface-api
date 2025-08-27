import json
from app.utils.text_cleaner import clean_text_classifier
from app.model.classifier import GamblingClassifier

classifier_model = GamblingClassifier("app/model/classifier")

def split_comments(comments, max_per_chunk=100):
    return [comments[i:i + max_per_chunk] for i in range(0, len(comments), max_per_chunk)]

def process_file(input_path: str, output_judol_path: str, output_non_judol_path: str):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    all_judol = []
    all_non_judol = []

    for chunk in data["chunks"]:
        comments = chunk["comments"]
        texts = [clean_text_classifier(comment["text"]) for comment in comments]
        preds = classifier_model.predict_batch(texts)

        for comment, pred in zip(comments, preds):
            comment["label"] = pred
            if pred == 1:
                all_judol.append(comment)
            else:
                all_non_judol.append(comment)

    judol_chunks = split_comments(all_judol, max_per_chunk=100)
    non_judol_chunks = split_comments(all_non_judol, max_per_chunk=100)

    judol_data = {
        "total_comments": len(all_judol),
        "total_chunks": len(judol_chunks),
        "chunks": [{"chunk_id": i + 1, "comments": chunk} for i, chunk in enumerate(judol_chunks)]
    }

    non_judol_data = {
        "total_comments": len(all_non_judol),
        "total_chunks": len(non_judol_chunks),
        "chunks": [{"chunk_id": i + 1, "comments": chunk} for i, chunk in enumerate(non_judol_chunks)]
    }

    with open(output_judol_path, "w", encoding="utf-8") as f:
        json.dump(judol_data, f, ensure_ascii=False, indent=2)

    with open(output_non_judol_path, "w", encoding="utf-8") as f:
        json.dump(non_judol_data, f, ensure_ascii=False, indent=2)