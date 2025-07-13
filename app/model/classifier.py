from transformers import BertTokenizer
from optimum.onnxruntime import ORTModelForSequenceClassification
import torch
import numpy as np
from typing import List

class GamblingClassifier:
    def __init__(self, model_path, max_length=128):
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.model = ORTModelForSequenceClassification.from_pretrained(
            model_path,
            provider="CPUExecutionProvider"
        )
        self.max_length = max_length

    def predict(self, text):
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
            predicted_class = torch.argmax(logits, dim=-1).item()
            confidence = probabilities[0][predicted_class].item()

        return {
            "predicted_label": predicted_class,
            "label_name": "Judol" if predicted_class == 1 else "Non-Judol",
            "confidence": round(confidence, 3),
            "probabilities": {
                "non_judol": round(probabilities[0][0].item(), 3),
                "judol": round(probabilities[0][1].item(), 3)
            }
        }
    
    def predict_batch(self, texts: List[str], batch_size: int = 16) -> List[int]:
        predictions = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                return_tensors="pt", 
                max_length=self.max_length
            )

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                batch_preds = torch.argmax(logits, dim=1).tolist()
                predictions.extend(batch_preds)

        return predictions