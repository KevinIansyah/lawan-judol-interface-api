import json
from typing import List, Dict
from app.model.ner import KeywordExtractor

keyword_model = KeywordExtractor("app/model/ner")

def extract_keywords_from_judol_data(judol_data: Dict) -> List[Dict]:
    all_keywords = {}
    
    for chunk in judol_data["chunks"]:
        comments = chunk["comments"]
        texts = [comment["text"] for comment in comments]
        
        for text in texts:
            entities = keyword_model.extract_entities_from_text(text)
            for entity in entities:
                keyword = entity['text'].strip()
                if keyword and len(keyword) > 1:
                    if keyword not in all_keywords:
                        all_keywords[keyword] = {
                            'count': 0,
                            'entity_type': entity['label']
                        }
                    all_keywords[keyword]['count'] += 1
    
    keywords_list = []
    sorted_keywords = sorted(all_keywords.items(), key=lambda x: x[1]['count'], reverse=True)
    
    for i, (keyword, data) in enumerate(sorted_keywords, 1):
        keywords_list.append({
            "id": i,
            "keyword": keyword,
            "label": 1,
            "entity_type": data['entity_type'],
            "frequency": data['count']
        })
    
    return keywords_list

def save_keywords_to_file(keywords: List[Dict], output_path: str):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(keywords, f, ensure_ascii=False, indent=2)