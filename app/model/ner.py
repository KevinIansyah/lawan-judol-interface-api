from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForTokenClassification
import torch
from typing import List, Dict
from app.utils.text_cleaner import clean_text_keywoard

class KeywordExtractor:
    def __init__(self, model_path: str, max_length: int = 128):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = ORTModelForTokenClassification.from_pretrained(
            model_path,
            provider="CPUExecutionProvider"
        )
        self.max_length = max_length
        self.label_to_id = {
            'O': 0,
            'B-SITE': 1,
            'I-SITE': 2,
            'B-GENERAL': 3,
            'I-GENERAL': 4
        }
        self.id_to_label = {v: k for k, v in self.label_to_id.items()}
    
    def extract_entities_from_text(self, text: str) -> List[Dict[str, str]]:
        cleaned_text = clean_text_keywoard(text)
        if not cleaned_text or not cleaned_text.strip():
            return []
        
        inputs = self.tokenizer(
            cleaned_text,
            return_tensors='pt',
            truncation=True,
            max_length=self.max_length,
            padding=True
        )
        
        offset_inputs = self.tokenizer(
            cleaned_text,
            return_offsets_mapping=True,
            truncation=True,
            max_length=self.max_length,
            padding=True
        )
        
        with torch.no_grad():
            outputs = self.model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
            predictions = torch.argmax(outputs.logits, dim=-1)
        
        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        predicted_labels = predictions[0].cpu().numpy()
        offset_mapping = offset_inputs['offset_mapping']
        
        entities = self._extract_entities_improved(
            cleaned_text, tokens, predicted_labels, offset_mapping
        )
        
        return entities
    
    def _extract_entities_improved(self, text: str, tokens: List[str], labels: List[int], offset_mapping: List) -> List[Dict[str, str]]:
        entities = []
        current_entity = {
            'label': None,
            'start_pos': None,
            'end_pos': None,
            'tokens': []
        }
        
        for i, (token, label_id, offset) in enumerate(zip(tokens, labels, offset_mapping)):
            if token in ['[CLS]', '[SEP]', '[PAD]'] or offset[0] is None:
                continue
            
            label = self.id_to_label[label_id]
            
            if label.startswith('B-'):
                if current_entity['start_pos'] is not None:
                    entity = self._finalize_entity(current_entity, text)
                    if entity:
                        entities.append(entity)
                
                current_entity = {
                    'label': label[2:],
                    'start_pos': offset[0],
                    'end_pos': offset[1],
                    'tokens': [token]
                }
                
            elif label.startswith('I-') and current_entity['start_pos'] is not None:
                if label[2:] == current_entity['label'] or token.startswith('##'):
                    current_entity['end_pos'] = offset[1]
                    current_entity['tokens'].append(token)
                else:
                    entity = self._finalize_entity(current_entity, text)
                    if entity:
                        entities.append(entity)
                    current_entity = {
                        'label': label[2:],
                        'start_pos': offset[0],
                        'end_pos': offset[1],
                        'tokens': [token]
                    }
                    
            elif label == 'O':
                if (current_entity['start_pos'] is not None and
                    token.startswith('##') and
                    len(current_entity['tokens']) > 0):
                    current_entity['end_pos'] = offset[1]
                    current_entity['tokens'].append(token)
                else:
                    if current_entity['start_pos'] is not None:
                        entity = self._finalize_entity(current_entity, text)
                        if entity:
                            entities.append(entity)
                        current_entity = {'label': None, 'start_pos': None, 'end_pos': None, 'tokens': []}
        
        if current_entity['start_pos'] is not None:
            entity = self._finalize_entity(current_entity, text)
            if entity:
                entities.append(entity)
        
        return entities
    
    def _finalize_entity(self, entity_data: Dict, original_text: str) -> Dict[str, str]:
        """Convert token-based entity to text format"""
        if entity_data['start_pos'] is None or entity_data['end_pos'] is None:
            return None
        
        start_pos = entity_data['start_pos']
        end_pos = entity_data['end_pos']
        
        entity_text = original_text[start_pos:end_pos].strip()
        
        if not entity_text:
            return None
        
        return {
            'text': entity_text,
            'label': entity_data['label']
        }
    
    def extract_keywords_batch(self, texts: List[str]) -> List[List[str]]:
        """Extract keywords from multiple texts"""
        results = []
        for text in texts:
            entities = self.extract_entities_from_text(text)
            keywords = [entity['text'] for entity in entities if entity['text'].strip()]
            results.append(keywords)
        return results
    
    def extract_all_keywords(self, texts: List[str]) -> List[str]:
        """Extract all unique keywords from list of texts"""
        all_keywords = set()
        for text in texts:
            entities = self.extract_entities_from_text(text)
            for entity in entities:
                keyword = entity['text'].strip()
                if keyword and len(keyword) > 1:
                    all_keywords.add(keyword)
        return list(all_keywords)