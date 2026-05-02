import spacy
from typing import List, Dict

class EntityExtractor:
    def __init__(self, config: dict):
        model = config.get('model', 'en_core_web_sm')
        self._nlp = spacy.load(model)

    def extract(self, text: str) -> Dict[str, List[str]]:
        """Extract named entities grouped by type."""
        doc = self._nlp(text[:10000])

        entities = {}
        for ent in doc.ents:
            label = ent.label_
            if label not in entities:
                entities[label] = []
            if ent.text not in entities[label]:
                entities[label].append(ent.text)

        return entities