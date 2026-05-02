from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class SentimentAnalyzer:
    def __init__(self, config: dict):
        model_name = config.get(
            'model',
            'cardiffnlp/twitter-roberta-base-sentiment-latest'
        )

        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(model_name)

    def analyze(self, text: str) -> dict:
        print("🔥 FIX RUNNING")

        safe_text = text[:800]

        inputs = self._tokenizer(
            safe_text,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )

        with torch.no_grad():
            outputs = self._model(**inputs)

        probs = torch.nn.functional.softmax(outputs.logits, dim=1)[0]
        pred = torch.argmax(probs).item()

        label = self._model.config.id2label[pred]

        return {
            "label": label,
            "score": round(probs[pred].item(), 4)
        }