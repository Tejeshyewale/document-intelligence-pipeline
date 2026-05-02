from transformers import AutoModelForQuestionAnswering, AutoTokenizer
import torch

class QAChatbot:
    def __init__(self, config: dict):
        model_name = config.get(chr(39)+"model"+chr(39), "deepset/roberta-base-squad2")
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForQuestionAnswering.from_pretrained(model_name)

    def answer(self, question: str, context: str) -> dict:
        truncated_context = " ".join(context.split()[:500])
        inputs = self._tokenizer(question, truncated_context, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self._model(**inputs)
        start = torch.argmax(outputs.start_logits)
        end = torch.argmax(outputs.end_logits) + 1
        answer = self._tokenizer.convert_tokens_to_string(self._tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][start:end]))
        confidence = round((torch.max(outputs.start_logits) + torch.max(outputs.end_logits)).item() / 20, 4)
        return {"answer": answer if answer.strip() else "No answer found.", "confidence": max(0.0, min(1.0, confidence))}
