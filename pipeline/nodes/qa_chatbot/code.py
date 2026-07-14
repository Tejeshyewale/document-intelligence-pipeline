from transformers import AutoModelForQuestionAnswering, AutoTokenizer
import torch


class QAChatbot:
    # Below this confidence, we tell the user honestly instead of guessing.
    CONFIDENCE_THRESHOLD = 0.15

    def __init__(self, config: dict):
        model_name = config.get('model', 'deepset/roberta-base-squad2')
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForQuestionAnswering.from_pretrained(model_name)

    def answer(self, question: str, context: str) -> dict:
        if not question or not question.strip():
            return {"answer": "No question provided.", "confidence": 0.0}
        if not context or not context.strip():
            return {"answer": "No document context available.", "confidence": 0.0}

        truncated_context = " ".join(context.split()[:500])
        inputs = self._tokenizer(
            question, truncated_context,
            return_tensors="pt", truncation=True, max_length=512
        )

        with torch.no_grad():
            outputs = self._model(**inputs)

        start = torch.argmax(outputs.start_logits)
        end = torch.argmax(outputs.end_logits) + 1

        # Guard against a malformed span (end before start)
        if end <= start:
            return {"answer": "Not confident enough to answer.", "confidence": 0.0}

        answer = self._tokenizer.convert_tokens_to_string(
            self._tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][start:end])
        )

        raw_score = (torch.max(outputs.start_logits) + torch.max(outputs.end_logits)).item() / 20
        confidence = round(max(0.0, min(1.0, raw_score)), 4)

        if not answer.strip() or confidence < self.CONFIDENCE_THRESHOLD:
            return {"answer": "Not confident enough to answer accurately.", "confidence": confidence}

        return {"answer": answer, "confidence": confidence}
