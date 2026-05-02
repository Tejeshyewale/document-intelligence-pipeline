from transformers import pipeline

class SentimentAnalyzer:
    def __init__(self, config: dict):
        model = config.get('model', 'cardiffnlp/twitter-roberta-base-sentiment-latest')
        self._pipe = pipeline('sentiment-analysis', model=model)

    def analyze(self, text: str) -> dict:
        """Return sentiment label and confidence score."""
        # Truncate to model max tokens
        truncated = ' '.join(text.split()[:400])
        result = self._pipe(truncated)[0]

        return {
            'label': result['label'],    # POSITIVE / NEGATIVE / NEUTRAL
            'score': round(result['score'], 4)
        }