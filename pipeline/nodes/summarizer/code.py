from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class Summarizer:
    def __init__(self, config: dict):
        model_name = config.get('model', 'sshleifer/distilbart-cnn-12-6')
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.max_length = config.get('max_length', 150)
        self.min_length = config.get('min_length', 50)

    def summarize(self, text: str) -> str:
        if len(text.split()) > 800:
            text = ' '.join(text.split()[:800])

        inputs = self._tokenizer(
            text,
            return_tensors='pt',
            max_length=1024,
            truncation=True
        )

        summary_ids = self._model.generate(
            inputs['input_ids'],
            max_length=self.max_length,
            min_length=self.min_length,
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True
        )

        return self._tokenizer.decode(
            summary_ids[0],
            skip_special_tokens=True
        )