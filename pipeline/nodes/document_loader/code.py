import os
import pdfplumber


class DocumentLoader:
    def __init__(self, config: dict):
        self.config = config

    def load(self, file_path: str) -> str:
        """Load PDF or text file and return raw text."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        if file_path.endswith('.pdf'):
            text = self._load_pdf(file_path)
        elif file_path.endswith('.txt'):
            text = self._load_txt(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_path}")

        if not text.strip():
            raise ValueError(f"No extractable text found in: {file_path}")

        return text

    def _load_pdf(self, path: str) -> str:
        text = ""
        try:
            with pdfplumber.open(path) as pdf:
                for page in pdf.pages:
                    extracted = page.extract_text()
                    if extracted:
                        text += extracted + "\n"
        except Exception as e:
            raise ValueError(f"Could not read PDF '{path}': {e}")
        return text.strip()

    def _load_txt(self, path: str) -> str:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except UnicodeDecodeError:
            raise ValueError(f"Could not decode '{path}' as UTF-8 text.")
