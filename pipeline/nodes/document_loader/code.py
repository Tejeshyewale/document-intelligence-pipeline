import fitz  # PyMuPDF
import pdfplumber

class DocumentLoader:
    def __init__(self, config: dict):
        self.config = config

    def load(self, file_path: str) -> str:
        """Load PDF or text file and return raw text."""
        if file_path.endswith('.pdf'):
            return self._load_pdf(file_path)
        elif file_path.endswith('.txt'):
            return self._load_txt(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_path}")

    def _load_pdf(self, path: str) -> str:
        text = ""
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
        return text.strip()

    def _load_txt(self, path: str) -> str:
        with open(path, 'r', encoding='utf-8') as f:
            return f.read().strip()