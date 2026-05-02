from typing import List

class TextChunker:
    def __init__(self, config: dict):
        self.chunk_size = config.get('chunk_size', 500)
        self.overlap = config.get('overlap', 50)

    def chunk(self, text: str) -> List[str]:
        """Split text into overlapping chunks."""
        words = text.split()
        chunks = []
        start = 0

        while start < len(words):
            end = start + self.chunk_size
            chunk = ' '.join(words[start:end])
            chunks.append(chunk)
            start += self.chunk_size - self.overlap

        return chunks