import json
from nodes.document_loader.code import DocumentLoader
from nodes.text_chunker.code import TextChunker
from nodes.summarizer.code import Summarizer
from nodes.sentiment_analyzer.code import SentimentAnalyzer
from nodes.entity_extractor.code import EntityExtractor
from nodes.qa_chatbot.code import QAChatbot


class DocumentIntelligencePipeline:
    def __init__(self, config: dict = {}):
        self.loader     = DocumentLoader(config.get('loader', {}))
        self.chunker    = TextChunker(config.get('chunker', {'chunk_size': 500, 'overlap': 50}))
        self.summarizer = Summarizer(config.get('summarizer', {}))
        self.sentiment  = SentimentAnalyzer(config.get('sentiment', {}))
        self.ner        = EntityExtractor(config.get('ner', {}))
        self.qa         = QAChatbot(config.get('qa', {}))

    def run(self, file_path: str, questions: list = []) -> dict:
        print(f"[1/6] Loading document: {file_path}")
        raw_text = self.loader.load(file_path)

        print("[2/6] Chunking text...")
        chunks = self.chunker.chunk(raw_text)

        print("[3/6] Summarizing...")
        summary = self.summarizer.summarize(raw_text)

        print("[4/6] Analyzing sentiment...")
        sentiment = self.sentiment.analyze(raw_text)

        print("[5/6] Extracting entities...")
        entities = self.ner.extract(raw_text)

        print("[6/6] Answering questions...")
        answers = []
        for q in questions:
            ans = self.qa.answer(q, raw_text)
            answers.append({
                'question': q,
                'answer': ans['answer'],
                'confidence': ans['confidence']
            })

        return {
            'file': file_path,
            'summary': summary,
            'sentiment': sentiment,
            'entities': entities,
            'qa': answers,
            'chunks_count': len(chunks)
        }


if __name__ == "__main__":
    pipeline = DocumentIntelligencePipeline()

    result = pipeline.run(
        file_path="../demo/sample.txt",
        questions=[
            "What is the main topic?",
            "Who are the key people mentioned?",
            "What are the conclusions?"
        ]
    )

    print("\n===== RESULTS =====")
    print(json.dumps(result, indent=2))