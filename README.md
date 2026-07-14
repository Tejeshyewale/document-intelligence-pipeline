# 📄 Document Intelligence Pipeline

![Python](https://img.shields.io/badge/python-3.13-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-active-brightgreen)

An end-to-end NLP pipeline for analyzing PDF/TXT documents — built while exploring the RocketRide AI Internship track. Combines summarization, sentiment analysis, entity extraction, and question answering into a single modular flow.

---

## 🔥 Features

- 📂 **Document Loader** — ingest PDF/TXT files
- ✂️ **Text Chunking** — splits long documents for downstream processing
- 🧠 **Summarization** — DistilBART
- 😊 **Sentiment Analysis** — RoBERTa
- 🏷️ **Named Entity Recognition** — spaCy
- ❓ **Question Answering** — RoBERTa (SQuAD2, extractive)

---

## 🧱 Architecture

```
[Document]
     ↓
[Text Chunking]
     ↓
 ┌───────────────┬───────────────┬───────────────┬───────────────┐
 │ Summarizer    │ Sentiment     │ NER           │ QA            │
 │ (DistilBART)  │ (RoBERTa)     │ (spaCy)       │ (RoBERTa-SQuAD2)│
 └───────────────┴───────────────┴───────────────┴───────────────┘
     ↓
[Final Output]
```

Each stage runs as an independent node under `pipeline/nodes/`, so components can be swapped, tested, or extended individually.

---

## 📸 Demo Screenshot

![Demo](./screenshot.png)

## 🚀 Live Demo

👉 [Temporary demo](https://f5a22b668fbafbf895.gradio.live) *(Gradio link — may expire; run locally for a stable version)*

---

## 🛠️ Tech Stack

- Python 3.13
- HuggingFace Transformers
- PyTorch
- spaCy
- Gradio

---

## 📂 Project Structure

```
pipeline/
  nodes/
    document_loader/
    text_chunker/
    summarizer/
    sentiment_analyzer/
    entity_extractor/
    qa_chatbot/
  pipeline.py

demo/
  app.py

pipelines/
  document_intelligence.json
```

---

## ▶️ How to Run

```bash
git clone https://github.com/Tejeshyewale/document-intelligence-pipeline
cd document-intelligence-pipeline
python -m venv venv
venv\Scripts\activate       # on Windows
# source venv/bin/activate  # on macOS/Linux
pip install -r requirements.txt
python demo/app.py
```

---

## ⚠️ Known Limitations

- Uses off-the-shelf pretrained models — no fine-tuning on domain-specific data yet
- QA is extractive only; it returns low-confidence answers for vague or multi-hop questions
- No automated test suite yet
- Single-document only — no cross-document reasoning

---

## 📌 Roadmap

- [ ] Add unit tests for each pipeline node
- [ ] Retrieval-Augmented Generation (RAG) for better QA
- [ ] Multi-document support
- [ ] Fine-tune models on a domain-specific dataset
- [ ] Add CI (GitHub Actions) for automated checks

---

## 🤝 Contributing

Suggestions and PRs are welcome — this is an active learning project and I'm open to feedback on architecture, code quality, or model choices.

---

## 👨‍💻 Author

**Tejesh Yewale**

---

⭐ If you find this useful, consider giving it a star!
