import sys
sys.path.append('../pipeline')

import gradio as gr
import json
from pipeline import DocumentIntelligencePipeline

pipeline = DocumentIntelligencePipeline()

def analyze_document(file, questions_text):
    questions = [q.strip() for q in questions_text.split('\n') if q.strip()]

    result = pipeline.run(
        file_path=file.name,
        questions=questions
    )

    summary    = result['summary']
    sentiment  = f"{result['sentiment']['label']} (confidence: {result['sentiment']['score']})"
    entities   = json.dumps(result['entities'], indent=2)
    qa_results = "\n\n".join([
        f"Q: {r['question']}\nA: {r['answer']} (confidence: {r['confidence']})"
        for r in result['qa']
    ])

    return summary, sentiment, entities, qa_results


with gr.Blocks(title="Document Intelligence Pipeline") as demo:
    gr.Markdown("# 📄 Document Intelligence Pipeline\nPowered by RocketRide + HuggingFace")

    with gr.Row():
        file_input = gr.File(label="Upload PDF or TXT", file_types=['.pdf', '.txt'])
        questions  = gr.Textbox(
            label="Questions (one per line)",
            placeholder="What is the main topic?\nWho are the key people?",
            lines=5
        )

    btn = gr.Button("🚀 Analyze", variant="primary")

    with gr.Row():
        summary_out   = gr.Textbox(label="📝 Summary", lines=5)
        sentiment_out = gr.Textbox(label="😊 Sentiment")

    with gr.Row():
        entities_out = gr.Textbox(label="🏷️ Entities (NER)", lines=8)
        qa_out       = gr.Textbox(label="💬 Q&A Answers", lines=8)

    btn.click(
        fn=analyze_document,
        inputs=[file_input, questions],
        outputs=[summary_out, sentiment_out, entities_out, qa_out]
    )

demo.launch(share=True)