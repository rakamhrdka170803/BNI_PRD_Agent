import gradio as gr
import os
import llmops
from llmops import SpanType
from dotenv import load_dotenv
from markdown_pdf import MarkdownPdf, Section

# --- IMPORT MODULES CLEAN CODE ---
from retriever import FeedbackRetriever
from agent_logic import PRDAgent

load_dotenv(override=True)
llmops.autolog("langchain")

# --- KONFIGURASI GLOBAL ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3-32b")
DB_PARAMS = {
    "dbname": os.getenv("DB_NAME", "prd_db"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", ""),
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", "5432"),
}
OLLAMA_CONFIG = {
    "model": os.getenv("OLLAMA_EMBED_MODEL", "qwen3-embedding:4b"),
    "base_url": os.getenv("OLLAMA_HOST", "http://localhost:11434"),
}

# --- INIT SERVICES ---
retriever = FeedbackRetriever(DB_PARAMS, OLLAMA_CONFIG)
agent = PRDAgent(api_key=GROQ_API_KEY, model_name=GROQ_MODEL)

# --- WRAPPER FUNCTIONS UNTUK GRADIO EVENTS ---
@llmops.trace_agent("user_generate", span_type=SpanType.AGENT)
def on_generate(obj, users, pain, features, ac, out_scope):
    inputs = {
        "1. Objective": obj, "2. Target": users, "3. Pain": pain,
        "4. Features": features, "5. AC": ac, "6. Out Scope": out_scope
    }
    context_query = f"Objektif: {obj}. Fitur: {features}. Pain Points: {pain}"

    feedback_context = retriever.get_relevant_feedback(context_query, limit=10)
    print(f"\n[DEBUG RAG] Feedback ditarik:\n{feedback_context}\n")

    markdown_output, new_history = agent.generate_initial(inputs, feedback_context)

    return markdown_output, new_history, gr.update(visible=False), gr.update(visible=True)

@llmops.trace_agent("user_revise", span_type=SpanType.AGENT)
def on_revise(instruction, history_state):
    markdown_output, updated_history = agent.revise(instruction, history_state)
    return markdown_output, updated_history, gr.update(value="")

def on_approve(markdown_text, history_state):
    pdf = MarkdownPdf(toc_level=2)
    pdf.add_section(Section(markdown_text))
    pdf_filename = "Approved_PRD.pdf"
    pdf.save(pdf_filename)
    return pdf_filename, history_state

def on_reset():
    return gr.update(visible=True), gr.update(visible=False), [], "", None

# --- UI GRADIO ---
with gr.Blocks(theme=gr.themes.Monochrome()) as demo:
    gr.Markdown("# 🚀 AI Product Manager Agent")
    memory_state = gr.State([])
    
    with gr.Row():
        with gr.Column(scale=1):
            with gr.Column(visible=True) as col_initial:
                gr.Markdown("### Setup PRD Awal")
                i_obj = gr.Textbox(label="1. Objective & Goals", lines=2)
                i_usr = gr.Textbox(label="2. Target Users", lines=1)
                i_pain = gr.Textbox(label="3. User Pain Points", lines=2)
                i_feat = gr.Textbox(label="4. Feature Requirements", lines=2)
                i_ac = gr.Textbox(label="5. Acceptance Criteria", lines=2)
                i_out = gr.Textbox(label="6. Out of Scope", lines=2)
                btn_generate = gr.Button("🧠 Generate PRD", variant="primary")
            
            with gr.Column(visible=False) as col_revise:
                gr.Markdown("### Chat & Revisi")
                i_revise = gr.Textbox(label="Instruksi Revisi", lines=3)
                btn_revise = gr.Button("🔄 Submit Revisi", variant="primary")
                btn_reset = gr.Button("🗑️ Mulai dari Awal (Reset)", variant="stop")
            
        with gr.Column(scale=1):
            gr.Markdown("### Live PRD Output")
            out_md = gr.Markdown(elem_id="output_md", height=500)
            with gr.Row():
                btn_approve = gr.Button("✅ Approve & Generate PDF")
                out_file = gr.File(label="Download PRD")

    # Mapping Events
    btn_generate.click(on_generate, [i_obj, i_usr, i_pain, i_feat, i_ac, i_out], [out_md, memory_state, col_initial, col_revise])
    btn_revise.click(on_revise, [i_revise, memory_state], [out_md, memory_state, i_revise])
    btn_reset.click(on_reset, None, [col_initial, col_revise, memory_state, out_md, out_file])
    btn_approve.click(on_approve, [out_md, memory_state], [out_file, memory_state])

if __name__ == "__main__":
    demo.launch()