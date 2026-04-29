import gradio as gr
import psycopg2
from pgvector.psycopg2 import register_vector
import numpy as np
from langchain_ollama import OllamaEmbeddings
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from markdown_pdf import MarkdownPdf, Section
import re
import os
from dotenv import load_dotenv

# Load variables from .env file
load_dotenv()

# --- KONFIGURASI ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DB_PARAMS = {"dbname": "prd_db", "user": "postgres", "password": "postgres", "host": "localhost"}

# --- INIT MODELS ---
llm = ChatGroq(temperature=0.3, groq_api_key=GROQ_API_KEY, model_name="qwen/qwen3-32b") # Sesuaikan model Groq
embedder = OllamaEmbeddings(model="qwen3-embedding:4b", base_url="http://localhost:11434")

# --- DATABASE RETRIEVAL (RAG) ---
def get_relevant_feedback(query_text, limit=10):
    query_vec = embedder.embed_query(query_text)
    conn = psycopg2.connect(**DB_PARAMS)
    register_vector(conn)
    cur = conn.cursor()
    
    # Cosine distance (<=>)
    cur.execute("""
        SELECT komentar FROM user_feedback 
        ORDER BY embedding <=> %s LIMIT %s
    """, (np.array(query_vec), limit))
    
    results = [row[0] for row in cur.fetchall()]
    conn.close()
    return "\n- " + "\n- ".join(results) if results else "Tidak ada feedback spesifik."

# --- GENERATE PRD LOGIC ---
def generate_prd(obj, users, pain, features, ac, out_scope, history_state):
    # 1. ANTI-CRASH: Cegah error saat Gradio mengirim state 'null' pertama kali
    if history_state is None:
        history_state = []
        
    # Gabungkan input PM untuk dicari relevansinya di Database
    pm_context = f"Objektif: {obj}. Fitur: {features}. Pain Points: {pain}"
    
    # RETRIEVE dari PostgreSQL
    relevant_feedbacks = get_relevant_feedback(pm_context)
    print(f"\n[DEBUG RAG] Feedback yang ditarik untuk prompt:\n{relevant_feedbacks}\n")
    
    # 2. PENANGKAL CoT: Prompt dibuat sangat galak dan instruktif
    system_prompt = f"""Kamu adalah Senior Product Manager AI. Buatlah dokumen PRD dalam format Markdown.
    
    ATURAN SANGAT KETAT:
    - JANGAN PERNAH menampilkan proses berpikir, analisa, atau kalimat pembuka seperti "Okay, let me start...", "Baiklah...", atau "Here is the PRD".
    - OUTPUT HARUS langsung dimulai dengan tag markdown "# Product Requirements Document (PRD)".
    - Gunakan bahasa Indonesia yang profesional dan lugas.
    - KEKANGAN SCOPE: Di bagian "Feature Requirements", HANYA tuliskan fitur yang secara eksplisit diminta oleh PM. JANGAN MENCIPTAKAN FITUR BARU berdasarkan feedback database. 
    - KEKANGAN FEEDBACK: Gunakan feedback dari database HANYA untuk memperkaya dan membuktikan bagian "User Pain Points" dan "Acceptance Criteria". Kelompokkan Pain Points menjadi kategori yang rapi (misal: UI/UX, Performa Sistem).
    - KEKANGAN OUT OF SCOPE: Jangan mengarang batasan yang tidak relevan. Cukup rapikan input dari PM.
    
    Feedback dari database (Gunakan sebagai justifikasi masalah):
    {relevant_feedbacks}
    
    Format Wajib (Selalu 6 Poin Ini):
    1. Objective & Goals
    2. Target Users
    3. User Pain Points (Kelompokkan dengan rapi: Input PM + Realita Database)
    4. Feature Requirements (HANYA kembangkan dari Input PM)
    5. Acceptance Criteria (Buat metrik kuantitatif yang jelas)
    6. Out of Scope
    """
    
    user_prompt = f"Input PM:\n1. Objective: {obj}\n2. Target: {users}\n3. Pain: {pain}\n4. Features: {features}\n5. AC: {ac}\n6. Out Scope: {out_scope}"
    
    # Masukkan ke History & Call LLM
    messages = [SystemMessage(content=system_prompt)]
    for msg in history_state:
        messages.append(msg) # Masukkan memory revisi sebelumnya
        
    messages.append(HumanMessage(content=user_prompt))
    
    response = llm.invoke(messages)
    raw_output = response.content
    
    # 3. FILTER CLEANER: Buang tag <think> jika LLM masih bandel berpikir keras di background
    clean_output = re.sub(r'<think>.*?</think>', '', raw_output, flags=re.DOTALL).strip()
    
    # Simpan interaksi yang sudah bersih ke memory (history_state)
    history_state.append(HumanMessage(content=user_prompt))
    history_state.append(AIMessage(content=clean_output))
    
    return clean_output, history_state

# --- APPROVE & EXPORT LOGIC ---
def approve_and_export(markdown_text, history_state): # Tambah parameter history_state
    # Bikin PDF pakai markdown-pdf (pure Python)
    pdf = MarkdownPdf(toc_level=2)
    pdf.add_section(Section(markdown_text))
    
    pdf_filename = "Approved_PRD.pdf"
    pdf.save(pdf_filename)
    
    # Kembalikan file PDF dan biarkan history_state UTUH
    return pdf_filename, history_state

# --- UI GRADIO (Single Form Layout) ---
with gr.Blocks() as demo:
    gr.Markdown("# 🚀 AI Product Manager Agent (PRD Generator)")
    
    # State untuk menyimpan memory per sesi
    memory_state = gr.State([])
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Form Input PM")
            i_obj = gr.Textbox(label="1. Objective & Goals", lines=2)
            i_usr = gr.Textbox(label="2. Target Users", lines=1)
            i_pain = gr.Textbox(label="3. User Pain Points", lines=2)
            i_feat = gr.Textbox(label="4. Feature Requirements", lines=2)
            i_ac = gr.Textbox(label="5. Acceptance Criteria", lines=2)
            i_out = gr.Textbox(label="6. Out of Scope", lines=2)
            
            btn_generate = gr.Button("🧠 Generate / Revise PRD", variant="primary")
            
        with gr.Column(scale=1):
            gr.Markdown("### Live PRD Output")
            out_markdown = gr.Markdown(elem_id="output_md", height=500)
            
            with gr.Row():
                btn_approve = gr.Button("✅ Approve & Generate PDF", variant="secondary")
                out_file = gr.File(label="Download PRD")

    # Events
    btn_generate.click(
        fn=generate_prd,
        inputs=[i_obj, i_usr, i_pain, i_feat, i_ac, i_out, memory_state],
        outputs=[out_markdown, memory_state]
    )
    
    # Jika di approve, generate file dan reset memory_state menjadi []
    btn_approve.click(
        fn=approve_and_export,
        inputs=[out_markdown, memory_state], # Tambahkan memory_state di sini
        outputs=[out_file, memory_state]
    )

demo.launch(theme=gr.themes.Soft())