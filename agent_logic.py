import re
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

class PRDAgent:
    def __init__(self, api_key, model_name="qwen/qwen3-32b"):
        self.llm = ChatGroq(temperature=0.3, groq_api_key=api_key, model_name=model_name)

    def _clean_output(self, raw_text):
        """Internal method untuk membersihkan tag <think> dari output LLM."""
        return re.sub(r'<think>.*?</think>', '', raw_text, flags=re.DOTALL).strip()

    def generate_initial(self, inputs, feedback_context):
        """Membuat draft PRD pertama."""
        system_prompt = f"""Kamu adalah Senior Product Manager AI. Buatlah dokumen PRD dalam format Markdown.
        
        ATURAN SANGAT KETAT:
        - JANGAN PERNAH menampilkan proses berpikir, analisa, atau kalimat pembuka.
        - OUTPUT HARUS langsung dimulai dengan tag markdown "# Product Requirements Document (PRD)".
        - Gunakan bahasa Indonesia yang profesional dan lugas.
        - KEKANGAN SCOPE: Di bagian "Feature Requirements", HANYA tuliskan fitur yang secara eksplisit diminta oleh PM.
        - KEKANGAN FEEDBACK: Gunakan feedback dari database HANYA untuk memperkaya bagian "User Pain Points".
        
        Feedback dari database:
        {feedback_context}
        
        Format Wajib:
        1. Objective & Goals
        2. Target Users
        3. User Pain Points
        4. Feature Requirements
        5. Acceptance Criteria
        6. Out of Scope
        """
        
        # Susun string dari dictionary input PM
        user_content = "\n".join([f"{k}: {v}" for k, v in inputs.items()])
        user_prompt = f"Input PM:\n{user_content}"
        
        messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
        response = self.llm.invoke(messages)
        clean_md = self._clean_output(response.content)
        
        # Kembalikan hasil dan inisiasi history untuk revisi
        new_history = [
            SystemMessage(content=system_prompt), 
            HumanMessage(content=user_prompt), 
            AIMessage(content=clean_md)
        ]
        return clean_md, new_history

    def revise(self, revision_instruction, history_state):
        """Merevisi PRD berdasarkan history percakapan."""
        user_prompt = f"Tolong revisi PRD sebelumnya berdasarkan instruksi ini: {revision_instruction}. JANGAN ubah poin yang tidak diminta direvisi. Langsung keluarkan hasil revisinya dalam format Markdown lengkap."
        
        messages = history_state.copy()
        messages.append(HumanMessage(content=user_prompt))
        
        response = self.llm.invoke(messages)
        clean_md = self._clean_output(response.content)
        
        # Update history dengan jawaban baru
        messages.append(AIMessage(content=clean_md))
        
        return clean_md, messages