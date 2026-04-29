import re
import llmops
from llmops import SpanType
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

class PRDAgent:
    def __init__(self, api_key, model_name="qwen/qwen3-32b"):
        self.model_name = model_name
        self.temperature = 0.3
        self.llm = ChatGroq(temperature=self.temperature, groq_api_key=api_key, model_name=model_name)
        self.prompt_template = llmops.load_prompt("prd_system@production")

    def _clean_output(self, raw_text):
        """Internal method untuk membersihkan tag <think> dari output LLM."""
        return re.sub(r'<think>.*?</think>', '', raw_text, flags=re.DOTALL).strip()

    @llmops.trace_agent("prd_generate_initial", span_type=SpanType.AGENT)
    def generate_initial(self, inputs, feedback_context):
        """Membuat draft PRD pertama."""
        llmops.set_trace_tags(env="dev", agent="prd-generator", method="generate_initial")
        llmops.log_hyperparams(model=self.model_name, temperature=self.temperature)

        system_prompt = self.prompt_template.format(feedback_context=feedback_context)

        user_content = "\n".join([f"{k}: {v}" for k, v in inputs.items()])
        user_prompt = f"Input PM:\n{user_content}"

        messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
        response = self.llm.invoke(messages)
        clean_md = self._clean_output(response.content)

        new_history = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
            AIMessage(content=clean_md),
        ]
        return clean_md, new_history

    @llmops.trace_agent("prd_revise", span_type=SpanType.AGENT)
    def revise(self, revision_instruction, history_state):
        """Merevisi PRD berdasarkan history percakapan."""
        llmops.set_trace_tags(env="dev", agent="prd-generator", method="revise")
        llmops.log_hyperparams(model=self.model_name, temperature=self.temperature)

        user_prompt = f"Tolong revisi PRD sebelumnya berdasarkan instruksi ini: {revision_instruction}. JANGAN ubah poin yang tidak diminta direvisi. Langsung keluarkan hasil revisinya dalam format Markdown lengkap."

        messages = history_state.copy()
        messages.append(HumanMessage(content=user_prompt))

        response = self.llm.invoke(messages)
        clean_md = self._clean_output(response.content)

        messages.append(AIMessage(content=clean_md))
        return clean_md, messages