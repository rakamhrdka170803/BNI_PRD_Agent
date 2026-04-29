from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

class PRDAgent:
    def __init__(self, api_key, model_name="qwen/qwen3-32b"):
        self.llm = ChatGroq(temperature=0.3, groq_api_key=api_key, model_name=model_name)

    def build_prompt(self, inputs, feedback_context):
        system_msg = f"""You are an expert Product Manager. Create a professional PRD in Markdown format.
        Use the provided user feedback to strengthen the 'Pain Points' and 'Justification' sections.
        
        USER FEEDBACK FROM DATABASE:
        {feedback_context}
        
        STRICT 6-POINT FORMAT:
        1. Objective & Goals
        2. Target Users
        3. User Pain Points (Combine PM input + Database feedback)
        4. Feature Requirements
        5. Acceptance Criteria
        6. Out of Scope
        """
        
        user_content = "\n".join([f"{k}: {v}" for k, v in inputs.items()])
        return system_msg, user_content

    def generate(self, system_prompt, user_content, history):
        messages = [SystemMessage(content=system_prompt)]
        # Add history for revision context
        messages.extend(history)
        messages.append(HumanMessage(content=user_content))
        
        response = self.llm.invoke(messages)
        return response.content