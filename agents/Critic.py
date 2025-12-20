from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

from state import CritiqueOutput, AgentState
from prompt_template import CRITIC_PROMPT
from llm_gateway import gateway 

load_dotenv()


def run_critique(state: AgentState) -> AgentState:
    print(f"\n🧐 NODE: Critic")
    
    if not state.plan:
        print("   ⚠️ No plan to critique.")
        return state
    
    prompt = ChatPromptTemplate.from_template(CRITIC_PROMPT)
    
    try:
        messages = prompt.format_messages(plan=state.plan.model_dump_json())
        
        critique = gateway.invoke(
            messages=messages,
            structured_output=CritiqueOutput
        )
        
        if critique.is_approved:
            print("   ✅ Plan Approved by Critic.")
        else:
            print(f"   🛑 Critique: {critique.feedback}")
            state.plan_feedback.append(f"CRITIQUE: {critique.feedback}")
            
    except Exception as e:
        print(f"   ❌ Critic Error: {e}")
        if not state.error_message:
            state.error_message = {}
        state.error_message["critique"] = str(e)
    
    return state