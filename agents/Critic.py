from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

from state import CritiqueOutput, AgentState
from prompt_template import CRITIC_PROMPT

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.1
)

def run_critique(state: AgentState) -> AgentState:
    print(f"\n🧐 NODE: Critic")
    
    if not state.plan:
        print("   ⚠️ No plan to critique.")
        return state
    
    prompt = ChatPromptTemplate.from_template(CRITIC_PROMPT)
    structured_llm = prompt | llm.with_structured_output(CritiqueOutput)
    
    try:
        critique = structured_llm.invoke({
            "plan": state.plan.model_dump_json()
        })
        
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