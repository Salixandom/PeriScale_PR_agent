import json
from dotenv import load_dotenv
from langchain_core.tools import structured
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

from state import PlanStep, ExecutionPlan, AgentState, AgentName
from prompt_template import PLANNER_GEN_PROMPT, CRITIC_PROMPT, AGENT_METADATA

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.2
)


def run_generate_plan(state: AgentState) -> AgentState:
    print(f"\n🧠 NODE: Planner (Iteration {state.planner_loop_count + 1})")
    
    feedback_str = "\n".join(state.plan_feedback) if state.plan_feedback else "No feedback yet"
    
    prompt = ChatPromptTemplate.from_template(PLANNER_GEN_PROMPT)
    structured_llm = prompt | llm.with_structured_output(ExecutionPlan)
    
    
    try:
        plan = structured_llm.invoke({
            "user_query": state.user_raw_query,
            "agents_metadata": AGENT_METADATA,
            "feedback_history": feedback_str
        })

        state.plan = plan
        print(f"   📝 Plan Generated with {len(plan.steps)} steps.")
    except Exception as e:
        print(f"   ❌ Planner Error: {e}")
        if not state.error_message:
            state.error_message = {}
        state.error_message["planner"] = str(e)
        
    return state