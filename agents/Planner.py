from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate

from state import ExecutionPlan, AgentState
from prompt_template import PLANNER_GEN_PROMPT, AGENT_METADATA
from llm_gateway import gateway

load_dotenv()


def run_generate_plan(state: AgentState) -> AgentState:
    print(f"\n🧠 NODE: Planner (Iteration {state.planner_loop_count + 1})")
    
    feedback_str = "\n".join(state.plan_feedback) if state.plan_feedback else "No feedback yet"
    
    prompt = ChatPromptTemplate.from_template(PLANNER_GEN_PROMPT)
    
    try:
        messages = prompt.format_messages(
            user_query=state.user_raw_query,
            agents_metadata=AGENT_METADATA,
            feedback_history=feedback_str
        )
        
        plan = gateway.invoke(
            messages=messages,
            structured_output=ExecutionPlan
        )

        state.plan = plan
        print(f"   📝 Plan Generated with {len(plan.steps)} steps.")
        
    except Exception as e:
        print(f"   ❌ Planner Error: {e}")
        if not state.error_message:
            state.error_message = {}
        state.error_message["planner"] = str(e)
        
    return state