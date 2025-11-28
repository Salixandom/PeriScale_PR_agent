import os
import sys
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Literal
from langgraph.graph import StateGraph, END

from agents.Critic import run_critique
from agents.Planner import run_generate_plan
from state import AgentState
import agent_loader

load_dotenv()

MAX_LOOPS = 3
PLANNER = "planner"
CRITIC = "critic"
HUMAN_PLANNER = "human_planner"
DIRECTOR = "director"

# NODES

def planner_node(state: AgentState):
    """Node: Generates or Refines the execution Plan"""
    
    return run_generate_plan(state)

def critic_node(state: AgentState):
    """Node: Reviews and gives feedback of the execution Plan"""
    
    new_state = run_critique(state)
    
    if new_state.plan_feedback and new_state.plan_feedback[-1].startswith("CRITIQUE:"):
        new_state.planner_loop_count += 1
    
    return new_state
    
def human_node_planner(state: AgentState):
    """Node: Human in the loop interaction"""
    
    print(f"\n👤 NODE: Human Review")
    
    if state.plan:
        print("\n📋 PROPOSED PLAN:")
        for step in state.plan.steps:
            print(
                f"\n  {step.step_number}. {step.agent_name.value}\n"
                f"  📝 Instruction: {step.instruction}\n"
                f"  💡 Reasoning:   {step.reasoning}\n"
            )    
        print(f"\n Summary: {state.plan.summary}")

    
    user_response = input("\n> Do you approve this plan? (yes/feedback): ").strip()
    
    if user_response.lower() in ["yes", "y", "ok"]:
        state.is_plan_approved = True
        print("   ✅ Human Approved.")
    else:
        state.is_plan_approved = False
        print(f"   📝 Feedback captured: {user_response}")
        state.plan_feedback.append(f"Human: {user_response}")
        
    return state

def director_node(state: AgentState):
    """Node: Executes the plan by calling the agents"""
    
    print("Director will run")

# CONDITIONAL EDGES LOGIC

def check_critic_loop(state: AgentState) -> Literal[PLANNER, HUMAN_PLANNER]:
    if state.plan_feedback and state.plan_feedback[-1].startswith("CRITIQUE:") \
       and state.planner_loop_count < MAX_LOOPS:
        return PLANNER

    return HUMAN_PLANNER


def check_human_approval(state: AgentState) -> Literal[PLANNER, DIRECTOR]:
    if state.is_plan_approved:
        return DIRECTOR
    else:
        print("   ↺ Human rejected. Re-planning...")
        return PLANNER
    
# GRAPH BUILDING

def build_graph():
    workflow = StateGraph(AgentState)
    
    workflow.add_node(PLANNER, planner_node)
    workflow.add_node(CRITIC, critic_node)
    workflow.add_node(HUMAN_PLANNER, human_node_planner)
    workflow.add_node(DIRECTOR, director_node)
    
    workflow.set_entry_point(PLANNER)
    workflow.add_edge(PLANNER, CRITIC)
    workflow.add_conditional_edges(CRITIC, check_critic_loop)
    workflow.add_conditional_edges(HUMAN_PLANNER, check_human_approval)
    workflow.set_finish_point(DIRECTOR)
    
    return workflow.compile()

if __name__ == "__main__":
    print("🚀 Initializing LangGraph Agent...")
    
    initial_query = input("Give your query: ")
    state = AgentState(user_raw_query=initial_query)
    
    app = build_graph()
    
    final_state = app.invoke(state, config = {"recursion_limit": 10})
    print("\nWorkflow Finished")