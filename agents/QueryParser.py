import os
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from dotenv import load_dotenv

from state import AgentState, ParsedQuery
from prompt_template import QUERY_PARSER_SYSTEM_PROMPT
from llm_gateway import gateway

load_dotenv()

query_parser_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(QUERY_PARSER_SYSTEM_PROMPT),
    HumanMessagePromptTemplate.from_template("{input}")
])


def run_query_parser(state: AgentState) -> AgentState:
    """Runs the query parser agent"""
    try:
        messages = query_parser_prompt.format_messages(input=state.user_raw_query)

        result = gateway.invoke(
            messages=messages,
            structured_output=ParsedQuery
        )
        
        state.parsed_query = result
        return state
    except Exception as e:
        state.error_message = {"query_parser": str(e)}
        return state