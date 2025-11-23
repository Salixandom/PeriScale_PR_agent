import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from dotenv import load_dotenv

from state import AgentState, ParsedQuery
from prompt_template import QUERY_PARSER_SYSTEM_PROMPT

load_dotenv()
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)

query_parser_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(QUERY_PARSER_SYSTEM_PROMPT),
    HumanMessagePromptTemplate.from_template("{input}")
])

query_parser_llm = query_parser_prompt | llm.with_structured_output(ParsedQuery)

def run_query_parser(state: AgentState) -> AgentState:
    """ Runs the query parser agent"""
    try:
        result = query_parser_llm.invoke(
            {"input": state.user_raw_query}
        )
        state.parsed_query = result
        return state
    except Exception as e:
        state.error_message = {"query_parser": str(e)}
        return state