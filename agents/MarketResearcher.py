import os
import json
import time
from tavily import TavilyClient
from firecrawl import FirecrawlApp
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from state import AgentState, MarketResearchData
from dotenv import load_dotenv

from prompt_template import MARKET_RESEARCHER_SYSTEM_PROMPT

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.1
)
tavily = TavilyClient()
firecrawl = FirecrawlApp()

SCRAPPER = "firecrawl"

def run_market_research(state: AgentState) -> AgentState:
    print(f"\n🕵️  AGENT: Starting Market Research...")
    
    if not state.parsed_query or not state.parsed_query.search_keywords:
        print("⚠️  No keywords found from Parser. Skipping.")
        return state

    keywords = state.parsed_query.search_keywords
    print(f"🔎 Processing {len(keywords)} keywords from parser...")
    
    unique_results = []
    seen_urls = set()
    
    for keyword in keywords:
        print(f"   🔗 Searching: '{keyword}'...")
        try:
            if SCRAPPER != "firecrawl":
                response = tavily.search(
                    query=keyword,
                    search_depth="basic",
                    max_results=5,
                    include_domains=[],
                    include_answer=False
                )
                
                results = response.get('results', [])
                new_items = 0
                
                for item in results:
                    url = item.get('url')
                    if url and url not in seen_urls:
                        seen_urls.add(url)
                        unique_results.append(item)
                        new_items += 1
                
                time.sleep(0.5)
            else:
                response = firecrawl.search(
                    query=keyword,
                    limit=5
                )
                
                results = response.web
                new_items = 0
                
                for item in results:
                    url = item.url
                    title = item.title
                    description = item.description
                    
                    if url and url not in seen_urls:
                        seen_urls.add(url)
                        unique_results.append({
                            "url": url,
                            "title": title,
                            "description": description
                        })
                        new_items += 1
                
                time.sleep(0.5)
            
            
        except Exception as e:
            print(f"   ❌ Error searching for '{keyword}': {e}")
            continue

    if not unique_results:
        print("❌ No results found for any keywords.")
        return state

    print(f"🧠 ANALYZING {len(unique_results)} unique results with Gemini...")
    
    try:
        analyze_prompt = ChatPromptTemplate.from_template(MARKET_RESEARCHER_SYSTEM_PROMPT)
        structured_llm = analyze_prompt | llm.with_structured_output(MarketResearchData)
        
        result = structured_llm.invoke({
            "product_name": state.parsed_query.product_name or "the product",
            "search_data": json.dumps(unique_results)
        })
        
        state.market_research_data = result
        print(f"✅ SUCCESS: Found {len(result.competitors)} competitors from combined search.")
        
    except Exception as e:
        print(f"❌ ERROR in Market Research Analysis: {str(e)}")
        if not state.error_message:
            state.error_message = {}
        state.error_message["market_research"] = str(e)
        
    return state