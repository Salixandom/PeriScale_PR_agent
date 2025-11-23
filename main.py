import os
from state import AgentState
from agents.QueryParser import run_query_parser
from agents.MarketResearcher import run_market_research


def main():
    user_input = "I want to sell organic bamboo toothbrushes in France. Is the market saturated?"
    
    state = AgentState(user_raw_query=user_input)
    state = run_query_parser(state)
    
    if state.parsed_query:
        pq = state.parsed_query
        print("\n--- 🎯 RESULT ------------------")
        print(f"Product:  {pq.product_name}")
        print(f"Category: {pq.category}")
        print(f"Price:    {pq.price}")
        print(f"Currency: {pq.currency}")
        print(f"Market:   {pq.target_market}")
        print(f"Intent:   {pq.query_intent}")
        
        if pq.search_keywords:
            state = run_market_research(state)
            
            if state.market_research_data:
                print(f"\n--- 📊 MARKET RESEARCH RESULTS ({len(state.market_research_data.competitors)} found) ---")
                print(f"📝 Summary: {state.market_research_data.market_summary}\n")
                
                for idx, comp in enumerate(state.market_research_data.competitors, 1):
                    price_display = f"{comp.price} {comp.currency.value}" if comp.price else "Price Hidden"
                    print(f"{idx}. {comp.name} | {price_display}")
                    print(f"   🔗 {comp.url}")
                    if comp.features:
                        print(f"   ✨ Features: {', '.join(comp.features[:2])}")
                    if comp.rating:
                        print(f"   🌟 Rating: {comp.rating}")
                    if comp.Reviews_Comments:
                        for rc in comp.Reviews_Comments:
                            print(f"   📝 {rc.comment}")
                            if rc.rating:
                                print(f"   🌟 {rc.rating}")
                    print("")
            else:
                print("\n❌ No competitors found.")
    else:
        print("Failed to parse query.")

if __name__ == "__main__":
    main()