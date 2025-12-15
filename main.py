import os
from state import AgentState
from agents.QueryParser import run_query_parser
from agents.MarketResearcher import run_market_research
from agents.TrendAnalysis import run_trend_analysis
from agents.SupplierSourcing import run_supplier_sourcing


def main():
    # user_input = "I want to sell organic bamboo toothbrushes in France. Is the market saturated?"
    
    user_input = input("Enter you query: ")
    
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
            # state = run_market_research(state)
            
            # if state.market_research_data:
            #     print(f"\n--- 📊 MARKET RESEARCH RESULTS ({len(state.market_research_data.competitors)} found) ---")
            #     print(f"📝 Summary: {state.market_research_data.market_summary}\n")
                
            #     for idx, comp in enumerate(state.market_research_data.competitors, 1):
            #         price_display = f"{comp.price} {comp.currency.value}" if comp.price else "Price Hidden"
            #         print(f"{idx}. {comp.name} | {price_display}")
            #         print(f"   🔗 {comp.url}")
            #         if comp.features:
            #             print(f"   ✨ Features: {', '.join(comp.features[:2])}")
            #         if comp.rating:
            #             print(f"   🌟 Rating: {comp.rating}")
            #         if comp.Reviews_Comments:
            #             for rc in comp.Reviews_Comments:
            #                 print(f"   📝 {rc.comment}")
            #                 if rc.rating:
            #                     print(f"   🌟 {rc.rating}")
            #         print("")
            # else:
            #     print("\n❌ No competitors found.")
            
            # state = run_trend_analysis(state)
            
            # if state.trend_analysis_data:
            #     print(f"\n--- 📊 TREND ANALYSIS RESULTS ---")
            #     print(f"🎯 Overall Market Direction: {state.trend_analysis_data.overall_market_direction}")
            #     print(f"🎯 Top Related Queries: {', '.join(state.trend_analysis_data.top_related_queries)}")
            #     print("\nInterest Over Time:")
            #     for idx, metrics in enumerate(state.trend_analysis_data.keyword_trends, 1):
            #         for idx2, metrics2 in enumerate(metrics.interest_over_time, 1):
            #             print(f"{idx} | {idx2}. {metrics2.date}: {metrics2.interest_value}")
            
            state = run_supplier_sourcing(state)
            
            if state.supplier_data:
                print(f"\n--- 📊 SUPPLIER SOURCING RESULTS ---")
                print(f"🎯 Average Unit Cost: {state.supplier_data.average_unit_cost}")
                print(f"🎯 Recommended Supplier: {state.supplier_data.recommended_supplier.supplier_name}")
                print(f"🎯 Suppliers: {len(state.supplier_data.suppliers)}")
                for supplier in state.supplier_data.suppliers:
                    print(f"   🔗 {supplier.product_url}")
                    print(f"   💰 Price: {supplier.price_per_unit}")
    else:
        print("Failed to parse query.")

if __name__ == "__main__":
    main()