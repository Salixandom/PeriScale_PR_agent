import time
from tavily import TavilyClient
from firecrawl import FirecrawlApp
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from state import AgentState, SupplierSourcingData, SupplierData
from prompt_template import PAGE_ANALYSIS_PROMPT

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)
tavily = TavilyClient()
firecrawl = FirecrawlApp() 

def run_supplier_sourcing(state: AgentState) -> AgentState:
    print(f"\n📦 AGENT: Starting Supplier Sourcing...")
    
    if not state.parsed_query or not state.parsed_query.product_name:
        print("⚠️  No product name found. Skipping.")
        return state

    product_name = state.parsed_query.product_name

    # --- PHASE 1: DISCOVERY (Tavily) ---
    print(f"   🔎 Scouting for suppliers of '{product_name}'...")
    
    search_query = f"wholesale {product_name} manufacturer supplier price"
    
    try:
        tavily_response = tavily.search(
            query=search_query,
            search_depth="basic",
            max_results=10,
            include_domains=["alibaba.com", "aliexpress.com", "made-in-china.com", "dhgate.com", "indiamart.com", "thomasnet.com"], 
            include_answer=False
        )
        
        candidate_urls = []
        for res in tavily_response.get('results', []):
            url = res.get('url')
            # Avoid search result pages, we want product pages
            if "search" not in url and "category" not in url:
                candidate_urls.append(url)
        
        print(f"   🎯 Identified {len(candidate_urls)} candidate URLs for scraping.")

    except Exception as e:
        print(f"   ❌ Tavily Search failed: {e}")
        return state

    if not candidate_urls:
        print("   ⚠️ No direct product pages found.")
        return state

    # --- PHASE 2: EXTRACTION (Firecrawl) ---
    valid_suppliers = []
    
    for url in candidate_urls:
        print(f"   🕷️  Scraping: {url}...")
        try:
            scrape_result = firecrawl.scrape(
                url, 
                formats=["markdown", "html"]
            )

            markdown_content = getattr(scrape_result, 'markdown', '')
            html_content = getattr(scrape_result, 'html', '')

            if not markdown_content:
                 print(f"      ⚠️  No markdown content returned for {url}")
                 continue
            
            # --- PHASE 3: PARSING (Gemini) ---
            analyze_prompt = ChatPromptTemplate.from_template(PAGE_ANALYSIS_PROMPT)
            structured_llm = analyze_prompt | llm.with_structured_output(SupplierData)
            
            supplier_info = structured_llm.invoke({
                "product_name": product_name,
                "markdown_content": markdown_content
            })
            
            # Fill in metadata
            supplier_info.product_url = url
            
            # Logic check: Only add if price is found
            if supplier_info.price_per_unit and supplier_info.price_per_unit > 0:
                valid_suppliers.append(supplier_info)
                print(f"      ✅ Found: {supplier_info.supplier_name} (${supplier_info.price_per_unit})")
            else:
                print(f"      ⚠️  Could not extract price from {url}")
                
            # Politeness sleep between scrapes
            time.sleep(2)

        except Exception as e:
            print(f"      ❌ Failed to scrape {url}: {e}")
            continue

    if not valid_suppliers:
        print("   ❌ No valid supplier data extracted.")
        return state

    # --- PHASE 4: AGGREGATION ---
    # Calculate Average
    prices = [s.price_per_unit for s in valid_suppliers]
    if prices:
        avg_cost = sum(prices) / len(prices)
    else:
        avg_cost = 0.0
    
    # Recommend Lowest Price for now
    if valid_suppliers:
        best_supplier = min(valid_suppliers, key=lambda x: x.price_per_unit)
    else:
        best_supplier = None

    sourcing_data = SupplierSourcingData(
        suppliers=valid_suppliers,
        average_unit_cost=round(avg_cost, 2),
        recommended_supplier=best_supplier
    )
    
    state.supplier_data = sourcing_data
    print(f"   ✅ SUCCESS: Sourcing complete. Avg Cost: ${avg_cost:.2f}")
    
    return state