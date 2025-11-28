import os
import requests
import pandas as pd
import time
import random
import pycountry
from datetime import datetime, timedelta
from pytrends.request import TrendReq
from fake_useragent import UserAgent
from dotenv import load_dotenv
from state import (
    AgentState, TrendAnalysisData, TrendMetrics, 
    TrendDirection, KeywordTrendAnalysis
)

load_dotenv()

def get_country_code(country_name: str) -> str:
    if not country_name or country_name.lower() in ["global", "world"]:
        return "" 
    try:
        return pycountry.countries.lookup(country_name).alpha_2
    except LookupError:
        return ""

def fetch_webshare_proxies():
    """
    Fetches the live list of proxies from Webshare API.
    Returns a list of formatted strings: 'http://user:pass@ip:port'
    """
    api_token = os.getenv("WEBSHARE_API_KEY")
    if not api_token:
        print("   ⚠️ No WEBSHARE_API_TOKEN found. Skipping API fetch.")
        return []

    print("   🛡️  Fetching fresh proxy list from Webshare API...")
    try:
        # We use the /proxy/list/ endpoint to get the actual IPs
        response = requests.get(
            "https://proxy.webshare.io/api/v2/proxy/list/",
            headers={"Authorization": f"Token {api_token}"},
            params={"mode": "direct", "page": 1, "page_size": 25}
        )
        response.raise_for_status()
        data = response.json()
        
        formatted_proxies = []
        for p in data.get('results', []):
            # Format: http://username:password@ip:port
            # Webshare JSON keys: 'username', 'password', 'proxy_address', 'port'
            proxy_str = f"http://{p['username']}:{p['password']}@{p['proxy_address']}:{p['port']}"
            formatted_proxies.append(proxy_str)
            
        print(f"   ✅ Loaded {len(formatted_proxies)} proxies from Webshare.")
        return formatted_proxies
        
    except Exception as e:
        print(f"   ❌ Failed to fetch Webshare proxies: {e}")
        return []

def run_trend_analysis(state: AgentState) -> AgentState:
    print(f"\n📈 AGENT: Starting Trend Analysis (Webshare API Mode)...")
    
    if not state.parsed_query or not state.parsed_query.search_keywords:
        print("⚠️  No keywords found. Skipping.")
        return state

    # 1. Setup Region & Timeframe
    target_country = state.parsed_query.target_market
    geo_code = get_country_code(target_country)
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=3*365)
    timeframe_str = f"{start_date.strftime('%Y-%m-%d')} {end_date.strftime('%Y-%m-%d')}"
    
    print(f"   🌍 Target: {target_country} (Code: '{geo_code}')")
    print(f"   📅 Range:  {timeframe_str}")

    # 2. Get Proxies via API
    ua = UserAgent()
    
    # Fetch the list once
    all_proxies = fetch_webshare_proxies()
    
    # If API fails, try the static fallback from env vars if they exist
    if not all_proxies and os.getenv("WEBSHARE_USERNAME"):
        fallback = f"http://{os.getenv('WEBSHARE_USERNAME')}:{os.getenv('WEBSHARE_PASSWORD')}@p.webshare.io:80"
        all_proxies = [fallback]

    # 3. Analyze Keywords
    keywords_to_check = state.parsed_query.search_keywords
    analyzed_trends = []
    all_related_queries = set()

    for i, keyword in enumerate(keywords_to_check):
        print(f"   📊 Analyzing: '{keyword}'...")
        
        # ROTATION LOGIC: Pick a different proxy for each keyword if available
        # This reduces the chance of a ban significantly
        if all_proxies:
            current_proxy = [all_proxies[i % len(all_proxies)]] # Rotate based on index
        else:
            current_proxy = [] # Local IP fallback

        try:
            # Initialize Pytrends with specific proxy for this request
            pytrends = TrendReq(
                hl='en-US', 
                tz=360, 
                proxies=current_proxy, 
                requests_args={'headers': {'User-Agent': ua.random}} 
            )
            
            pytrends.build_payload(
                [keyword], 
                cat=0, 
                timeframe=timeframe_str, 
                geo=geo_code, 
                gprop=''
            )
            
            # Random sleep
            time.sleep(random.uniform(2, 3))

            df = pytrends.interest_over_time()
            
            if df.empty:
                print(f"      ❌ No data for '{keyword}'")
                continue

            # --- METRICS LOGIC ---
            data_col = df[keyword]
            
            zeros_count = (data_col == 0).sum()
            total_points = len(data_col)
            
            if zeros_count / total_points > 0.7:
                print(f"      ⚠️ '{keyword}' has insufficient data (mostly 0s).")
                direction = TrendDirection.STABLE
                avg_val = 0.0
            else:
                # SMA Logic
                sma_12 = data_col.rolling(window=12).mean()
                current_trend_val = sma_12.iloc[-1]
                past_trend_val = sma_12.iloc[-26] 
                
                if pd.isna(current_trend_val) or pd.isna(past_trend_val):
                     direction = TrendDirection.STABLE
                elif current_trend_val > past_trend_val * 1.10:
                    direction = TrendDirection.RISING
                elif current_trend_val < past_trend_val * 0.90:
                    direction = TrendDirection.FALLING
                else:
                    direction = TrendDirection.STABLE
                
                avg_val = float(data_col.mean())

            # Peak Month
            try:
                monthly_seasonality = data_col.groupby(data_col.index.month).mean()
            except:
                monthly_seasonality = data_col.resample('M').mean()

            if not monthly_seasonality.empty:
                peak_month_num = monthly_seasonality.idxmax()
                peak_month = datetime(2000, peak_month_num, 1).strftime('%B')
            else:
                peak_month = "N/A"

            # History
            history_points = []
            for date, val in data_col.items():
                history_points.append(TrendMetrics(
                    date=date,
                    interest_value=int(val)
                ))

            # Related Queries
            if keyword == keywords_to_check[0]:
                try:
                    related = pytrends.related_queries()
                    if related and keyword in related:
                        rising = related[keyword].get('rising')
                        if rising is not None:
                            all_related_queries.update(rising['query'].head(5).tolist())
                except:
                    pass

            analyzed_trends.append(KeywordTrendAnalysis(
                keyword=keyword,
                trend_direction=direction,
                average_interest=round(avg_val, 1),
                peak_month=peak_month,
                interest_over_time=history_points
            ))
            
            print(f"      ✅ Got {len(history_points)} weeks of data.")

        except Exception as e:
            print(f"      ⚠️ Error on '{keyword}': {e}")
            time.sleep(2)

    if not analyzed_trends:
        print("❌ Failed to analyze any keywords.")
        return state

    # 4. Overall Assessment
    directions = [t.trend_direction for t in analyzed_trends]
    if directions.count(TrendDirection.RISING) >= directions.count(TrendDirection.FALLING):
        overall = TrendDirection.RISING
    elif directions.count(TrendDirection.FALLING) > len(directions) / 2:
        overall = TrendDirection.FALLING
    else:
        overall = TrendDirection.STABLE

    # 5. Save to State
    state.trend_analysis_data = TrendAnalysisData(
        target_country_iso=geo_code if geo_code else "Global",
        timeframe=timeframe_str,
        keyword_trends=analyzed_trends,
        overall_market_direction=overall,
        top_related_queries=list(all_related_queries)
    )
    
    print(f"   ✅ SUCCESS: Market is {overall.value.upper()}.")
    return state