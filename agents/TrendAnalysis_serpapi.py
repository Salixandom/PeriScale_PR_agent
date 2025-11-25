import os
import time
import random
import pycountry
from datetime import datetime, timedelta
import serpapi
from dotenv import load_dotenv
from state import (
    AgentState, TrendAnalysisData, TrendMetrics, 
    TrendDirection, KeywordTrendAnalysis
)

load_dotenv()
client = serpapi.Client()

def get_country_code(country_name: str) -> str:
    """Converts 'France' -> 'FR'. Defaults to World if unknown."""
    if not country_name or country_name.lower() in ["global", "world"]:
        return "" 
    try:
        return pycountry.countries.lookup(country_name).alpha_2
    except LookupError:
        return ""

def run_trend_analysis(state: AgentState) -> AgentState:
    print(f"\n📈 AGENT: Starting Trend Analysis (SerpApi)...")
    
    if not state.parsed_query or not state.parsed_query.search_keywords:
        print("⚠️  No keywords found. Skipping.")
        return state

    # 1. Setup Region
    target_country = state.parsed_query.target_market
    geo_code = get_country_code(target_country)
    
    # 2. Setup Timeframe (Last 3 Years)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=3*365)
    timeframe_str = f"{start_date.strftime('%Y-%m-%d')} {end_date.strftime('%Y-%m-%d')}"
    
    print(f"   🌍 Target: {target_country} (Code: '{geo_code}')")
    print(f"   📅 Range:  {timeframe_str}")

    # 3. Process Top 3 Keywords
    keywords_to_check = state.parsed_query.search_keywords[:3]
    analyzed_trends = []
    all_related_queries = set()

    api_key = os.getenv("SERPAPI_API_KEY")
    if not api_key:
        print("❌ Missing SERPAPI_API_KEY in .env")
        return state

    for keyword in keywords_to_check:
        print(f"   📊 Analyzing: '{keyword}'...")
        
        params = {
            "engine": "google_trends",
            "q": keyword,
            "data_type": "TIMESERIES",
            "date": timeframe_str,
            "geo": geo_code,
            "api_key": api_key
        }

        try:
            search = client.search(params)
            # FIX: Access dictionary directly
            results = search
            
            if "error" in results:
                print(f"      ❌ SerpApi Error: {results['error']}")
                continue

            # Extract Timeline
            interest_over_time = results.get("interest_over_time", {})
            timeline_data = interest_over_time.get("timeline_data", [])
            
            if not timeline_data:
                print(f"      ⚠️ No data points found for '{keyword}'.")
                continue

            # --- PROCESS DATA ---
            history_points = []
            values = []
            
            for point in timeline_data:
                # SerpApi Timestamp is a string, convert to int then datetime
                ts = int(point.get('timestamp', 0))
                dt = datetime.fromtimestamp(ts)
                
                # Extract Value (Handle " < 1" cases if they exist)
                val_list = point.get('values', [])
                if val_list:
                    # extracted_value is usually an integer 0-100
                    val = val_list[0].get('extracted_value', 0)
                    values.append(val)
                    history_points.append(TrendMetrics(
                        date=dt,
                        interest_value=val
                    ))

            # --- CALCULATE METRICS ---
            if not values:
                continue

            avg_val = sum(values) / len(values)
            
            # Trend Direction (Last 6 months ~ 26 weeks)
            if len(values) > 26:
                recent_avg = sum(values[-26:]) / 26
                old_avg = sum(values[-52:-26]) / 26 if len(values) > 52 else avg_val
                
                if recent_avg > old_avg * 1.1:
                    direction = TrendDirection.RISING
                elif recent_avg < old_avg * 0.9:
                    direction = TrendDirection.FALLING
                else:
                    direction = TrendDirection.STABLE
            else:
                direction = TrendDirection.STABLE

            # Peak Month (Manual Grouping)
            month_sums = {}
            month_counts = {}
            
            for pt in history_points:
                m = pt.date.month
                month_sums[m] = month_sums.get(m, 0) + pt.interest_value
                month_counts[m] = month_counts.get(m, 0) + 1
            
            best_month_idx = 0
            highest_avg = -1
            
            for m in range(1, 13):
                if m in month_sums and month_counts[m] > 0:
                    avg = month_sums[m] / month_counts[m]
                    if avg > highest_avg:
                        highest_avg = avg
                        best_month_idx = m
            
            peak_month = datetime(2000, best_month_idx, 1).strftime('%B') if best_month_idx > 0 else "N/A"

            # Add to Analysis
            analyzed_trends.append(KeywordTrendAnalysis(
                keyword=keyword,
                trend_direction=direction,
                average_interest=round(avg_val, 1),
                peak_month=peak_month,
                interest_over_time=history_points
            ))
            
            print(f"      ✅ Got {len(history_points)} data points.")
            
        except Exception as e:
            print(f"      ⚠️ Error on '{keyword}': {e}")
            continue

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