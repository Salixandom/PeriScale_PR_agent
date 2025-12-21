"""
Financial Modeling Agent

This agent handles:
1. Marketing CPA estimation (using social media intel)
2. Pricing strategy (based on market research)
3. Profit calculations (using logistics data for costs)
4. Break-even analysis
5. ROI projections
6. Go/No-Go recommendations

Prerequisites:
- market_research_data (for competitive pricing)
- supplier_data (for product costs)
- logistics_data (for shipping & customs) ← FROM LOGISTICS AGENT
- social_media_intel (optional, for CPA refinement)
"""

from pydantic import BaseModel, Field
from typing import List, Optional
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate

from state import AgentState, FinancialModelingData, FinancialMetrics, SentimentType
from llm_gateway import gateway

load_dotenv()


# ==========================================
# DATA MODELS
# ==========================================

class MarketingCPAEstimate(BaseModel):
    """Marketing cost per acquisition estimate"""
    base_cpa_percentage: float = Field(description="Base CPA as % of selling price (e.g., 0.25 = 25%)")
    competition_multiplier: float = Field(description="Adjustment for competition level")
    sentiment_multiplier: float = Field(description="Adjustment for market sentiment")
    viral_multiplier: float = Field(description="Adjustment for viral potential")
    final_cpa_percentage: float = Field(description="Final adjusted CPA %")
    reasoning: str = Field(description="Explanation of the estimate")


# ==========================================
# MARKETING CPA ESTIMATION
# ==========================================

def estimate_marketing_cpa(state: AgentState, target_price: float) -> tuple[float, List[str]]:
    """
    Estimate marketing CPA using multi-source intelligence
    
    Sources:
    1. Industry baseline (by category)
    2. Social media intel (competition, sentiment, viral potential)
    3. Market research (saturation level)
    4. Trend analysis (market momentum)
    
    Returns:
        (cpa_dollar_amount, reasoning_list)
    """
    
    print(f"      📢 Estimating Marketing CPA...")
    
    category = state.parsed_query.category or "general"
    
    # ==========================================
    # BASE RATE (Industry Benchmarks)
    # ==========================================
    
    industry_cpa_rates = {
        "electronics": 0.30,           # 30% - High competition
        "phone accessories": 0.28,
        "computers": 0.32,
        "clothing": 0.28,
        "fashion": 0.30,
        "footwear": 0.27,
        "beauty & personal care": 0.25,
        "cosmetics": 0.26,
        "home & kitchen": 0.22,        # 22% - Medium competition
        "furniture": 0.20,
        "toys": 0.20,
        "sports & outdoors": 0.23,
        "jewelry": 0.35,               # 35% - Very high competition
        "watches": 0.33,
        "books": 0.18,                 # 18% - Low competition
        "pet supplies": 0.22,
        "automotive": 0.25,
        "general": 0.25,               # 25% - Default
    }
    
    category_lower = category.lower()
    base_cpa_pct = 0.25  # Default
    
    for key, rate in industry_cpa_rates.items():
        if key in category_lower:
            base_cpa_pct = rate
            break
    
    print(f"         Industry baseline: {base_cpa_pct*100:.1f}% for {category}")
    
    reasoning = [f"Industry baseline for '{category}': {base_cpa_pct*100:.1f}%"]
    
    # ==========================================
    # SOCIAL MEDIA ADJUSTMENTS
    # ==========================================
    
    competition_multiplier = 1.0
    sentiment_multiplier = 1.0
    viral_multiplier = 1.0
    
    if state.social_media_intel:
        intel = state.social_media_intel
        
        # 1. Competition Factor
        total_competitor_ads = sum(
            c.total_active_ads for c in intel.competitor_ad_strategies
        )
        
        if total_competitor_ads > 100:
            competition_multiplier = 1.5
            reasoning.append(f"High ad competition ({total_competitor_ads} competitor ads) → +50% CPA")
        elif total_competitor_ads > 50:
            competition_multiplier = 1.3
            reasoning.append(f"Medium-high competition ({total_competitor_ads} ads) → +30% CPA")
        elif total_competitor_ads > 20:
            competition_multiplier = 1.1
            reasoning.append(f"Moderate competition ({total_competitor_ads} ads) → +10% CPA")
        else:
            competition_multiplier = 0.9
            reasoning.append(f"Low competition ({total_competitor_ads} ads) → -10% CPA")
        
        # 2. Sentiment Factor
        if intel.overall_sentiment == SentimentType.POSITIVE:
            sentiment_multiplier = 0.85
            reasoning.append("Positive market sentiment → -15% CPA (organic reach)")
        elif intel.overall_sentiment == SentimentType.NEGATIVE:
            sentiment_multiplier = 1.2
            reasoning.append("Negative sentiment → +20% CPA (need more ads to convert)")
        else:
            reasoning.append("Neutral sentiment → No CPA adjustment")
        
        # 3. Viral Potential
        viral_themes_count = len(intel.viral_content_themes)
        if viral_themes_count >= 3:
            viral_multiplier = 0.8
            reasoning.append(f"High viral potential ({viral_themes_count} themes) → -20% CPA")
        elif viral_themes_count >= 1:
            viral_multiplier = 0.9
            reasoning.append(f"Some viral potential ({viral_themes_count} theme) → -10% CPA")
        else:
            reasoning.append("No viral content detected → No CPA adjustment")
    
    else:
        reasoning.append("No social media data → Using baseline only")
    
    # ==========================================
    # MARKET SATURATION ADJUSTMENT
    # ==========================================
    
    saturation_multiplier = 1.0
    
    if state.market_research_data:
        num_competitors = len(state.market_research_data.competitors)
        
        if num_competitors > 15:
            saturation_multiplier = 1.3
            reasoning.append(f"High market saturation ({num_competitors} competitors) → +30% CPA")
        elif num_competitors > 10:
            saturation_multiplier = 1.15
            reasoning.append(f"Moderate saturation ({num_competitors} competitors) → +15% CPA")
        elif num_competitors < 3:
            saturation_multiplier = 0.85
            reasoning.append(f"Low saturation ({num_competitors} competitors) → -15% CPA")
    
    # ==========================================
    # TREND MOMENTUM ADJUSTMENT
    # ==========================================
    
    trend_multiplier = 1.0
    
    if state.trend_analysis_data:
        from state import TrendDirection
        
        overall_trend = state.trend_analysis_data.overall_market_direction
        
        if overall_trend == TrendDirection.RISING:
            trend_multiplier = 0.9
            reasoning.append("Rising market trend → -10% CPA (easier to acquire)")
        elif overall_trend == TrendDirection.FALLING:
            trend_multiplier = 1.15
            reasoning.append("Falling market trend → +15% CPA (harder to acquire)")
    
    # ==========================================
    # FINAL CALCULATION
    # ==========================================
    
    adjusted_cpa_pct = (
        base_cpa_pct *
        competition_multiplier *
        sentiment_multiplier *
        viral_multiplier *
        saturation_multiplier *
        trend_multiplier
    )
    
    # Convert to dollar amount
    cpa_dollar = target_price * adjusted_cpa_pct
    
    # Floor: Minimum $3 to acquire a customer
    cpa_dollar = max(cpa_dollar, 3.0)
    
    print(f"         Final CPA: ${cpa_dollar:.2f} ({adjusted_cpa_pct*100:.1f}% of price)")
    
    reasoning.append(f"Final CPA: ${cpa_dollar:.2f} ({adjusted_cpa_pct*100:.1f}% of selling price)")
    
    return round(cpa_dollar, 2), reasoning


# ==========================================
# PRICING STRATEGY
# ==========================================

def determine_target_price(state: AgentState) -> tuple[float, List[str]]:
    """
    Determine optimal selling price
    
    Strategy:
    1. If market data exists → Price competitively
    2. Ensure minimum margin (2.5x landed cost)
    3. Adjust for trend/sentiment
    
    Returns:
        (target_price, reasoning_list)
    """
    
    print(f"      💵 Determining target selling price...")
    
    reasoning = []
    
    # Get landed cost from logistics
    if not state.logistics_data:
        print(f"         ❌ No logistics data - cannot determine pricing")
        return 0.0, ["ERROR: No logistics data available"]
    
    landed_cost = state.logistics_data.total_landed_cost_per_unit
    
    reasoning.append(f"Landed cost: ${landed_cost:.2f}")
    
    # ==========================================
    # COMPETITIVE PRICING
    # ==========================================
    
    if state.market_research_data:
        competitors = state.market_research_data.competitors
        valid_prices = [c.price for c in competitors if c.price and c.price > 0]
        
        if valid_prices:
            avg_market_price = sum(valid_prices) / len(valid_prices)
            max_market_price = max(valid_prices)
            min_market_price = min(valid_prices)
            
            print(f"         Market range: ${min_market_price:.2f} - ${max_market_price:.2f}")
            print(f"         Market average: ${avg_market_price:.2f}")
            
            # Price slightly below average to be competitive
            competitive_price = avg_market_price * 0.95
            
            reasoning.append(f"Market average: ${avg_market_price:.2f}")
            reasoning.append(f"Competitive price (5% below avg): ${competitive_price:.2f}")
            
            target_price = competitive_price
        else:
            reasoning.append("No competitor prices found → Using markup strategy")
            target_price = landed_cost * 3.5  # Default 3.5x markup
    else:
        reasoning.append("No market data → Using standard 3.5x markup")
        target_price = landed_cost * 3.5
    
    # ==========================================
    # MINIMUM MARGIN GUARANTEE
    # ==========================================
    
    # Ensure at least 2.5x markup
    min_acceptable_price = landed_cost * 2.5
    
    if target_price < min_acceptable_price:
        reasoning.append(f"Price too low ({target_price:.2f}) → Adjusting to minimum {min_acceptable_price:.2f}")
        target_price = min_acceptable_price
    
    # ==========================================
    # TREND ADJUSTMENT
    # ==========================================
    
    if state.trend_analysis_data:
        from state import TrendDirection
        
        trend = state.trend_analysis_data.overall_market_direction
        
        if trend == TrendDirection.RISING:
            # Can charge 10% premium on rising trends
            trend_adjusted = target_price * 1.10
            reasoning.append(f"Rising trend → +10% premium (${trend_adjusted:.2f})")
            target_price = trend_adjusted
        elif trend == TrendDirection.FALLING:
            # Need to be more aggressive on price
            trend_adjusted = target_price * 0.95
            reasoning.append(f"Falling trend → -5% to stay competitive (${trend_adjusted:.2f})")
            target_price = trend_adjusted
    
    # ==========================================
    # SENTIMENT ADJUSTMENT
    # ==========================================
    
    if state.social_media_intel:
        sentiment = state.social_media_intel.overall_sentiment
        
        if sentiment == SentimentType.POSITIVE:
            # Positive buzz = can charge slightly more
            sentiment_adjusted = target_price * 1.05
            reasoning.append(f"Positive sentiment → +5% (${sentiment_adjusted:.2f})")
            target_price = sentiment_adjusted
        elif sentiment == SentimentType.NEGATIVE:
            # Need to compete on price
            sentiment_adjusted = target_price * 0.97
            reasoning.append(f"Negative sentiment → -3% (${sentiment_adjusted:.2f})")
            target_price = sentiment_adjusted
    
    print(f"         Final price: ${target_price:.2f}")
    
    reasoning.append(f"FINAL TARGET PRICE: ${target_price:.2f}")
    
    return round(target_price, 2), reasoning


# ==========================================
# MAIN FINANCIAL MODELING AGENT
# ==========================================

def run_financial_modeling(state: AgentState) -> AgentState:
    """
    Main Financial Modeling Agent
    
    Uses data from:
    - Logistics Agent (landed cost)
    - Market Research (competitive pricing)
    - Social Media Intel (CPA refinement)
    - Trend Analysis (pricing/CPA adjustments)
    
    Calculates:
    - Target selling price
    - Marketing CPA
    - Profit margins
    - Break-even point
    - ROI projections
    - Go/No-Go recommendation
    """
    
    print(f"\n💰 AGENT: Starting Financial Modeling...")
    
    # ==========================================
    # VALIDATION
    # ==========================================
    
    required_missing = []
    
    if not state.market_research_data:
        required_missing.append("market_research_data")
    
    if not state.supplier_data:
        required_missing.append("supplier_data")
    
    if not state.logistics_data:
        required_missing.append("logistics_data")
    
    if required_missing:
        error_msg = f"Missing required data: {', '.join(required_missing)}"
        print(f"   ❌ {error_msg}")
        
        if not state.error_message:
            state.error_message = {}
        state.error_message["financial_modeling"] = error_msg
        return state
    
    product_name = state.parsed_query.product_name or "Unknown Product"
    category = state.parsed_query.category or "General"
    
    print(f"   📊 Analyzing: {product_name} ({category})")
    
    # ==========================================
    # GET KEY NUMBERS
    # ==========================================
    
    # Landed cost (from Logistics Agent)
    landed_cost = state.logistics_data.total_landed_cost_per_unit
    
    print(f"\n   💰 Base Costs:")
    print(f"      Landed Cost: ${landed_cost:.2f}")
    print(f"      (Product + Shipping + Customs + Packaging)")
    
    # ==========================================
    # PHASE 1: PRICING STRATEGY
    # ==========================================
    
    print(f"\n   💵 Pricing Strategy:")
    
    target_price, pricing_reasoning = determine_target_price(state)
    
    for reason in pricing_reasoning:
        print(f"      • {reason}")
    
    # ==========================================
    # PHASE 2: MARKETING CPA
    # ==========================================
    
    print(f"\n   📢 Marketing Costs:")
    
    cpa, cpa_reasoning = estimate_marketing_cpa(state, target_price)
    
    for reason in cpa_reasoning:
        print(f"      • {reason}")
    
    # ==========================================
    # PHASE 3: PROFIT CALCULATIONS
    # ==========================================
    
    print(f"\n   📊 Profit Analysis:")
    
    # Gross Profit = Revenue - COGS
    gross_profit = target_price - landed_cost
    gross_margin_pct = (gross_profit / target_price) * 100
    
    print(f"      Revenue:        ${target_price:.2f}")
    print(f"      - Landed Cost:  ${landed_cost:.2f}")
    print(f"      ─────────────────────────")
    print(f"      Gross Profit:   ${gross_profit:.2f} ({gross_margin_pct:.1f}%)")
    
    # Net Profit = Gross Profit - Marketing
    net_profit = gross_profit - cpa
    net_margin_pct = (net_profit / target_price) * 100
    
    print(f"      - Marketing:    ${cpa:.2f}")
    print(f"      ─────────────────────────")
    print(f"      Net Profit:     ${net_profit:.2f} ({net_margin_pct:.1f}%)")
    
    # ==========================================
    # PHASE 4: BREAK-EVEN ANALYSIS
    # ==========================================
    
    print(f"\n   🎯 Break-Even Analysis:")
    
    # Fixed costs for starting an e-commerce business
    fixed_costs = {
        "LLC Formation": 300,
        "Website (Shopify/WooCommerce)": 50,
        "Product Photography": 200,
        "Initial Inventory (Sample)": 500,
        "Ad Testing Budget": 500,
        "Misc (Domain, Apps, etc.)": 250,
    }
    
    total_fixed_costs = sum(fixed_costs.values())
    
    print(f"      Fixed Costs:")
    for item, cost in fixed_costs.items():
        print(f"         {item}: ${cost}")
    print(f"      Total: ${total_fixed_costs}")
    
    if net_profit > 0:
        break_even_units = int(total_fixed_costs / net_profit) + 1
    else:
        break_even_units = 999999
    
    print(f"\n      Break-Even: {break_even_units} units")
    print(f"      (${break_even_units * target_price:.2f} in revenue)")
    
    # ==========================================
    # PHASE 5: ROI PROJECTIONS
    # ==========================================
    
    print(f"\n   📈 Revenue Projections:")
    
    # Monthly sales velocity (conservative estimate)
    monthly_units_conservative = 50
    monthly_units_moderate = 100
    monthly_units_optimistic = 200
    
    scenarios = [
        ("Conservative", monthly_units_conservative),
        ("Moderate", monthly_units_moderate),
        ("Optimistic", monthly_units_optimistic),
    ]
    
    for scenario_name, units in scenarios:
        revenue = target_price * units
        profit = net_profit * units
        print(f"      {scenario_name} ({units} units/mo):")
        print(f"         Revenue: ${revenue:.2f}/mo")
        print(f"         Profit:  ${profit:.2f}/mo")
    
    # Use moderate scenario for final metrics
    monthly_revenue = target_price * monthly_units_moderate
    monthly_profit = net_profit * monthly_units_moderate
    
    # ==========================================
    # PHASE 6: RECOMMENDATION
    # ==========================================
    
    print(f"\n   🚦 Recommendation:")
    
    if net_margin_pct >= 30:
        verdict = "GO 🟢 (Excellent Margins)"
        explanation = f"Net margin of {net_margin_pct:.1f}% is excellent. Strong profit potential with room for scaling."
        recommendation_emoji = "🚀"
    elif net_margin_pct >= 20:
        verdict = "GO 🟢 (Good Margins)"
        explanation = f"Net margin of {net_margin_pct:.1f}% is healthy. Solid business opportunity."
        recommendation_emoji = "✅"
    elif net_margin_pct >= 15:
        verdict = "CAUTION 🟡 (Acceptable Margins)"
        explanation = f"Net margin of {net_margin_pct:.1f}% is workable but tight. Watch costs and optimize marketing."
        recommendation_emoji = "⚠️"
    elif net_margin_pct >= 10:
        verdict = "CAUTION 🟡 (Tight Margins)"
        explanation = f"Net margin of {net_margin_pct:.1f}% is low. Consider negotiating better supplier pricing or reducing CPA."
        recommendation_emoji = "⚠️"
    else:
        verdict = "NO-GO 🔴 (Unprofitable)"
        explanation = f"Net margin of {net_margin_pct:.1f}% is too low. Not recommended unless costs can be reduced significantly."
        recommendation_emoji = "🛑"
    
    print(f"      {recommendation_emoji} {verdict}")
    print(f"      {explanation}")
    
    # ==========================================
    # PHASE 7: COMPILE RESULTS
    # ==========================================
    
    metrics = FinancialMetrics(
        gross_profit_per_unit=round(gross_profit, 2),
        net_profit_per_unit=round(net_profit, 2),
        margin_percentage=round(net_margin_pct, 2),
        break_even_units=break_even_units,
        monthly_revenue_potential=round(monthly_revenue, 2),
        recommendation=verdict
    )
    
    # Compile all assumptions
    assumptions = []
    
    # Add logistics assumptions
    if state.logistics_data:
        assumptions.extend(state.logistics_data.logistics_assumptions)
    
    # Add pricing reasoning
    assumptions.append("─── PRICING ───")
    assumptions.extend(pricing_reasoning)
    
    # Add CPA reasoning
    assumptions.append("─── MARKETING CPA ───")
    assumptions.extend(cpa_reasoning)
    
    # Add fixed costs
    assumptions.append("─── FIXED COSTS ───")
    for item, cost in fixed_costs.items():
        assumptions.append(f"{item}: ${cost}")
    
    # Add final verdict
    assumptions.append("─── VERDICT ───")
    assumptions.append(explanation)
    
    state.financial_data = FinancialModelingData(
        target_selling_price=round(target_price, 2),
        total_landed_cost=round(landed_cost, 2),
        marketing_cpa=round(cpa, 2),
        metrics=metrics,
        assumptions=assumptions
    )
    
    print(f"\n   ✅ Financial Modeling Complete!")
    
    return state