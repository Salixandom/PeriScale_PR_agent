"""
Product Research Agent System - Complete State & Output Models
Optimized for Firecrawl/Anycrawl scraping + deterministic calculations
"""

from pydantic import BaseModel, Field, field_validator, model_validator
from typing import List, Dict, Any, Optional, Literal
from datetime import datetime
from enum import Enum

# ==================== ENUMS ====================

class RiskLevel(str, Enum):
    """Risk level categories"""
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

class RecommendationType(str, Enum):
    """Final recommendation types"""
    LAUNCH_NOW = "LAUNCH_NOW"
    WAIT_30_DAYS = "WAIT_30_DAYS"
    WAIT_60_DAYS = "WAIT_60_DAYS"
    WAIT_90_DAYS = "WAIT_90_DAYS"
    RECONSIDER = "RECONSIDER"
    NOT_RECOMMENDED = "NOT_RECOMMENDED"

class QueryIntent(str, Enum):
    """User query intent classification"""
    RESEARCH = "research"
    ANALYSIS = "analysis"
    VALIDATION = "validation"
    COMPARISON = "comparison"
    TREND_FORECAST = "trend_forecast"

class QueryComplexity(str, Enum):
    """Query complexity levels"""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"

class TrendDirection(str, Enum):
    """Market trend directions"""
    RISING = "rising"
    STABLE = "stable"
    DECLINING = "declining"
    VOLATILE = "volatile"

class ExecutionMode(str, Enum):
    """Agent execution modes"""
    FAST = "fast"
    BALANCED = "balanced"
    THOROUGH = "thorough"

class AgentName(str, Enum):
    """All available agent names"""
    # Core preprocessing
    QUERY_PARSER = "query_parser"
    
    # Data collection (mostly scraping/API - minimal LLM)
    MARKET_RESEARCH = "market_research"
    TREND_ANALYSIS = "trend_analysis"
    SOCIAL_MEDIA_INTEL = "social_media_intel"
    SUPPLIER_SOURCING = "supplier_sourcing"
    LOGISTICS_SHIPPING = "logistics_shipping"
    
    # Analysis (LLM-heavy)
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    COMPETITIVE_ANALYSIS = "competitive_analysis"
    COUNTRY_ANALYZER = "country_analyzer"
    
    # Planning (Mix of calculations + LLM)
    SEASONAL_ANALYSIS = "seasonal_analysis"
    HISTORICAL_DATABASE = "historical_database"
    FINANCIAL_MODELING = "financial_modeling"  # Mostly calculations
    RISK_ASSESSMENT = "risk_assessment"
    TIMING_OPTIMIZATION = "timing_optimization"
    
    # Strategy (LLM-heavy)
    MARKETING_STRATEGY = "marketing_strategy"
    SOCIAL_MEDIA_PROMOTION = "social_media_promotion"
    PRICING_STRATEGY = "pricing_strategy"
    BUSINESS_LAUNCH_GUIDE = "business_launch_guide"

# ==================== HELPER/NESTED MODELS ====================

class CompetitorData(BaseModel):
    """Individual competitor from scraping (Firecrawl output)"""
    name: str = Field(..., description="Product/company name")
    price: Optional[float] = Field(None, ge=0, description="Price in USD")
    original_price: Optional[float] = Field(None, ge=0, description="Original price if discounted")
    currency: str = Field(default="USD", description="Currency code")
    rating: Optional[float] = Field(None, ge=0, le=5, description="Rating 0-5")
    reviews_count: Optional[int] = Field(None, ge=0, description="Review count")
    features: List[str] = Field(default_factory=list, description="Product features")
    seller: Optional[str] = Field(None, description="Seller name")
    url: str = Field(..., description="Product URL")
    platform: str = Field(..., description="Platform (Amazon, Alibaba, etc.)")
    stock_status: Optional[str] = Field(None, description="In stock/out of stock")
    shipping_info: Optional[str] = Field(None, description="Shipping details")
    image_url: Optional[str] = Field(None, description="Product image URL")
    scraped_at: datetime = Field(default_factory=datetime.now, description="Scrape timestamp")

class SupplierData(BaseModel):
    """Supplier information from Alibaba/AliExpress (Firecrawl output)"""
    name: str = Field(..., description="Supplier name")
    company: Optional[str] = Field(None, description="Company name")
    platform: str = Field(..., description="Alibaba/AliExpress/DHgate/etc")
    url: str = Field(..., description="Supplier profile URL")
    product_url: Optional[str] = Field(None, description="Specific product URL")
    unit_price: float = Field(..., ge=0, description="Price per unit USD")
    min_order_quantity: int = Field(..., ge=1, description="Minimum order quantity")
    max_order_quantity: Optional[int] = Field(None, description="Maximum order quantity")
    bulk_pricing: Optional[List[Dict[str, Any]]] = Field(None, description="Bulk discounts")
    shipping_time: Optional[str] = Field(None, description="Shipping duration")
    shipping_cost: Optional[float] = Field(None, ge=0, description="Shipping cost estimate")
    rating: Optional[float] = Field(None, ge=0, le=5, description="Supplier rating")
    reviews_count: Optional[int] = Field(None, ge=0, description="Review count")
    response_rate: Optional[float] = Field(None, ge=0, le=100, description="Response rate %")
    on_time_delivery: Optional[float] = Field(None, ge=0, le=100, description="On-time delivery %")
    verified: bool = Field(default=False, description="Verified supplier")
    gold_supplier: bool = Field(default=False, description="Gold/Premium supplier")
    years_in_business: Optional[int] = Field(None, ge=0, description="Years in business")
    location: Optional[str] = Field(None, description="Supplier location")
    certifications: List[str] = Field(default_factory=list, description="Certifications")
    payment_terms: List[str] = Field(default_factory=list, description="Payment methods")
    sample_available: bool = Field(default=False, description="Sample available")
    sample_price: Optional[float] = Field(None, ge=0, description="Sample price")
    customization_available: bool = Field(default=False, description="Customization offered")

class ParsedProduct(BaseModel):
    """Structured product from query parsing"""
    name: str = Field(..., description="Product name")
    category: Optional[str] = Field(None, description="Product category")
    subcategory: Optional[str] = Field(None, description="Product subcategory")
    price_range_min: Optional[float] = Field(None, ge=0, description="Min expected price")
    price_range_max: Optional[float] = Field(None, ge=0, description="Max expected price")
    target_market: Optional[str] = Field(None, description="Target market/country")
    target_audience: Optional[str] = Field(None, description="Target demographic")
    keywords: List[str] = Field(default_factory=list, description="Search keywords")
    attributes: Dict[str, Any] = Field(default_factory=dict, description="Product attributes")
    use_case: Optional[str] = Field(None, description="Primary use case")

class TrendForecast(BaseModel):
    """Trend forecast from Google Trends API + calculations"""
    period: str = Field(..., description="Forecast period (30/60/90 days)")
    demand_score: float = Field(..., ge=0, le=100, description="Demand score 0-100")
    demand_change: float = Field(..., description="Change vs current (%)")
    price_trend: float = Field(..., description="Price trend multiplier")
    competition_level: float = Field(..., ge=0, le=100, description="Competition 0-100")
    search_volume_trend: str = Field(..., description="Trend direction")
    confidence: float = Field(..., ge=0, le=1, description="Forecast confidence")
    key_factors: List[str] = Field(default_factory=list, description="Influencing factors")

class RevenueProjection(BaseModel):
    """Revenue projection from financial calculations"""
    period: str = Field(..., description="Period (month_1, month_2, etc.)")
    period_number: int = Field(..., ge=1, description="Period number")
    units_sold: int = Field(..., ge=0, description="Units sold")
    unit_price: float = Field(..., ge=0, description="Selling price per unit")
    revenue: float = Field(..., ge=0, description="Total revenue")
    cogs: float = Field(..., ge=0, description="Cost of goods sold")
    gross_profit: float = Field(..., description="Gross profit")
    marketing_cost: float = Field(..., ge=0, description="Marketing expenses")
    operating_cost: float = Field(..., ge=0, description="Operating expenses")
    net_profit: float = Field(..., description="Net profit")
    cumulative_profit: float = Field(..., description="Cumulative profit to date")
    roi: float = Field(..., description="Return on investment %")

class RiskAssessmentDetail(BaseModel):
    """Individual risk assessment"""
    category: str = Field(..., description="Risk category")
    score: float = Field(..., ge=0, le=1, description="Risk score 0-1")
    level: RiskLevel = Field(..., description="Risk level")
    description: str = Field(..., description="Risk description")
    probability: float = Field(..., ge=0, le=1, description="Probability 0-1")
    impact: float = Field(..., ge=0, le=1, description="Impact if occurs 0-1")
    mitigation: str = Field(..., description="Mitigation strategy")
    triggers: List[str] = Field(default_factory=list, description="Risk triggers")

class MarketingChannel(BaseModel):
    """Marketing channel recommendation"""
    channel: str = Field(..., description="Channel name")
    budget_allocation: float = Field(..., ge=0, le=100, description="Budget % allocation")
    expected_roi: float = Field(..., description="Expected ROI %")
    estimated_cpa: Optional[float] = Field(None, ge=0, description="Cost per acquisition")
    estimated_reach: Optional[int] = Field(None, ge=0, description="Estimated reach")
    target_audience: str = Field(..., description="Target audience")
    content_types: List[str] = Field(..., description="Content types")
    posting_frequency: Optional[str] = Field(None, description="Posting frequency")
    difficulty: str = Field(..., description="easy/medium/hard")
    time_to_roi: Optional[str] = Field(None, description="Time to see ROI")

class LaunchPhase(BaseModel):
    """Business launch phase"""
    phase_number: int = Field(..., ge=1, description="Phase number")
    phase_name: str = Field(..., description="Phase name")
    duration_days: int = Field(..., ge=1, description="Duration in days")
    start_day: int = Field(..., ge=0, description="Start day relative to launch")
    tasks: List[str] = Field(..., description="Tasks to complete")
    milestones: List[str] = Field(..., description="Key milestones")
    dependencies: List[str] = Field(default_factory=list, description="Dependencies on other phases")
    estimated_cost: Optional[float] = Field(None, ge=0, description="Phase cost")
    success_criteria: List[str] = Field(default_factory=list, description="Success metrics")

class CountryAnalysis(BaseModel):
    """Country-specific market analysis"""
    country: str = Field(..., description="Country name")
    country_code: str = Field(..., description="ISO country code")
    market_size_usd: Optional[float] = Field(None, ge=0, description="Market size in USD")
    population: Optional[int] = Field(None, ge=0, description="Population")
    gdp_per_capita: Optional[float] = Field(None, ge=0, description="GDP per capita")
    internet_penetration: Optional[float] = Field(None, ge=0, le=100, description="Internet %")
    ecommerce_penetration: Optional[float] = Field(None, ge=0, le=100, description="E-commerce %")
    demand_score: float = Field(..., ge=0, le=100, description="Demand score")
    competition_score: float = Field(..., ge=0, le=100, description="Competition score")
    market_opportunity_score: float = Field(..., ge=0, le=100, description="Opportunity score")
    entry_barriers: List[str] = Field(default_factory=list, description="Entry barriers")
    advantages: List[str] = Field(default_factory=list, description="Market advantages")
    recommended: bool = Field(..., description="Recommended for entry")
    entry_difficulty: str = Field(..., description="easy/medium/hard")
    estimated_timeline: Optional[str] = Field(None, description="Entry timeline")

class SentimentScore(BaseModel):
    """Sentiment analysis by aspect"""
    aspect: str = Field(..., description="Product aspect")
    score: float = Field(..., ge=-1, le=1, description="Sentiment -1 to 1")
    positive_mentions: int = Field(..., ge=0, description="Positive count")
    negative_mentions: int = Field(..., ge=0, description="Negative count")
    neutral_mentions: int = Field(..., ge=0, description="Neutral count")
    confidence: float = Field(..., ge=0, le=1, description="Confidence")
    top_positive_keywords: List[str] = Field(default_factory=list)
    top_negative_keywords: List[str] = Field(default_factory=list)

class CustomerReview(BaseModel):
    """Customer review from scraping"""
    platform: str = Field(..., description="Platform")
    product_name: str = Field(..., description="Product")
    rating: float = Field(..., ge=0, le=5, description="Rating")
    title: Optional[str] = Field(None, description="Review title")
    text: str = Field(..., description="Review text")
    date: Optional[str] = Field(None, description="Review date")
    verified_purchase: bool = Field(default=False)
    helpful_count: Optional[int] = Field(None, ge=0)
    url: Optional[str] = Field(None, description="Review URL")

class LogisticsOption(BaseModel):
    """Shipping/logistics option"""
    provider: str = Field(..., description="Provider name")
    method: str = Field(..., description="Shipping method")
    origin: str = Field(..., description="Origin country")
    destination: str = Field(..., description="Destination country")
    cost_per_unit: float = Field(..., ge=0, description="Cost per unit USD")
    transit_days_min: int = Field(..., ge=1, description="Min transit days")
    transit_days_max: int = Field(..., ge=1, description="Max transit days")
    tracking: bool = Field(..., description="Tracking available")
    insurance: bool = Field(..., description="Insurance available")
    customs_handling: bool = Field(..., description="Customs handling included")
    reliability_score: Optional[float] = Field(None, ge=0, le=5, description="Reliability")

# ==================== AGENT OUTPUT SCHEMAS ====================

class QueryParserOutput(BaseModel):
    """
    Output from Query Parser Agent
    LLM: YES (for parsing natural language query)
    """
    parsed_product: ParsedProduct
    query_intent: QueryIntent
    query_complexity: QueryComplexity
    clarity_score: float = Field(..., ge=0, le=1)
    clarity_issues: List[Dict[str, str]] = Field(default_factory=list)
    needs_clarification: bool
    search_queries: List[str] = Field(..., description="Generated search queries")
    suggested_refinements: Optional[List[str]] = None

class MarketResearchOutput(BaseModel):
    """
    Output from Market Research Agent
    LLM: MINIMAL (only for insights summary)
    Data: Firecrawl scraping + Tavily search
    """
    competitors: List[CompetitorData]
    total_competitors_found: int = Field(..., ge=0)
    avg_price: float = Field(..., ge=0)
    min_price: float = Field(..., ge=0)
    max_price: float = Field(..., ge=0)
    price_std_dev: float = Field(..., ge=0)
    avg_rating: Optional[float] = Field(None, ge=0, le=5)
    market_saturation_score: float = Field(..., ge=0, le=100, description="Calculated from competitor count")
    top_sellers: List[str] = Field(default_factory=list)
    market_insights: str = Field(..., description="LLM-generated summary")
    data_sources: List[str] = Field(..., description="URLs scraped")

class TrendAnalysisOutput(BaseModel):
    """
    Output from Trend Analysis Agent
    LLM: NO (pure Google Trends API + calculations)
    """
    trend_direction: TrendDirection
    trend_strength: float = Field(..., ge=0, le=1, description="From Google Trends data")
    current_interest: int = Field(..., ge=0, le=100, description="Current Google Trends score")
    peak_interest: int = Field(..., ge=0, le=100, description="Peak interest in period")
    average_interest: float = Field(..., ge=0, le=100, description="Average interest")
    yoy_growth: float = Field(..., description="Year-over-year growth %")
    mom_growth: float = Field(..., description="Month-over-month growth %")
    forecast_30_days: TrendForecast
    forecast_60_days: TrendForecast
    forecast_90_days: TrendForecast
    related_queries: List[str] = Field(default_factory=list, description="From Google Trends")
    rising_queries: List[str] = Field(default_factory=list, description="From Google Trends")
    regional_interest: Dict[str, int] = Field(default_factory=dict, description="Interest by region")
    seasonality_detected: bool = Field(..., description="Seasonal pattern detected")

class SocialMediaIntelOutput(BaseModel):
    """
    Output from Social Media Intelligence Agent
    LLM: MINIMAL (only for summary)
    Data: Firecrawl scraping of social platforms
    """
    platforms_analyzed: List[str]
    total_mentions: int = Field(..., ge=0)
    total_engagement: int = Field(..., ge=0)
    meta_ads_count: int = Field(..., ge=0, description="Facebook/Instagram ads found")
    meta_ads_data: List[Dict[str, Any]] = Field(default_factory=list)
    tiktok_hashtag_views: Optional[int] = Field(None, ge=0)
    tiktok_trending: bool = Field(default=False)
    reddit_discussions: int = Field(default=0, ge=0)
    reddit_sentiment: Optional[str] = Field(None)
    youtube_videos_count: int = Field(default=0, ge=0)
    influencer_mentions: List[Dict[str, Any]] = Field(default_factory=list)
    social_buzz_score: float = Field(..., ge=0, le=100, description="Calculated from engagement")
    viral_potential: float = Field(..., ge=0, le=100, description="Calculated score")
    summary: str = Field(..., description="LLM summary")

class SentimentAnalysisOutput(BaseModel):
    """
    Output from Sentiment Analysis Agent
    LLM: YES (for extracting pain points and analyzing text)
    """
    customer_reviews: List[CustomerReview]
    total_reviews_analyzed: int = Field(..., ge=0)
    aggregated_sentiment: List[SentimentScore]
    overall_sentiment: float = Field(..., ge=-1, le=1)
    average_rating: float = Field(..., ge=0, le=5)
    pain_points: List[str]
    desired_features: List[str]
    common_complaints: List[str]
    common_praises: List[str]
    review_summary: str
    confidence: float = Field(..., ge=0, le=1)

class CompetitiveAnalysisOutput(BaseModel):
    """
    Output from Competitive Analysis Agent
    LLM: YES (for strategic analysis)
    """
    competitors_analyzed: int = Field(..., ge=0)
    competitive_gaps: List[str]
    competitive_advantages: List[str]
    competitive_threats: List[str]
    market_positioning: str
    differentiation_opportunities: List[str]
    swot_analysis: Dict[str, List[str]] = Field(
        ...,
        description="Strengths, Weaknesses, Opportunities, Threats"
    )
    competitive_summary: str

class SupplierSourcingOutput(BaseModel):
    """
    Output from Supplier Sourcing Agent
    LLM: MINIMAL (only for recommendation reasoning)
    Data: Firecrawl scraping of Alibaba/AliExpress
    """
    suppliers: List[SupplierData]
    total_suppliers_found: int = Field(..., ge=0)
    recommended_supplier: Optional[SupplierData]
    avg_unit_price: float = Field(..., ge=0)
    min_unit_price: float = Field(..., ge=0)
    max_unit_price: float = Field(..., ge=0)
    avg_moq: float = Field(..., ge=0)
    avg_shipping_days: Optional[float] = Field(None, ge=0)
    verified_suppliers_count: int = Field(..., ge=0)
    gold_suppliers_count: int = Field(..., ge=0)
    sourcing_insights: str = Field(..., description="LLM insights")
    quality_score: float = Field(..., ge=0, le=100, description="Calculated from ratings")
    negotiation_tips: List[str] = Field(default_factory=list)

class LogisticsShippingOutput(BaseModel):
    """
    Output from Logistics & Shipping Agent
    LLM: MINIMAL
    Data: API calls to shipping providers or scraped data
    """
    options: List[LogisticsOption]
    recommended_option: Optional[LogisticsOption]
    total_options_analyzed: int = Field(..., ge=0)
    avg_cost_per_unit: float = Field(..., ge=0)
    avg_transit_days: float = Field(..., ge=0)
    fastest_option: Optional[LogisticsOption]
    cheapest_option: Optional[LogisticsOption]
    customs_info: Optional[Dict[str, Any]] = None
    import_duties_estimate: Optional[float] = Field(None, ge=0)
    fulfillment_strategy: str

class CountryAnalyzerOutput(BaseModel):
    """
    Output from Country Analyzer Agent
    LLM: YES (for strategic insights)
    Data: Mix of API data + analysis
    """
    countries_analyzed: List[CountryAnalysis]
    total_countries: int = Field(..., ge=0)
    top_markets: List[str] = Field(..., description="Top 3-5 countries")
    recommended_entry_sequence: List[str] = Field(..., description="Entry order")
    market_entry_strategy: str
    localization_requirements: List[str] = Field(default_factory=list)
    total_addressable_market: Optional[float] = Field(None, ge=0)

class SeasonalAnalysisOutput(BaseModel):
    """
    Output from Seasonal Analysis Agent
    LLM: MINIMAL
    Data: Historical data + calculations
    """
    monthly_scores: Dict[str, float] = Field(
        ...,
        description="Demand score for each month 0-100"
    )
    best_launch_months: List[str]
    worst_launch_months: List[str]
    current_month: str
    current_month_score: float = Field(..., ge=0, le=100)
    seasonal_multiplier: float = Field(..., gt=0)
    peak_season: str
    off_season: str
    seasonality_strength: float = Field(..., ge=0, le=1, description="How seasonal is it")
    recommendations: str

class HistoricalDatabaseOutput(BaseModel):
    """
    Output from Historical Database Agent
    LLM: YES (for insights from similar products)
    Data: Database query + LLM analysis
    """
    similar_products: List[Dict[str, Any]]
    total_similar_products: int = Field(..., ge=0)
    success_rate: float = Field(..., ge=0, le=1)
    failure_rate: float = Field(..., ge=0, le=1)
    avg_time_to_profitability: int = Field(..., ge=0, description="Days")
    avg_peak_revenue_month: int = Field(..., ge=0, description="Month number")
    success_factors: List[str]
    failure_factors: List[str]
    historical_insights: List[str]
    patterns: List[str]
    recommendations: str

class FinancialModelingOutput(BaseModel):
    """
    Output from Financial Modeling Agent
    LLM: NO (pure financial calculations)
    """
    revenue_projections: List[RevenueProjection]
    total_months_projected: int = Field(..., ge=1)
    break_even_month: int = Field(..., ge=0, description="Month when cumulative profit > 0")
    break_even_units: int = Field(..., ge=0, description="Units needed to break even")
    total_investment_required: float = Field(..., ge=0)
    projected_12m_revenue: float = Field(..., ge=0)
    projected_12m_profit: float = Field(...)
    roi_12m: float = Field(..., description="12-month ROI %")
    gross_margin: float = Field(..., ge=0, le=100, description="Gross margin %")
    net_margin: float = Field(..., description="Net margin %")
    payback_period_months: int = Field(..., ge=0)
    assumptions: List[str]
    sensitivity_analysis: Optional[Dict[str, Any]] = None

class RiskAssessmentOutput(BaseModel):
    """
    Output from Risk Assessment Agent
    LLM: YES (for risk identification and mitigation)
    """
    risk_categories: List[RiskAssessmentDetail]
    overall_risk_score: float = Field(..., ge=0, le=1)
    risk_level: RiskLevel
    top_risks: List[str] = Field(..., description="Top 5 risks")
    mitigation_strategies: List[str]
    contingency_plans: List[str]
    risk_summary: str
    recommended_actions: List[str]

class TimingOptimizationOutput(BaseModel):
    """
    Output from Timing Optimization Agent
    LLM: YES (for comprehensive analysis)
    Data: Combines all previous agent outputs
    """
    optimal_launch_date: str = Field(..., description="ISO format YYYY-MM-DD")
    optimal_launch_month: str
    timing_confidence: float = Field(..., ge=0, le=1)
    days_until_optimal: int = Field(..., ge=0)
    launch_windows: List[Dict[str, Any]] = Field(..., description="Alternative launch windows")
    market_readiness_score: float = Field(..., ge=0, le=100)
    competitive_timing_score: float = Field(..., ge=0, le=100)
    seasonal_timing_score: float = Field(..., ge=0, le=100)
    reasoning: str

class MarketingStrategyOutput(BaseModel):
    """
    Output from Marketing Strategy Agent
    LLM: YES (strategic planning)
    """
    channels: List[MarketingChannel]
    total_monthly_budget: float = Field(..., ge=0)
    budget_breakdown: Dict[str, float]
    customer_acquisition_cost_target: float = Field(..., ge=0)
    lifetime_value_estimate: float = Field(..., ge=0)
    expected_overall_roi: float = Field(..., description="Expected ROI %")
    campaign_ideas: List[str]
    content_themes: List[str]
    influencer_strategy: Optional[str] = None
    ugc_strategy: Optional[str] = None
    launch_timeline: List[Dict[str, str]] = Field(default_factory=list)

class SocialMediaPromotionOutput(BaseModel):
    """
    Output from Social Media Promotion Agent
    LLM: YES (content planning)
    """
    recommended_platforms: List[str]
    content_calendar_30d: List[Dict[str, str]]
    post_ideas: List[str]
    hashtag_strategy: List[str]
    posting_schedule: Dict[str, str] = Field(..., description="Platform -> frequency")
    engagement_tactics: List[str]
    ad_budget_allocation: Dict[str, float]
    expected_reach: Optional[int] = Field(None, ge=0)
    expected_engagement_rate: Optional[float] = Field(None, ge=0, le=100)
    periscale_services: Dict[str, Any] = Field(
        ...,
        description="PeriScale social media services offering"
    )

class PricingStrategyOutput(BaseModel):
    """
    Output from Pricing Strategy Agent
    LLM: YES (strategic pricing)
    Calculations: Based on costs + market data
    """
    recommended_price: float = Field(..., ge=0)
    price_range_min: float = Field(..., ge=0)
    price_range_max: float = Field(..., ge=0)
    competitor_price_comparison: Dict[str, float]
    pricing_strategy: str = Field(..., description="premium/competitive/penetration/skimming")
    psychological_price: float = Field(..., ge=0, description="e.g., 19.99 instead of 20")
    discount_strategy: Optional[str] = None
    bundle_opportunities: List[str] = Field(default_factory=list)
    pricing_tiers: Optional[List[Dict[str, Any]]] = None
    margin_analysis: Dict[str, float]

class BusinessLaunchGuideOutput(BaseModel):
    """
    Output from Business Launch Guide Agent
    LLM: YES (comprehensive guide generation)
    """
    launch_phases: List[LaunchPhase]
    total_timeline_days: int = Field(..., ge=0)
    pre_launch_checklist: List[str]
    legal_requirements: List[str]
    required_licenses: List[str] = Field(default_factory=list)
    platform_recommendations: List[Dict[str, str]] = Field(
        ...,
        description="E-commerce platforms, payment gateways, etc."
    )
    tools_needed: List[Dict[str, str]]
    team_requirements: List[Dict[str, str]] = Field(default_factory=list)
    initial_investment_breakdown: Dict[str, float]
    scaling_strategy: str
    expansion_roadmap: List[str]

# ==================== MAIN STATE MODEL ====================

class ProductResearchState(BaseModel):
    """
    Main state object that flows through all agents
    Each agent reads what it needs and writes its results
    """
    
    # ==================== INPUT ====================
    query: str = Field(..., description="User's product research query")
    user_id: str = Field(..., description="User ID")
    session_id: str = Field(..., description="Session ID")
    timestamp: datetime = Field(default_factory=datetime.now)
    execution_mode: ExecutionMode = Field(default=ExecutionMode.BALANCED)
    
    # ==================== SECURITY ====================
    blocked: bool = Field(default=False)
    block_reason: Optional[str] = None
    prompt_guard_passed: bool = Field(default=True)
    relevance_passed: bool = Field(default=True)
    pii_detected: bool = Field(default=False)
    pii_redacted: Optional[Dict[str, List[str]]] = None
    
    # ==================== MEMORY ====================
    cache_hit: bool = Field(default=False)
    cache_key: Optional[str] = None
    cached_response: Optional[Dict[str, Any]] = None
    related_products: Optional[List[str]] = None
    user_history: Optional[List[Dict[str, Any]]] = None
    
    # ==================== QUERY PREPROCESSING ====================
    needs_clarification: bool = Field(default=False)
    clarity_score: Optional[float] = None
    clarity_issues: Optional[List[Dict[str, str]]] = None
    parsed_product: Optional[ParsedProduct] = None
    query_intent: Optional[QueryIntent] = None
    query_complexity: Optional[QueryComplexity] = None
    search_queries: List[str] = Field(default_factory=list)
    
    # ==================== EXECUTION MANAGEMENT ====================
    selected_agents: List[AgentName] = Field(default_factory=list)
    execution_plan: Optional[Dict[str, Any]] = None
    parallel_groups: Optional[List[List[AgentName]]] = None
    current_phase: int = Field(default=0)
    agents_executed: List[AgentName] = Field(default_factory=list)
    agents_skipped: List[AgentName] = Field(default_factory=list)
    execution_path: List[str] = Field(default_factory=list)
    
    # ==================== MARKET RESEARCH ====================
    competitors: List[CompetitorData] = Field(default_factory=list)
    total_competitors_found: Optional[int] = None
    market_saturation_score: Optional[float] = None
    avg_competitor_price: Optional[float] = None
    min_competitor_price: Optional[float] = None
    max_competitor_price: Optional[float] = None
    avg_competitor_rating: Optional[float] = None
    market_insights: Optional[str] = None
    
    # ==================== TREND ANALYSIS ====================
    trend_direction: Optional[TrendDirection] = None
    trend_strength: Optional[float] = None
    current_interest: Optional[int] = None
    yoy_growth: Optional[float] = None
    mom_growth: Optional[float] = None
    forecast_30_days: Optional[TrendForecast] = None
    forecast_60_days: Optional[TrendForecast] = None
    forecast_90_days: Optional[TrendForecast] = None
    related_queries: List[str] = Field(default_factory=list)
    seasonality_detected: Optional[bool] = None
    
    # ==================== SOCIAL MEDIA ====================
    social_platforms_analyzed: List[str] = Field(default_factory=list)
    total_social_mentions: Optional[int] = None
    social_buzz_score: Optional[float] = None
    viral_potential: Optional[float] = None
    meta_ads_count: Optional[int] = None
    tiktok_trending: Optional[bool] = None
    influencer_mentions: List[Dict[str, Any]] = Field(default_factory=list)
    
    # ==================== SENTIMENT ====================
    customer_reviews: List[CustomerReview] = Field(default_factory=list)
    total_reviews_analyzed: Optional[int] = None
    overall_sentiment: Optional[float] = None
    average_customer_rating: Optional[float] = None
    pain_points: List[str] = Field(default_factory=list)
    desired_features: List[str] = Field(default_factory=list)
    common_complaints: List[str] = Field(default_factory=list)
    
    # ==================== COMPETITIVE ====================
    competitive_gaps: List[str] = Field(default_factory=list)
    competitive_advantages: List[str] = Field(default_factory=list)
    competitive_threats: List[str] = Field(default_factory=list)
    market_positioning: Optional[str] = None
    swot_analysis: Optional[Dict[str, List[str]]] = None
    
    # ==================== SUPPLIER SOURCING ====================
    suppliers: List[SupplierData] = Field(default_factory=list)
    total_suppliers_found: Optional[int] = None
    recommended_supplier: Optional[SupplierData] = None
    avg_supplier_price: Optional[float] = None
    min_supplier_price: Optional[float] = None
    max_supplier_price: Optional[float] = None
    avg_moq: Optional[float] = None
    
    # ==================== LOGISTICS ====================
    logistics_options: List[LogisticsOption] = Field(default_factory=list)
    recommended_logistics: Optional[LogisticsOption] = None
    avg_shipping_cost: Optional[float] = None
    avg_transit_days: Optional[float] = None
    import_duties_estimate: Optional[float] = None
    
    # ==================== COUNTRY ANALYSIS ====================
    countries_analyzed: List[CountryAnalysis] = Field(default_factory=list)
    top_markets: List[str] = Field(default_factory=list)
    market_entry_strategy: Optional[str] = None
    
    # ==================== SEASONAL ====================
    monthly_demand_scores: Optional[Dict[str, float]] = None
    best_launch_months: List[str] = Field(default_factory=list)
    worst_launch_months: List[str] = Field(default_factory=list)
    current_month_score: Optional[float] = None
    seasonal_multiplier: Optional[float] = None
    
    # ==================== HISTORICAL ====================
    similar_products: List[Dict[str, Any]] = Field(default_factory=list)
    success_rate: Optional[float] = None
    avg_time_to_profitability: Optional[int] = None
    historical_insights: List[str] = Field(default_factory=list)
    
    # ==================== FINANCIAL ====================
    revenue_projections: List[RevenueProjection] = Field(default_factory=list)
    break_even_month: Optional[int] = None
    break_even_units: Optional[int] = None
    total_investment_required: Optional[float] = None
    projected_12m_revenue: Optional[float] = None
    projected_12m_profit: Optional[float] = None
    roi_12m: Optional[float] = None
    gross_margin: Optional[float] = None
    
    # ==================== RISK ====================
    risk_categories: List[RiskAssessmentDetail] = Field(default_factory=list)
    overall_risk_score: Optional[float] = None
    risk_level: Optional[RiskLevel] = None
    top_risks: List[str] = Field(default_factory=list)
    mitigation_strategies: List[str] = Field(default_factory=list)
    
    # ==================== TIMING ====================
    optimal_launch_date: Optional[str] = None
    timing_confidence: Optional[float] = None
    days_until_optimal: Optional[int] = None
    market_readiness_score: Optional[float] = None
    
    # ==================== MARKETING ====================
    marketing_channels: List[MarketingChannel] = Field(default_factory=list)
    total_marketing_budget: Optional[float] = None
    expected_cac: Optional[float] = None
    expected_ltv: Optional[float] = None
    campaign_ideas: List[str] = Field(default_factory=list)
    
    # ==================== SOCIAL MEDIA PROMOTION ====================
    recommended_social_platforms: List[str] = Field(default_factory=list)
    content_calendar: List[Dict[str, str]] = Field(default_factory=list)
    hashtag_strategy: List[str] = Field(default_factory=list)
    periscale_services: Optional[Dict[str, Any]] = None
    
    # ==================== PRICING ====================
    recommended_price: Optional[float] = None
    price_range_min: Optional[float] = None
    price_range_max: Optional[float] = None
    pricing_strategy: Optional[str] = None
    
    # ==================== BUSINESS LAUNCH ====================
    launch_phases: List[LaunchPhase] = Field(default_factory=list)
    total_timeline_days: Optional[int] = None
    pre_launch_checklist: List[str] = Field(default_factory=list)
    required_tools: List[Dict[str, str]] = Field(default_factory=list)
    scaling_strategy: Optional[str] = None
    
    # ==================== SYNTHESIS ====================
    market_opportunity_score: Optional[float] = None
    confidence_score: Optional[float] = None
    recommendation: Optional[RecommendationType] = None
    reasoning: List[str] = Field(default_factory=list)
    action_items: List[str] = Field(default_factory=list)
    executive_summary: Optional[str] = None
    
    # ==================== OUTPUT ====================
    final_response: Optional[str] = None
    response_time: Optional[float] = None
    total_cost: Optional[float] = None
    
    # ==================== ERROR HANDLING ====================
    errors: List[Dict[str, str]] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    
    # ==================== METADATA ====================
    version: str = Field(default="1.0.0")
    debug_info: Optional[Dict[str, Any]] = None
    
    # ==================== VALIDATORS ====================
    
    @field_validator('clarity_score', 'overall_sentiment', 'trend_strength', 
                     'timing_confidence', 'success_rate', 'overall_risk_score')
    @classmethod
    def validate_ratio(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and not 0 <= v <= 1:
            raise ValueError('Score must be between 0 and 1')
        return v
    
    @field_validator('market_saturation_score', 'social_buzz_score', 'viral_potential',
                     'current_month_score', 'market_opportunity_score', 'confidence_score',
                     'market_readiness_score')
    @classmethod
    def validate_percentage(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and not 0 <= v <= 100:
            raise ValueError('Score must be between 0 and 100')
        return v
    
    @model_validator(mode='after')
    def validate_block_reason(self) -> 'ProductResearchState':
        if self.blocked and not self.block_reason:
            raise ValueError('block_reason required when blocked=True')
        return self
    
    model_config = {
        "use_enum_values": True,
        "validate_assignment": True,
        "arbitrary_types_allowed": True,
        "json_encoders": {
            datetime: lambda v: v.isoformat(),
        }
    }

# ==================== UTILITY FUNCTIONS ====================

def create_initial_state(
    query: str,
    user_id: str,
    session_id: str,
    execution_mode: ExecutionMode = ExecutionMode.BALANCED
) -> ProductResearchState:
    """Create initial state for new research request"""
    return ProductResearchState(
        query=query,
        user_id=user_id,
        session_id=session_id,
        execution_mode=execution_mode
    )

# ==================== AGENT LLM REQUIREMENTS ====================

AGENT_LLM_REQUIREMENTS = {
    # NO LLM NEEDED (Pure calculations/API data)
    "no_llm": [
        AgentName.TREND_ANALYSIS,      # Google Trends API + math
        AgentName.FINANCIAL_MODELING,  # Pure financial calculations
    ],
    
    # MINIMAL LLM (Just for summary/insights)
    "minimal_llm": [
        AgentName.MARKET_RESEARCH,     # Scraping + summary
        AgentName.SOCIAL_MEDIA_INTEL,  # Scraping + summary
        AgentName.SUPPLIER_SOURCING,   # Scraping + recommendation
        AgentName.LOGISTICS_SHIPPING,  # API data + summary
        AgentName.SEASONAL_ANALYSIS,   # Data + summary
    ],
    
    # HEAVY LLM (Strategic analysis)
    "heavy_llm": [
        AgentName.QUERY_PARSER,
        AgentName.SENTIMENT_ANALYSIS,
        AgentName.COMPETITIVE_ANALYSIS,
        AgentName.COUNTRY_ANALYZER,
        AgentName.HISTORICAL_DATABASE,
        AgentName.RISK_ASSESSMENT,
        AgentName.TIMING_OPTIMIZATION,
        AgentName.MARKETING_STRATEGY,
        AgentName.SOCIAL_MEDIA_PROMOTION,
        AgentName.PRICING_STRATEGY,
        AgentName.BUSINESS_LAUNCH_GUIDE,
    ]
}

if __name__ == "__main__":
    # Test state creation
    state = create_initial_state(
        query="Smart AI safety watch for women in Bangladesh",
        user_id="user_123",
        session_id="session_abc"
    )
    print(f"✅ State created: {state.query}")
    print(f"✅ Total agent types: {len(AgentName)}")
    print(f"✅ No LLM agents: {len(AGENT_LLM_REQUIREMENTS['no_llm'])}")
    print(f"✅ Minimal LLM agents: {len(AGENT_LLM_REQUIREMENTS['minimal_llm'])}")
    print(f"✅ Heavy LLM agents: {len(AGENT_LLM_REQUIREMENTS['heavy_llm'])}")