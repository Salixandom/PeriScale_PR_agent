"""
Product Research Agent System - Final Production State Schema
Optimized for Firecrawl scraping, PII masking, and Deterministic Financial Math.
"""

from pydantic import BaseModel, Field, field_validator, model_validator
from typing import List, Dict, Any, Optional, Literal
from datetime import datetime
from enum import Enum

# ==================== 1. ENUMS & CONSTANTS ====================

class RiskLevel(str, Enum):
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

class RecommendationType(str, Enum):
    LAUNCH_NOW = "LAUNCH_NOW"
    WAIT_30_DAYS = "WAIT_30_DAYS"
    WAIT_60_DAYS = "WAIT_60_DAYS"
    RECONSIDER = "RECONSIDER"
    NOT_RECOMMENDED = "NOT_RECOMMENDED"

class QueryIntent(str, Enum):
    RESEARCH = "research"
    ANALYSIS = "analysis"
    VALIDATION = "validation"
    COMPARISON = "comparison"
    TREND_FORECAST = "trend_forecast"

class ExecutionMode(str, Enum):
    FAST = "fast"          # Skip deep scraping, use cache/API
    BALANCED = "balanced"  # Standard flow
    THOROUGH = "thorough"  # Retry scraping, detailed analysis

class Currency(str, Enum):
    USD = "USD"
    EUR = "EUR"
    GBP = "GBP"
    CNY = "CNY"
    BDT = "BDT"

class AgentName(str, Enum):
    # Orchestration
    QUERY_PARSER = "query_parser"
    
    # Stage 1: Data Collection
    MARKET_RESEARCH = "market_research"
    TREND_ANALYSIS = "trend_analysis"
    SOCIAL_MEDIA_INTEL = "social_media_intel"
    HISTORICAL_DATABASE = "historical_database"
    
    # Stage 2: Deep Analysis
    COUNTRY_ANALYZER = "country_analyzer"
    COMPETITIVE_ANALYSIS = "competitive_analysis"
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    SEASONAL_ANALYSIS = "seasonal_analysis"
    
    # Stage 3: Sourcing
    SUPPLIER_SOURCING = "supplier_sourcing"
    LOGISTICS_SHIPPING = "logistics_shipping"
    
    # Stage 4: Strategy
    PRICING_STRATEGY = "pricing_strategy"
    MARKETING_STRATEGY = "marketing_strategy"
    SOCIAL_MEDIA_PROMOTION = "social_media_promotion"
    
    # Stage 5: Validation & Launch
    FINANCIAL_MODELING = "financial_modeling"
    RISK_ASSESSMENT = "risk_assessment"
    TIMING_OPTIMIZATION = "timing_optimization"
    BUSINESS_LAUNCH_GUIDE = "business_launch_guide"

# ==================== 2. SHARED DATA MODELS ====================

class ParsedProduct(BaseModel):
    """Structured product extraction"""
    name: str
    category: Optional[str] = None
    target_audience: Optional[str] = None
    keywords: List[str] = Field(default_factory=list)

class CompetitorData(BaseModel):
    """Firecrawl Output: Competitor Details"""
    name: str
    url: str
    price: float = Field(..., ge=0)
    currency: Currency = Field(default=Currency.USD)
    features: List[str] = Field(default_factory=list)
    rating: Optional[float] = None
    reviews_count: Optional[int] = None
    scraped_at: datetime = Field(default_factory=datetime.now)

class SupplierData(BaseModel):
    """Firecrawl Output: Supplier Details"""
    vendor_name: str
    platform: str = Field(..., description="Alibaba, AliExpress, etc.")
    profile_url: str
    product_url: Optional[str] = None
    unit_price: float = Field(..., ge=0)
    currency: Currency = Field(default=Currency.USD)
    moq: int = Field(..., ge=1, description="Minimum Order Quantity")
    lead_time_days: Optional[int] = None
    is_verified: bool = False
    quality_rating: Optional[float] = None

class TrendForecast(BaseModel):
    """Google Trends Forecast"""
    period: str
    demand_score: float = Field(..., ge=0, le=100)
    trend_direction: str
    confidence: float

class PeriScaleServiceOffer(BaseModel):
    """Specific PeriScale Agency Offers"""
    service_name: str
    tier: Literal["Starter", "Growth", "Enterprise"]
    estimated_price: float
    deliverables: List[str]

# ==================== 3. AGENT OUTPUT SCHEMAS ====================

class QueryParserOutput(BaseModel):
    parsed_product: ParsedProduct
    query_intent: QueryIntent
    needs_clarification: bool
    clarification_questions: List[str] = Field(default_factory=list)
    search_queries: List[str]

class MarketResearchOutput(BaseModel):
    competitors: List[CompetitorData]
    market_saturation_score: float = Field(..., ge=0, le=100)
    avg_price: float
    market_insights: str

class TrendAnalysisOutput(BaseModel):
    current_interest_score: int
    forecast_90d: TrendForecast
    seasonality_detected: bool
    top_related_queries: List[str]

class SocialMediaIntelOutput(BaseModel):
    platforms_analyzed: List[str]
    viral_hooks_found: List[str]
    meta_ads_summary: str
    influencer_mentions: int

class CountryAnalysisOutput(BaseModel):
    recommended_country: str
    market_size_estimate: float
    regulatory_notes: str

class SentimentAnalysisOutput(BaseModel):
    pain_points: List[str]
    delighters: List[str]
    overall_sentiment_score: float = Field(..., ge=-1, le=1)

class SupplierSourcingOutput(BaseModel):
    suppliers: List[SupplierData]
    recommended_supplier: SupplierData
    avg_unit_cost: float
    sourcing_notes: str

class LogisticsOutput(BaseModel):
    recommended_method: str
    estimated_shipping_cost: float
    customs_duty_percent: float
    total_landed_cost_estimate: float

class PricingStrategyOutput(BaseModel):
    recommended_selling_price: float
    currency: Currency
    profit_margin_target: float
    strategy_name: str = Field(..., description="e.g. 'Premium Skimming'")

class MarketingStrategyOutput(BaseModel):
    primary_channels: List[str]
    budget_allocation: Dict[str, float]
    campaign_hooks: List[str]

class SocialMediaPromotionOutput(BaseModel):
    content_calendar_summary: str
    periscale_services: List[PeriScaleServiceOffer] = Field(..., description="Agency upsell offers")

class FinancialModelingOutput(BaseModel):
    """Deterministic Financial Calculation Output"""
    # Inputs used for calculation (Traceability)
    used_unit_cost: float
    used_shipping_cost: float
    used_selling_price: float
    
    # Outputs
    gross_profit_per_unit: float
    gross_margin_percent: float
    break_even_units: int
    roi_projection_12m: float
    monthly_revenue_forecast: List[float]

class BusinessLaunchGuideOutput(BaseModel):
    """Final Aggregate Output"""
    executive_summary: str
    launch_checklist: List[str]
    timeline_weeks: int
    risk_mitigation_plan: str

# ==================== 4. MAIN STATE (NESTED & CLEAN) ====================

class ProductResearchState(BaseModel):
    """
    The Single Source of Truth.
    Now uses composition (nested models) instead of flat fields.
    """
    # --- Identity & Security ---
    query: str
    user_id: str
    session_id: str
    timestamp: datetime = Field(default_factory=datetime.now)
    
    # PII Masking
    is_pii_detected: bool = False
    pii_redacted_map: Dict[str, str] = Field(default_factory=dict)
    
    # Flow Control
    execution_mode: ExecutionMode = ExecutionMode.BALANCED
    current_stage: str = "init"
    errors: List[str] = Field(default_factory=list)
    
    # --- STAGE 1: Data Collection ---
    query_parser: Optional[QueryParserOutput] = None
    market_research: Optional[MarketResearchOutput] = None
    trend_analysis: Optional[TrendAnalysisOutput] = None
    social_intel: Optional[SocialMediaIntelOutput] = None
    
    # --- STAGE 2: Analysis ---
    country_analysis: Optional[CountryAnalysisOutput] = None
    sentiment_analysis: Optional[SentimentAnalysisOutput] = None
    # (Optional: Seasonal & Competitive can share models if simple)
    
    # --- STAGE 3: Sourcing (Hard Numbers) ---
    supplier_sourcing: Optional[SupplierSourcingOutput] = None
    logistics: Optional[LogisticsOutput] = None
    
    # --- STAGE 4: Strategy ---
    pricing_strategy: Optional[PricingStrategyOutput] = None
    marketing_strategy: Optional[MarketingStrategyOutput] = None
    social_promotion: Optional[SocialMediaPromotionOutput] = None
    
    # --- STAGE 5: Validation & Final ---
    financial_model: Optional[FinancialModelingOutput] = None
    launch_guide: Optional[BusinessLaunchGuideOutput] = None

# ==================== 5. HELPER FUNCTIONS ====================

def create_initial_state(query: str, user_id: str, session_id: str) -> ProductResearchState:
    return ProductResearchState(
        query=query,
        user_id=user_id,
        session_id=session_id
    )

AGENT_GROUPS = {
    "stage_1_parallel": [
        AgentName.MARKET_RESEARCH,
        AgentName.TREND_ANALYSIS,
        AgentName.SOCIAL_MEDIA_INTEL
    ],
    "stage_3_parallel": [
        AgentName.SUPPLIER_SOURCING,
        AgentName.LOGISTICS_SHIPPING
    ],
    "stage_5_sequential": [
        AgentName.FINANCIAL_MODELING, # Depends on Stage 3 & 4
        AgentName.RISK_ASSESSMENT,
        AgentName.BUSINESS_LAUNCH_GUIDE
    ]
}