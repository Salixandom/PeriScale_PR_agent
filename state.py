from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from enum import Enum
from datetime import datetime

class QueryIntent(str, Enum):
    PRODUCT_DISCOVERY = "product_discovery"
    MARKET_VALIDATION = "market_validation"
    TREND_FORECAST = "trend_forecast"
    COMPETITOR_ANALYSIS = "competitor_analysis"
    SOURCING_LOGISTICS = "sourcing_logistics"
    FINANCIAL_ANALYSIS = "financial_analysis"
    MARKETING_STRATEGY = "marketing_strategy"
    LAUNCH_PLANNING = "launch_planning"

class Currency(str, Enum):
    USD = "USD"
    EUR = "EUR"
    GBP = "GBP"
    CNY = "CNY"
    BDT = "BDT"
    INR = "INR"
    JPY = "JPY"
    
class TrendDirection(str, Enum):
    RISING = "rising"
    FALLING = "falling"
    STABLE = "stable"
    SEASONAL = "seasonal"
    
class AgentName(str, Enum):
    QUERY_PARSER = "query_parser"
    MARKET_RESEARCHER = "market_researcher"
    TREND_ANALYSIS = "trend_analysis"
    SUPPLIER_SOURCING = "supplier_sourcing"
    FINANCIAL_MODELING = "financial_modeling"
    
class PlanStep(BaseModel):
    step_number: int = Field(description="Step number of the plan")
    agent_name: AgentName = Field(description="Name of the agent")
    instruction: str = Field(description="Specific goal for this step")
    reasoning: str = Field(description="Why this step is needed")
    status: str = Field(default="pending")
    
    
class ExecutionPlan(BaseModel):
    goal: str
    steps: List[PlanStep] = Field(description="Steps to achieve the goal")
    final_objective: str = Field(description="Final objective of the plan")
    

class CritiqueOutput(BaseModel):
    is_approved: bool = Field(description="True if the plan is approved by the critic")
    feedback: str = Field(description="Specific instructions on what to fix")
    
    
class ParsedQuery(BaseModel):
    """Information extracted from the user input/query"""
    query_intent: QueryIntent = Field(description="The main goal of the user")
    product_name: Optional[str] = Field(description="Name of the product")
    category: Optional[str] = Field(description="Category of the product")
    price: Optional[float] = Field(description="Price of the product")
    currency: Optional[str] = Field(description="Currency of the product")
    target_market: Optional[str] = Field(description="Target market/country for the product")
    search_keywords: List[str] = Field(description="Search keywords for the product for google/amazon")
    

class Review_Comment(BaseModel):
    """ Review or Comment extracted from the product page"""
    comment: str = Field(description="Comment or Review of the product")
    rating: Optional[float] = Field(description="Rating given in the review")


class CompetitorData(BaseModel):
    name: str = Field(description="Name of the competitor")
    url: str = Field(description="URL of the competitor website")
    price: Optional[float] = Field(default=None, description="Price of the product if visible")
    currency: Currency = Field(default=Currency.USD, description="Currency of the product")
    features: Optional[List[str]] = Field(default_factory=list, description="Key Features mentioned in the product")
    rating: Optional[float] = Field(default=None, description="Overall Rating of the product")
    Reviews_Comments: Optional[List[Review_Comment]] = Field(default_factory=list, description="Reviews and Comments of the product")
    scraped_at: datetime = Field(default_factory=datetime.now)
    

class MarketResearchData(BaseModel):
    competitors: List[CompetitorData] = Field(description="Top 5 competitors")
    market_size: Optional[float] = Field(description="Market size")
    market_summary: str = Field(description="Brief summary of the search results")
    

class TrendMetrics(BaseModel):
    """Single data point in the time series"""
    date: datetime = Field(description="Date of data point")
    interest_value: int = Field(ge=0, le=100, description="Interest 0-100")

class KeywordTrendAnalysis(BaseModel):
    """Analysis for a SINGLE keyword"""
    keyword: str = Field(description="Keyword")
    trend_direction: TrendDirection = Field(description="Trend direction (rising, falling, stable, seasonal)")
    average_interest: float = Field(description="Average interest over the last 12 months")
    peak_month: str = Field(description="Month with typically highest interest")
    interest_over_time: List[TrendMetrics] = Field(description="Weekly data points over 3 years")

class TrendAnalysisData(BaseModel):
    """Aggregated Output for ALL keywords"""
    target_country_iso: str = Field(description="Target Region Code (e.g. FR)")
    timeframe: str = Field(description="Date range used")
    keyword_trends: List[KeywordTrendAnalysis] = Field(description="List of analysis per keyword")
    overall_market_direction: TrendDirection = Field(description="Overall market direction")
    top_related_queries: List[str] = Field(description="Top 5 related rising search items")
    
class AgentState(BaseModel):
    """Shared memory for the system"""
    # INPUT
    user_raw_query: str
    
    # PLANNING STEP
    plan: Optional[ExecutionPlan] = None
    plan_feedback: List[str] = Field(default_factory=list)
    planner_loop_count: int = Field(default=0)
    is_plan_approved: bool = Field(default=False)
    
    # AGENT -> ParsedQuery
    parsed_query: Optional[ParsedQuery] = None
    
    # AGENT -> MarketResearchData
    market_research_data: Optional[MarketResearchData] = None
    
    # AGENT -> TrendAnalysisData
    trend_analysis_data: Optional[TrendAnalysisData] = None
    
    # System Info
    error_message: Optional[Dict[str, str]] = None