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
    summary: str = Field(description="""Give a detailed overview of the steps.
                            Tell it like you are talking to yourself of what you have to do. 
                            Why you need them and how will they work in which order and what problem they will solve. 
                            Give detailed plan for human understanding
                            Give the result in Markdown format""")
    

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
    
class SupplierData(BaseModel):
    supplier_name: str = Field(description="Name of the supplier")
    platform: str = Field(description="Platform (Alibaba, AliExpress, etc.)")
    product_url: str = Field(description="Link to the product")
    price_per_unit: float = Field(description="Estimated cost per unit")
    moq: int = Field(default=1, description="Minimum Order Quantity")
    rating: Optional[float] = Field(default=None, description="Supplier rating")
    delivery_time_days: Optional[str] = Field(default=None, description="Estimated shipping time")

class SupplierSourcingData(BaseModel):
    suppliers: List[SupplierData] = Field(description="List of potential suppliers")
    average_unit_cost: float = Field(description="Average cost per unit")
    recommended_supplier: Optional[SupplierData] = Field(default=None, description="Best supplier")

class FinancialMetrics(BaseModel):
    gross_profit_per_unit: float = Field(description="Selling Price - (Product Cost + Shipping)")
    net_profit_per_unit: float = Field(description="Gross Profit - Ad/Marketing Costs")
    margin_percentage: float = Field(description="Net Profit / Selling Price * 100")
    break_even_units: int = Field(description="Units needed to cover initial investment")
    monthly_revenue_potential: float = Field(description="Estimated monthly revenue")
    recommendation: str = Field(description="'GO', 'CAUTION', or 'NO-GO' based on margins")

class FinancialModelingData(BaseModel):
    target_selling_price: float = Field(description="Recommended price based on market")
    total_landed_cost: float = Field(description="Product Cost + Estimated Shipping + Customs")
    marketing_cpa: float = Field(description="Estimated Cost Per Acquisition (Ad spend)")
    metrics: FinancialMetrics
    assumptions: List[str] = Field(description="List of assumptions made")

class ProductDimensions(BaseModel):
    """Physical product specifications"""
    weight_kg: float = Field(description="Weight in kilograms")
    length_cm: Optional[float] = Field(default=None, description="Length in cm")
    width_cm: Optional[float] = Field(default=None, description="Width in cm")
    height_cm: Optional[float] = Field(default=None, description="Height in cm")
    volumetric_weight_kg: Optional[float] = Field(default=None, description="Calculated volumetric weight")


class FreightOption(BaseModel):
    """Shipping method option"""
    method: str = Field(description="Air, Sea, Express, ePacket")
    cost_per_unit: float = Field(description="Shipping cost per unit in USD")
    transit_days: int = Field(description="Estimated delivery time in days")
    min_order_quantity: int = Field(default=1, description="Minimum units for this method")
    recommended: bool = Field(default=False, description="Is this the recommended option")
    notes: str = Field(description="Additional information")


class CustomsDutyInfo(BaseModel):
    """Customs and import duty information"""
    hs_code: Optional[str] = Field(default=None, description="Harmonized System code")
    duty_rate_percentage: float = Field(description="Import duty rate as decimal (0.15 = 15%)")
    duty_cost_per_unit: float = Field(description="Calculated duty cost per unit")
    additional_fees: float = Field(default=0.0, description="MPF, HMF, and other fees")
    total_customs_cost: float = Field(description="Total customs cost per unit")
    notes: str = Field(description="Explanation of duty calculation")


class LogisticsData(BaseModel):
    """Complete logistics and shipping information"""
    product_dimensions: ProductDimensions
    origin_country: str = Field(default="China", description="Manufacturing country")
    destination_country: str = Field(description="Target market country")
    
    freight_options: List[FreightOption] = Field(description="Available shipping methods")
    recommended_freight: FreightOption = Field(description="Best shipping option")
    
    customs_duty: CustomsDutyInfo
    
    packaging_cost_per_unit: float = Field(description="Box, bubble wrap, tape, etc.")
    total_landed_cost_per_unit: float = Field(description="Product cost + Shipping + Customs + Packaging")
    
    estimated_delivery_time: str = Field(description="Human-readable delivery estimate")
    logistics_assumptions: List[str] = Field(description="All assumptions made")

    
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
    
    # AGENT -> SupplierSourcingData
    supplier_data: Optional[SupplierSourcingData] = None
    
    # AGENT -> FinancialModelingData
    financial_data: Optional[FinancialModelingData] = None
    
    # System Info
    error_message: Optional[Dict[str, str]] = None