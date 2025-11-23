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
    
    
class AgentState(BaseModel):
    """Shared memory for the system"""
    # INPUT
    user_raw_query: str
    
    # AGENT -> ParsedQuery
    parsed_query: Optional[ParsedQuery] = None
    
    # AGENT -> MarketResearchData
    market_research_data: Optional[MarketResearchData] = None
    
    # System Info
    error_message: Optional[Dict[str, str]] = None