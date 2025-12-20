import re
import time
import json
from datetime import datetime, timedelta
from typing import List, Optional, Dict
from urllib.parse import urljoin, urlparse
import chromadb
from sentence_transformers import SentenceTransformer
from tavily import TavilyClient
from firecrawl import FirecrawlApp
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

from state import AgentState, SupplierSourcingData, SupplierData
from prompt_template import PAGE_ANALYSIS_PROMPT
from llm_gateway import gateway 

load_dotenv()

tavily = TavilyClient()
firecrawl = FirecrawlApp()


class SupplierCache:
    """
    Intelligent caching system for supplier sourcing results.
    Uses semantic search with sentence-transformers for optimal performance.
    """
    
    _instance = None  # Singleton pattern
    
    def __new__(cls, cache_duration_days=7):
        if cls._instance is None:
            cls._instance = super(SupplierCache, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, cache_duration_days=7):
        """
        Initialize the supplier cache with embedding model
        
        Args:
            cache_duration_days: How long to consider cached data "fresh" (default: 7 days)
        """
        if self._initialized:
            return
            
        self.cache_duration = timedelta(days=cache_duration_days)
        
        print("   🔧 Initializing Supplier Cache System...")
        
        # Initialize ChromaDB with persistence
        try:
            self.chroma_client = chromadb.PersistentClient(
                path="./chroma_db/supplier_cache"
            )
            
            # Create/get collection for supplier data
            self.collection = self.chroma_client.get_or_create_collection(
                name="supplier_cache",
                metadata={"description": "Cached supplier sourcing results with timestamps"}
            )
            
            # Create/get collection for search page links (new!)
            self.links_collection = self.chroma_client.get_or_create_collection(
                name="search_page_links",
                metadata={"description": "Cached product links extracted from search pages"}
            )
            
            print(f"   ✅ ChromaDB initialized. Cached entries: {self.collection.count()}")
            print(f"   ✅ Cached search page links: {self.links_collection.count()}")
            
        except Exception as e:
            print(f"   ❌ ChromaDB initialization failed: {e}")
            raise
        
        # Load embedding model (sentence-transformers - cleaner & faster!)
        try:
            print("   📦 Loading embedding model (all-MiniLM-L6-v2)...")
            self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            print("   ✅ Embedding model loaded successfully.")
            
        except Exception as e:
            print(f"   ❌ Model loading failed: {e}")
            raise
        
        self._initialized = True
    
    def get_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a text query
        
        Args:
            text: Product name or query to embed
            
        Returns:
            384-dimensional embedding vector
        """
        # Normalize the input
        text = text.lower().strip()
        
        # sentence-transformers handles everything: tokenization, pooling, normalization
        embedding = self.model.encode(
            text,
            normalize_embeddings=True,  # L2 normalization built-in
            show_progress_bar=False
        )
        
        return embedding.tolist()
    
    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts at once (more efficient)
        
        Args:
            texts: List of product names/queries
            
        Returns:
            List of 384-dimensional embedding vectors
        """
        # Normalize inputs
        texts = [t.lower().strip() for t in texts]
        
        # Batch encoding with automatic batching and progress
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=32,  # Optimal for CPU
            show_progress_bar=len(texts) > 10
        )
        
        return embeddings.tolist()
    
    def search_cache(
        self, 
        product_name: str,
        similarity_threshold: float = 0.7,
        n_results: int = 3
    ) -> Optional[SupplierSourcingData]:
        """
        Search cache for similar product queries
        
        Args:
            product_name: Product to search for
            similarity_threshold: Minimum similarity score (0-1) to consider a match
            n_results: Number of top results to check
            
        Returns:
            Cached SupplierSourcingData if fresh match found, None otherwise
        """
        try:
            query_embedding = self.get_embedding(product_name)
            
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=["metadatas", "documents", "distances"]
            )
            
            if not results['ids'][0]:
                print("   🔍 No cache hits found.")
                return None
            
            # Check the best match
            best_match_meta = results['metadatas'][0][0]
            best_match_doc = results['documents'][0][0]
            best_distance = results['distances'][0][0]
            
            # With normalized embeddings, ChromaDB distance is already cosine distance
            # distance = 1 - similarity, so similarity = 1 - distance
            similarity = 1 - best_distance
            
            if similarity < similarity_threshold:
                print(f"   🔍 Best match similarity too low ({similarity:.3f}). Fetching fresh data.")
                return None
            
            # Check freshness
            cached_time = datetime.fromisoformat(best_match_meta['timestamp'])
            age = datetime.now() - cached_time
            
            if age > self.cache_duration:
                print(f"   ⏰ Cache hit is stale ({age.days} days old). Fetching fresh data.")
                return None
            
            # Valid cache hit!
            print(f"   ✅ CACHE HIT! '{best_match_meta['product_name']}'")
            print(f"      Similarity: {similarity:.3f} | Age: {age.days} days")
            print(f"      Cached suppliers: {best_match_meta['num_suppliers']}")
            
            # Reconstruct SupplierSourcingData from cache
            cached_data = json.loads(best_match_doc)
            return SupplierSourcingData(**cached_data)
            
        except Exception as e:
            print(f"   ⚠️ Cache search error: {e}")
            return None
    
    def search_links_cache(
        self,
        search_url: str
    ) -> Optional[List[str]]:
        """
        Search for cached product links from a search page URL
        
        Args:
            search_url: The search page URL
            
        Returns:
            List of cached product URLs if found and fresh, None otherwise
        """
        try:
            # Use URL as the document ID (normalized)
            normalized_url = search_url.lower().strip()
            
            results = self.links_collection.get(
                ids=[normalized_url],
                include=["metadatas", "documents"]
            )
            
            if not results['ids']:
                return None
            
            metadata = results['metadatas'][0]
            cached_time = datetime.fromisoformat(metadata['timestamp'])
            age = datetime.now() - cached_time
            
            # Links cache expires faster (3 days)
            if age > timedelta(days=3):
                return None
            
            print(f"      ✅ Found {metadata['num_links']} cached links from this search page")
            
            # Return the list of URLs
            return json.loads(results['documents'][0])
            
        except Exception as e:
            print(f"      ⚠️ Links cache search error: {e}")
            return None
    
    def store_links_cache(
        self,
        search_url: str,
        product_links: List[str]
    ):
        """
        Store extracted product links from a search page
        
        Args:
            search_url: The search page URL
            product_links: List of extracted product URLs
        """
        try:
            normalized_url = search_url.lower().strip()
            
            self.links_collection.upsert(
                ids=[normalized_url],
                documents=[json.dumps(product_links)],
                metadatas=[{
                    "search_url": search_url,
                    "timestamp": datetime.now().isoformat(),
                    "num_links": len(product_links)
                }]
            )
            
            print(f"      💾 Cached {len(product_links)} links from search page")
            
        except Exception as e:
            print(f"      ⚠️ Links cache storage error: {e}")
    
    def store_cache(
        self, 
        product_name: str, 
        supplier_data: SupplierSourcingData
    ):
        """
        Store supplier data in cache with embedding
        
        Args:
            product_name: Product identifier
            supplier_data: Sourcing results to cache
        """
        try:
            embedding = self.get_embedding(product_name)
            
            # Create unique ID with timestamp
            doc_id = f"{product_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            self.collection.add(
                ids=[doc_id],
                embeddings=[embedding],
                documents=[supplier_data.model_dump_json()],
                metadatas=[{
                    "product_name": product_name,
                    "timestamp": datetime.now().isoformat(),
                    "num_suppliers": len(supplier_data.suppliers),
                    "avg_cost": supplier_data.average_unit_cost
                }]
            )
            
            print(f"   💾 Cached {len(supplier_data.suppliers)} suppliers for '{product_name}'")
            
        except Exception as e:
            print(f"   ⚠️ Cache storage error: {e}")
    
    def clear_stale_entries(self):
        """Remove entries older than cache_duration"""
        try:
            # Clean supplier cache
            all_entries = self.collection.get(include=["metadatas"])
            
            stale_ids = []
            for idx, metadata in enumerate(all_entries['metadatas']):
                cached_time = datetime.fromisoformat(metadata['timestamp'])
                if datetime.now() - cached_time > self.cache_duration:
                    stale_ids.append(all_entries['ids'][idx])
            
            if stale_ids:
                self.collection.delete(ids=stale_ids)
                print(f"   🧹 Cleaned {len(stale_ids)} stale supplier cache entries.")
            
            # Clean links cache (3 day expiry)
            links_entries = self.links_collection.get(include=["metadatas"])
            stale_link_ids = []
            
            for idx, metadata in enumerate(links_entries['metadatas']):
                cached_time = datetime.fromisoformat(metadata['timestamp'])
                if datetime.now() - cached_time > timedelta(days=3):
                    stale_link_ids.append(links_entries['ids'][idx])
            
            if stale_link_ids:
                self.links_collection.delete(ids=stale_link_ids)
                print(f"   🧹 Cleaned {len(stale_link_ids)} stale link cache entries.")
                
        except Exception as e:
            print(f"   ⚠️ Cache cleanup error: {e}")
    
    def get_cache_stats(self) -> dict:
        """Get statistics about the cache"""
        try:
            total_entries = self.collection.count()
            
            all_entries = self.collection.get(include=["metadatas"])
            
            if not all_entries['metadatas']:
                return {
                    "total_entries": 0,
                    "fresh_entries": 0,
                    "stale_entries": 0,
                    "avg_suppliers_per_entry": 0,
                    "cached_search_pages": self.links_collection.count()
                }
            
            fresh_count = 0
            stale_count = 0
            total_suppliers = 0
            
            for metadata in all_entries['metadatas']:
                cached_time = datetime.fromisoformat(metadata['timestamp'])
                age = datetime.now() - cached_time
                
                if age <= self.cache_duration:
                    fresh_count += 1
                else:
                    stale_count += 1
                
                total_suppliers += metadata.get('num_suppliers', 0)
            
            return {
                "total_entries": total_entries,
                "fresh_entries": fresh_count,
                "stale_entries": stale_count,
                "avg_suppliers_per_entry": total_suppliers / total_entries if total_entries > 0 else 0,
                "cached_search_pages": self.links_collection.count()
            }
            
        except Exception as e:
            print(f"   ⚠️ Stats error: {e}")
            return {}


# Initialize cache globally (singleton pattern ensures one instance)
supplier_cache = SupplierCache(cache_duration_days=7)


def extract_product_links_from_search_page(markdown_content: str, base_url: str) -> List[str]:
    """
    Extract product links from a search/category page.
    
    Args:
        markdown_content: Scraped markdown from search page
        base_url: Original search page URL for domain context
        
    Returns:
        List of product page URLs
    """
    product_urls = []
    
    # Extract all links from markdown
    # Markdown links: [text](url)
    link_pattern = r'\[([^\]]+)\]\(([^)]+)\)'
    matches = re.findall(link_pattern, markdown_content)
    
    domain = urlparse(base_url).netloc
    
    # Platform-specific product URL patterns
    product_patterns = [
        '/product/',
        '/item/',
        '/p/',
        '/dp/',  # Amazon
        'product_detail',
        'productdetail',
        'product-detail',
        'alibaba.com/product-detail',
        'aliexpress.com/item',
        'made-in-china.com/product/',
        'dhgate.com/product/',
        'indiamart.com/proddetail',
        'globalsources.com/si/',
        'thomasnet.com/products/',
    ]
    
    # Navigation/filter patterns to exclude
    navigation_patterns = [
        'page=',
        'sort=',
        'filter=',
        'login',
        'register',
        'cart',
        'checkout',
        'account',
        'search',
        'category',
        'wholesale',
        '/find',
        '#',  # Anchor links
    ]
    
    for link_text, url in matches:
        # Convert relative URLs to absolute
        full_url = urljoin(base_url, url)
        
        # Check if it's a product page
        is_product = any(pattern in full_url.lower() for pattern in product_patterns)
        
        # Check if it's navigation/filter
        is_navigation = any(pattern in full_url.lower() for pattern in navigation_patterns)
        
        # Must be from same domain and be a product page
        if is_product and not is_navigation and domain in full_url:
            product_urls.append(full_url)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_urls = []
    for url in product_urls:
        if url not in seen:
            seen.add(url)
            unique_urls.append(url)
    
    return unique_urls[:15]  # Limit to top 15 products per search page


def extract_relevant_content(markdown_content: str, max_tokens: int = 2000) -> str:
    """
    Smart content extraction that prioritizes pricing information.
    
    Strategy:
    1. Search for price-related sections
    2. Extract context around pricing
    3. Include beginning (product name/title) + middle (specs) + end (pricing)
    
    Args:
        markdown_content: Full scraped markdown
        max_tokens: Approximate token limit (~4 chars per token)
        
    Returns:
        Optimized content string for LLM parsing
    """
    
    # Price-related keywords (case-insensitive)
    price_keywords = [
        'price', 'cost', '$', '€', '£', '¥', 'usd', 'moq',
        'minimum order', 'unit price', 'wholesale', 'bulk',
        'per piece', 'per unit', 'pricing', 'total cost',
        'order quantity', 'price range', 'sample price'
    ]
    
    lines = markdown_content.split('\n')
    total_lines = len(lines)
    
    # Find lines with price information
    price_line_indices = []
    for idx, line in enumerate(lines):
        line_lower = line.lower()
        if any(keyword in line_lower for keyword in price_keywords):
            price_line_indices.append(idx)
    
    if not price_line_indices:
        # No explicit price found - use smart chunking
        # Take: First 20% + Last 40% (where pricing usually is)
        char_limit = max_tokens * 4
        first_chunk_size = int(len(markdown_content) * 0.2)
        last_chunk_size = char_limit - first_chunk_size
        
        result = (
            markdown_content[:first_chunk_size] +
            "\n\n... [middle content omitted] ...\n\n" +
            markdown_content[-last_chunk_size:]
        )
        return result
    
    # Build context around price mentions
    context_window = 10  # Lines before/after price mention
    important_sections = set()
    
    for price_idx in price_line_indices:
        start = max(0, price_idx - context_window)
        end = min(total_lines, price_idx + context_window)
        important_sections.update(range(start, end))
    
    # Always include the beginning (product title/name)
    important_sections.update(range(min(30, total_lines)))
    
    # Extract selected lines
    selected_lines = [lines[i] for i in sorted(important_sections)]
    result = '\n'.join(selected_lines)
    
    # If still too long, truncate from middle
    char_limit = max_tokens * 4
    if len(result) > char_limit:
        # Keep first 30% and last 70% (prioritize pricing at end)
        first_part_size = int(char_limit * 0.3)
        last_part_size = char_limit - first_part_size
        
        result = (
            result[:first_part_size] +
            "\n\n[...]\n\n" +
            result[-last_part_size:]
        )
    
    return result


def extract_structured_sections(markdown_content: str) -> Dict[str, str]:
    """
    Alternative approach: Parse markdown into structured sections.
    Works well for well-formatted product pages.
    
    Returns:
        Dictionary with extracted sections
    """
    sections = {
        'title': '',
        'description': '',
        'specifications': '',
        'pricing': '',
        'moq': '',
        'shipping': ''
    }
    
    lines = markdown_content.split('\n')
    current_section = 'description'
    
    for line in lines:
        line_lower = line.lower()
        
        # Detect section headers
        if line.startswith('#'):
            if any(word in line_lower for word in ['price', 'pricing', 'cost']):
                current_section = 'pricing'
            elif any(word in line_lower for word in ['specification', 'details', 'feature']):
                current_section = 'specifications'
            elif any(word in line_lower for word in ['shipping', 'delivery', 'logistics']):
                current_section = 'shipping'
            continue
        
        # Detect inline pricing (even without headers)
        if any(char in line for char in ['$', '€', '£', '¥']):
            if 'moq' in line_lower or 'minimum' in line_lower:
                sections['moq'] += line + '\n'
            else:
                sections['pricing'] += line + '\n'
        
        # Capture first heading as title
        elif not sections['title'] and line.strip():
            sections['title'] = line.strip()
        
        # Add to current section
        else:
            sections[current_section] += line + '\n'
    
    return sections


def run_supplier_sourcing(state: AgentState) -> AgentState:
    """
    Enhanced supplier sourcing with two-phase crawling:
    
    ARCHITECTURE:
    1. Cache Check: Search for similar product queries in cache
    2. Discovery: Find both search pages AND direct product pages
    3. Expansion: Extract product links from search pages (with caching)
    4. Extraction: Scrape all product pages for pricing data
    5. Aggregation: Calculate metrics and recommend best supplier
    6. Cache Storage: Store results for future queries
    
    Args:
        state: Current agent state with parsed query
        
    Returns:
        Updated state with supplier_data populated
    """
    print(f"\n📦 AGENT: Starting Enhanced Supplier Sourcing...")
    
    if not state.parsed_query or not state.parsed_query.product_name:
        print("⚠️  No product name found. Skipping.")
        return state

    product_name = state.parsed_query.product_name
    
    # ==========================================
    # PHASE 0: CACHE CHECK (Supplier Data)
    # ==========================================
    cached_data = supplier_cache.search_cache(
        product_name=product_name,
        similarity_threshold=0.7
    )
    
    if cached_data:
        state.supplier_data = cached_data
        print(f"   🚀 Returned cached data with {len(cached_data.suppliers)} suppliers")
        return state
    
    # ==========================================
    # PHASE 1: DISCOVERY (Tavily Search)
    # ==========================================
    print(f"   🔎 Scouting for suppliers of '{product_name}'...")
    
    search_query = f"wholesale {product_name} manufacturer supplier price"
    
    try:
        tavily_response = tavily.search(
            query=search_query,
            search_depth="basic",
            max_results=15,  # Increased for better coverage
            include_domains=[
                "alibaba.com", 
                "aliexpress.com", 
                "made-in-china.com", 
                "dhgate.com", 
                "indiamart.com", 
                "thomasnet.com",
                "globalsources.com",
                "tradewheel.com"
            ], 
            include_answer=False
        )
        
        all_urls = [res.get('url') for res in tavily_response.get('results', [])]
        
        # Separate search pages from direct product pages
        search_pages = []
        product_pages = []
        
        for url in all_urls:
            url_lower = url.lower()
            
            # Identify search/category pages
            if any(term in url_lower for term in [
                'search', 'category', '/find', '/s/', 'results',
                'wholesale', 'products', 'categories'
            ]):
                search_pages.append(url)
            # Identify direct product pages
            elif any(term in url_lower for term in [
                '/product/', '/item/', '/p/', '/dp/', 
                'product-detail', 'productdetail', 'proddetail'
            ]):
                product_pages.append(url)
        
        print(f"   📋 Discovery Results:")
        print(f"      • {len(search_pages)} search/category pages")
        print(f"      • {len(product_pages)} direct product pages")

    except Exception as e:
        print(f"   ❌ Tavily Search failed: {e}")
        if not state.error_message:
            state.error_message = {}
        state.error_message["supplier_sourcing_search"] = str(e)
        return state
    
    # ==========================================
    # PHASE 2: EXPANSION (Extract Links from Search Pages)
    # ==========================================
    expanded_product_pages = list(product_pages)  # Start with direct links
    
    max_search_pages = min(len(search_pages), 4)  # Limit to top 4 search pages
    
    for search_url in search_pages[:max_search_pages]:
        print(f"   🔗 Expanding: {search_url[:70]}...")
        
        # Check cache first
        cached_links = supplier_cache.search_links_cache(search_url)
        
        if cached_links:
            expanded_product_pages.extend(cached_links)
            continue
        
        # Cache miss - scrape the search page
        try:
            scrape_result = firecrawl.scrape(search_url, formats=["markdown"])
            markdown = getattr(scrape_result, 'markdown', '')
            
            if markdown and len(markdown) > 200:
                extracted_links = extract_product_links_from_search_page(markdown, search_url)
                
                if extracted_links:
                    expanded_product_pages.extend(extracted_links)
                    # Cache the extracted links
                    supplier_cache.store_links_cache(search_url, extracted_links)
                    print(f"      ✅ Extracted {len(extracted_links)} product links")
                else:
                    print(f"      ⚠️  No product links found on page")
            else:
                print(f"      ⚠️  Insufficient content")
            
            time.sleep(1)  # Rate limiting between search page scrapes
            
        except Exception as e:
            print(f"      ❌ Failed to expand: {e}")
            continue
    
    # Remove duplicates while preserving order
    unique_product_pages = list(dict.fromkeys(expanded_product_pages))
    
    print(f"   🎯 Total unique product pages identified: {len(unique_product_pages)}")
    
    if not unique_product_pages:
        print("   ⚠️ No product pages found after expansion.")
        if not state.error_message:
            state.error_message = {}
        state.error_message["supplier_sourcing_discovery"] = "No product pages found"
        return state
    
    # ==========================================
    # PHASE 3: EXTRACTION (Scrape Product Pages)
    # ==========================================
    valid_suppliers = []
    max_scrapes = min(len(unique_product_pages), 10)  # Increased limit
    
    print(f"   🕷️  Starting detailed extraction from {max_scrapes} product pages...")
    
    for idx, url in enumerate(unique_product_pages[:max_scrapes], 1):
        print(f"   [{idx}/{max_scrapes}] Scraping: {url[:65]}...")
        
        try:
            scrape_result = firecrawl.scrape(url, formats=["markdown"])
            markdown_content = getattr(scrape_result, 'markdown', '')

            if not markdown_content or len(markdown_content) < 100:
                print(f"      ⚠️  Insufficient content (< 100 chars)")
                continue
            
            # ==========================================
            # SMART CONTENT EXTRACTION
            # ==========================================
            
            # Try structured extraction first (better for organized pages)
            sections = extract_structured_sections(markdown_content)
            
            # Build optimized prompt content
            if sections['pricing'] or sections['moq']:
                # Structured approach found pricing info
                optimized_content = f"""
PRODUCT TITLE:
{sections['title']}

PRICING INFORMATION:
{sections['pricing']}

MINIMUM ORDER QUANTITY:
{sections['moq']}

SPECIFICATIONS:
{sections['specifications'][:1000]}

SHIPPING INFO:
{sections['shipping'][:500]}
"""
            else:
                # Fallback to smart extraction
                optimized_content = extract_relevant_content(markdown_content, max_tokens=2000)
            
            print(f"      📄 Extracted {len(optimized_content)} chars (from {len(markdown_content)} total)")
            
            # ==========================================
            # PHASE 3.1: LLM PARSING (with Gateway)
            # ==========================================
            analyze_prompt = ChatPromptTemplate.from_template(PAGE_ANALYSIS_PROMPT)
            messages = analyze_prompt.format_messages(
                product_name=product_name,
                markdown_content=optimized_content
            )
            
            supplier_info = gateway.invoke(
                messages=messages,
                structured_output=SupplierData
            )
            
            # Fill in metadata
            supplier_info.product_url = url
            
            # Validate extracted data
            if supplier_info.price_per_unit and supplier_info.price_per_unit > 0:
                valid_suppliers.append(supplier_info)
                print(f"      ✅ {supplier_info.supplier_name[:30]} | ${supplier_info.price_per_unit}/unit | MOQ: {supplier_info.moq}")
            else:
                print(f"      ⚠️  Price not found - trying FULL CONTENT fallback...")
                
                # FALLBACK: Try with last 4000 chars (where pricing usually is)
                fallback_content = markdown_content[-4000:]
                
                messages = analyze_prompt.format_messages(
                    product_name=product_name,
                    markdown_content=fallback_content
                )
                
                supplier_info = gateway.invoke(
                    messages=messages,
                    structured_output=SupplierData
                )
                supplier_info.product_url = url
                
                if supplier_info.price_per_unit and supplier_info.price_per_unit > 0:
                    valid_suppliers.append(supplier_info)
                    print(f"      ✅ (Fallback) {supplier_info.supplier_name[:30]} | ${supplier_info.price_per_unit}")
                else:
                    print(f"      ❌ Still no valid price found")
            
            # Rate limiting - be polite to servers
            time.sleep(2)

        except Exception as e:
            print(f"      ❌ Extraction failed: {str(e)[:100]}")
            continue

    # ==========================================
    # PHASE 4: VALIDATION & AGGREGATION
    # ==========================================
    if not valid_suppliers:
        print("   ❌ No valid supplier data extracted from any page.")
        if not state.error_message:
            state.error_message = {}
        state.error_message["supplier_sourcing_extraction"] = "No valid suppliers found"
        return state
    
    # Calculate metrics
    prices = [s.price_per_unit for s in valid_suppliers]
    avg_cost = sum(prices) / len(prices)
    
    # Recommend supplier based on best price/rating balance
    def score_supplier(supplier: SupplierData) -> float:
        """Lower score = better. Balances price and rating."""
        price_score = supplier.price_per_unit
        
        # Penalty for low/missing ratings
        if not supplier.rating:
            rating_penalty = 1.0  # Assume mediocre rating
        else:
            rating_penalty = (5 - supplier.rating) * 0.3
        
        return price_score + rating_penalty
    
    best_supplier = min(valid_suppliers, key=score_supplier)
    
    # ==========================================
    # PHASE 5: BUILD RESULT OBJECT
    # ==========================================
    sourcing_data = SupplierSourcingData(
        suppliers=valid_suppliers,
        average_unit_cost=round(avg_cost, 2),
        recommended_supplier=best_supplier
    )
    
    state.supplier_data = sourcing_data
    
    # ==========================================
    # PHASE 6: CACHE STORAGE
    # ==========================================
    supplier_cache.store_cache(product_name, sourcing_data)
    
    # ==========================================
    # SUCCESS SUMMARY
    # ==========================================
    print(f"\n   ✅ SUCCESS: Supplier Sourcing Complete")
    print(f"      • Found: {len(valid_suppliers)} valid suppliers")
    print(f"      • Average Cost: ${avg_cost:.2f}")
    print(f"      • Price Range: ${min(prices):.2f} - ${max(prices):.2f}")
    if best_supplier:
        print(f"      • 🏆 Recommended: {best_supplier.supplier_name}")
        print(f"        - Price: ${best_supplier.price_per_unit}/unit")
        print(f"        - MOQ: {best_supplier.moq}")
        if best_supplier.rating:
            print(f"        - Rating: {best_supplier.rating}/5.0")
    
    return state