"""
Logistics & Shipping Agent

This agent handles:
1. Product weight estimation
2. Shipping cost calculation (China/India to target market)
3. Customs duty rates
4. Delivery time estimates
5. Freight method recommendations (Air vs Sea)
6. Packaging cost estimation
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from datetime import datetime, timedelta
from dotenv import load_dotenv
from firecrawl import FirecrawlApp
from langchain_core.prompts import ChatPromptTemplate

from state import AgentState, ProductDimensions, FreightOption, CustomsDutyInfo, LogisticsData
from prompt_template import PRODUCT_DIMENSION_EXTRACTOR_SYSTEM_PROMPT
from llm_gateway import gateway

load_dotenv()
firecrawl = FirecrawlApp()

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def estimate_product_dimensions(product_name: str, category: str) -> ProductDimensions:
    """
    Estimate product dimensions using web search and category heuristics
    """
    print(f"      📦 Estimating dimensions for '{product_name}'...")
    
    # Try web search first
    try:
        search_query = f"{product_name} dimensions weight specifications"
        print(f"         🔍 Searching: {search_query}")
        
        search_results = firecrawl.search(search_query, limit=3)
        
        if hasattr(search_results, 'web') and search_results.web:
            # Extract from search results
            results_text = ""
            for item in search_results.web[:3]:
                results_text += f"\nTitle: {item.title}\n"
                results_text += f"Description: {item.description}\n"
            
            # Ask LLM to extract dimensions
            extract_prompt = ChatPromptTemplate.from_template(PRODUCT_DIMENSION_EXTRACTOR_SYSTEM_PROMPT)
            
            messages = extract_prompt.format_messages(
                product=product_name,
                results=results_text
            )
            
            extracted = gateway.invoke(
                messages=messages,
                structured_output=ProductDimensions
            )
            
            if extracted.weight_kg > 0:
                print(f"         ✅ Found via search: {extracted.weight_kg} kg")
                
                # Calculate volumetric weight if dimensions available
                if all([extracted.length_cm, extracted.width_cm, extracted.height_cm]):
                    volumetric = (extracted.length_cm * extracted.width_cm * extracted.height_cm) / 5000
                    extracted.volumetric_weight_kg = round(volumetric, 2)
                    print(f"         📐 Volumetric weight: {volumetric:.2f} kg")
                
                return extracted
        
    except Exception as e:
        print(f"         ⚠️ Search failed: {e}")
    
    # Fallback to category-based estimates
    print(f"         📊 Using category estimates...")
    
    category_specs = {
        "electronics": {"weight": 0.5, "length": 20, "width": 15, "height": 5},
        "phone accessories": {"weight": 0.1, "length": 15, "width": 10, "height": 3},
        "clothing": {"weight": 0.3, "length": 30, "width": 25, "height": 5},
        "shoes": {"weight": 0.8, "length": 35, "width": 25, "height": 15},
        "home & kitchen": {"weight": 1.5, "length": 30, "width": 30, "height": 20},
        "toys": {"weight": 0.5, "length": 25, "width": 20, "height": 15},
        "beauty & personal care": {"weight": 0.2, "length": 15, "width": 10, "height": 5},
        "sports & outdoors": {"weight": 1.2, "length": 40, "width": 30, "height": 20},
        "books": {"weight": 0.4, "length": 25, "width": 20, "height": 3},
        "jewelry": {"weight": 0.05, "length": 10, "width": 10, "height": 5},
        "furniture": {"weight": 15.0, "length": 100, "width": 60, "height": 80},
    }
    
    category_lower = category.lower() if category else "general"
    
    # Find matching category
    specs = None
    for key, val in category_specs.items():
        if key in category_lower:
            specs = val
            break
    
    if not specs:
        specs = {"weight": 1.0, "length": 25, "width": 20, "height": 10}
    
    dimensions = ProductDimensions(
        weight_kg=specs["weight"],
        length_cm=specs["length"],
        width_cm=specs["width"],
        height_cm=specs["height"]
    )
    
    # Calculate volumetric weight
    volumetric = (specs["length"] * specs["width"] * specs["height"]) / 5000
    dimensions.volumetric_weight_kg = round(volumetric, 2)
    
    print(f"         📦 Estimated: {specs['weight']} kg, {specs['length']}x{specs['width']}x{specs['height']} cm")
    
    return dimensions


def calculate_freight_options(
    dimensions: ProductDimensions,
    origin: str,
    destination: str,
    quantity: int = 100
) -> List[FreightOption]:
    """
    Calculate shipping costs for different freight methods
    
    Standard Routes:
    - China/India → USA/EU/UK
    
    Methods:
    1. ePacket (China only, <2kg, cheap but slow)
    2. Air Freight (Fast, expensive)
    3. Sea Freight (Slow, cheap for bulk)
    4. Express (DHL/FedEx, fastest, most expensive)
    """
    
    # Use actual weight or volumetric weight (whichever is higher)
    chargeable_weight = max(
        dimensions.weight_kg,
        dimensions.volumetric_weight_kg or 0
    )
    
    options = []
    
    # ==========================================
    # 1. ePacket (China to US/EU only)
    # ==========================================
    if origin.lower() in ["china", "cn"] and chargeable_weight <= 2.0:
        # ePacket rates (per kg)
        if chargeable_weight <= 0.5:
            epacket_cost = 2.50
        elif chargeable_weight <= 1.0:
            epacket_cost = 3.50
        else:
            epacket_cost = 2.50 + (chargeable_weight * 2.0)
        
        options.append(FreightOption(
            method="ePacket",
            cost_per_unit=round(epacket_cost, 2),
            transit_days=15,
            min_order_quantity=1,
            recommended=(quantity < 50),
            notes="Cheapest for small items, 15-25 days delivery"
        ))
    
    # ==========================================
    # 2. Air Freight
    # ==========================================
    # Air freight pricing (per kg)
    if chargeable_weight <= 1.0:
        air_rate_per_kg = 6.0
    elif chargeable_weight <= 5.0:
        air_rate_per_kg = 5.0
    elif chargeable_weight <= 10.0:
        air_rate_per_kg = 4.5
    else:
        air_rate_per_kg = 4.0
    
    air_cost = chargeable_weight * air_rate_per_kg
    
    # Add fuel surcharge (typically 15-20%)
    air_cost *= 1.18
    
    # Add handling fee
    air_cost += 2.0
    
    options.append(FreightOption(
        method="Air Freight",
        cost_per_unit=round(air_cost, 2),
        transit_days=7,
        min_order_quantity=10,
        recommended=(quantity >= 50 and quantity < 500),
        notes="Fast delivery, good for medium quantities"
    ))
    
    # ==========================================
    # 3. Sea Freight (for bulk orders)
    # ==========================================
    if quantity >= 100:
        # Sea freight is charged per CBM (cubic meter)
        # Cost per CBM: ~$150-200 (China to US West Coast)
        
        volume_cbm = (
            dimensions.length_cm * 
            dimensions.width_cm * 
            dimensions.height_cm
        ) / 1000000  # Convert cm³ to m³
        
        # Multiply by quantity to get total volume
        total_cbm = volume_cbm * quantity
        
        # Minimum 1 CBM charge
        if total_cbm < 1.0:
            total_cbm = 1.0
        
        # Sea freight rate per CBM
        sea_rate_per_cbm = 180.0
        
        total_sea_cost = total_cbm * sea_rate_per_cbm
        sea_cost_per_unit = total_sea_cost / quantity
        
        options.append(FreightOption(
            method="Sea Freight",
            cost_per_unit=round(sea_cost_per_unit, 2),
            transit_days=35,
            min_order_quantity=100,
            recommended=(quantity >= 500),
            notes=f"Cheapest for bulk. {total_cbm:.2f} CBM total volume."
        ))
    
    # ==========================================
    # 4. Express (DHL/FedEx)
    # ==========================================
    # Express rates (premium service)
    if chargeable_weight <= 0.5:
        express_cost = 15.0
    elif chargeable_weight <= 2.0:
        express_cost = 12.0 + (chargeable_weight * 8.0)
    elif chargeable_weight <= 5.0:
        express_cost = 10.0 + (chargeable_weight * 7.0)
    else:
        express_cost = 8.0 + (chargeable_weight * 6.5)
    
    options.append(FreightOption(
        method="Express (DHL/FedEx)",
        cost_per_unit=round(express_cost, 2),
        transit_days=3,
        min_order_quantity=1,
        recommended=False,  # Usually too expensive
        notes="Fastest but most expensive. Use for samples only."
    ))
    
    return options


def get_customs_duty_info(
    category: str,
    product_cost: float,
    destination: str = "US"
) -> CustomsDutyInfo:
    """
    Calculate customs duty and import fees
    
    US Import Fees:
    1. Customs Duty (varies by HS code)
    2. MPF (Merchandise Processing Fee): 0.3464% (min $27.75, max $538.40)
    3. HMF (Harbor Maintenance Fee): 0.125% (for sea freight)
    
    EU Import Fees:
    1. Customs Duty (varies)
    2. VAT (varies by country: 19-27%)
    """
    
    print(f"      🏛️ Calculating customs for '{category}' to {destination}...")
    
    # ==========================================
    # US HARMONIZED TARIFF SCHEDULE
    # ==========================================
    
    if destination.upper() in ["US", "USA", "UNITED STATES"]:
        # Simplified HS Code mapping
        hs_codes = {
            "electronics": ("8517.62", 0.0),          # Phones/tablets - Duty free
            "phone accessories": ("8517.70", 0.0),   # Phone parts - Duty free
            "computers": ("8471.30", 0.0),
            "headphones": ("8518.30", 0.0),
            "clothing": ("6203", 0.165),             # Men's clothing - 16.5%
            "t-shirts": ("6109", 0.165),
            "textiles": ("6302", 0.125),
            "footwear": ("6404", 0.125),             # Shoes - 12.5%
            "toys": ("9503", 0.0),                   # Toys - Duty free
            "home & kitchen": ("7323", 0.035),       # Kitchenware - 3.5%
            "furniture": ("9403", 0.0),
            "beauty & personal care": ("3304", 0.0),
            "cosmetics": ("3304", 0.0),
            "jewelry": ("7113", 0.055),              # Jewelry - 5.5%
            "watches": ("9102", 0.095),
            "bags": ("4202", 0.175),                 # Handbags - 17.5%
            "sports & outdoors": ("9506", 0.04),
            "books": ("4901", 0.0),
        }
        
        # Find matching category
        hs_code = None
        duty_rate = 0.05  # Default 5%
        
        category_lower = category.lower() if category else ""
        
        for key, (code, rate) in hs_codes.items():
            if key in category_lower:
                hs_code = code
                duty_rate = rate
                break
        
        # Calculate duty
        duty_cost = product_cost * duty_rate
        
        # MPF (Merchandise Processing Fee) - for shipments over $2500
        # For small quantities, this is negligible
        mpf_rate = 0.003464  # 0.3464%
        mpf_cost = max(product_cost * mpf_rate, 0.0)
        
        # For sea freight, add HMF (Harbor Maintenance Fee)
        hmf_cost = 0.0  # We'll add this if using sea freight
        
        additional_fees = mpf_cost + hmf_cost
        
        total_customs = duty_cost + additional_fees
        
        notes = f"HS Code: {hs_code or 'General'}, Duty: {duty_rate*100:.1f}%, MPF: ${mpf_cost:.2f}"
        
        return CustomsDutyInfo(
            hs_code=hs_code,
            duty_rate_percentage=duty_rate,
            duty_cost_per_unit=round(duty_cost, 2),
            additional_fees=round(additional_fees, 2),
            total_customs_cost=round(total_customs, 2),
            notes=notes
        )
    
    # ==========================================
    # EU CUSTOMS
    # ==========================================
    elif destination.upper() in ["EU", "EUROPE", "UK", "GB", "DE", "FR", "ES", "IT"]:
        # EU duty rates (simplified)
        eu_duty_rates = {
            "electronics": 0.0,
            "clothing": 0.12,
            "footwear": 0.12,
            "toys": 0.048,
            "jewelry": 0.025,
            "general": 0.05,
        }
        
        category_lower = category.lower() if category else ""
        duty_rate = 0.05
        
        for key, rate in eu_duty_rates.items():
            if key in category_lower:
                duty_rate = rate
                break
        
        duty_cost = product_cost * duty_rate
        
        # VAT (varies by country, using 20% average)
        vat_rate = 0.20
        vat_cost = (product_cost + duty_cost) * vat_rate
        
        total_customs = duty_cost + vat_cost
        
        notes = f"Duty: {duty_rate*100:.1f}%, VAT: {vat_rate*100:.0f}%"
        
        return CustomsDutyInfo(
            hs_code=None,
            duty_rate_percentage=duty_rate,
            duty_cost_per_unit=round(duty_cost, 2),
            additional_fees=round(vat_cost, 2),
            total_customs_cost=round(total_customs, 2),
            notes=notes
        )
    
    # ==========================================
    # OTHER COUNTRIES (General estimate)
    # ==========================================
    else:
        duty_rate = 0.08  # 8% general estimate
        duty_cost = product_cost * duty_rate
        
        return CustomsDutyInfo(
            hs_code=None,
            duty_rate_percentage=duty_rate,
            duty_cost_per_unit=round(duty_cost, 2),
            additional_fees=0.0,
            total_customs_cost=round(duty_cost, 2),
            notes=f"Estimated duty: {duty_rate*100:.0f}%"
        )


def estimate_packaging_cost(dimensions: ProductDimensions) -> float:
    """
    Estimate packaging cost (box, bubble wrap, tape, labels)
    """
    # Base packaging cost
    base_cost = 0.50  # Poly mailer or small box
    
    # Adjust based on product size
    if dimensions.volumetric_weight_kg and dimensions.volumetric_weight_kg > 5:
        packaging_cost = 2.50  # Large box + padding
    elif dimensions.volumetric_weight_kg and dimensions.volumetric_weight_kg > 2:
        packaging_cost = 1.50  # Medium box
    elif dimensions.weight_kg > 1:
        packaging_cost = 1.00  # Small box
    else:
        packaging_cost = base_cost  # Poly mailer
    
    return round(packaging_cost, 2)


# ==========================================
# MAIN AGENT FUNCTION
# ==========================================

def run_logistics_shipping(state: AgentState) -> AgentState:
    """
    Main Logistics & Shipping Agent
    
    Calculates:
    1. Product dimensions and weight
    2. Shipping costs for different methods
    3. Customs duty and import fees
    4. Packaging costs
    5. Total landed cost
    
    Args:
        state: AgentState with parsed_query and supplier_data
        
    Returns:
        Updated state with logistics_data populated
    """
    
    print(f"\n🚢 AGENT: Starting Logistics & Shipping Analysis...")
    
    # ==========================================
    # VALIDATION
    # ==========================================
    
    if not state.parsed_query:
        print("⚠️  No parsed query. Skipping.")
        return state
    
    if not state.supplier_data:
        print("⚠️  No supplier data. Need product cost for customs calculation.")
        # We can still estimate shipping, but customs will be inaccurate
    
    product_name = state.parsed_query.product_name or "Unknown Product"
    category = state.parsed_query.category or "General"
    target_market = state.parsed_query.target_market or "United States"
    
    print(f"   📦 Product: {product_name}")
    print(f"   📍 Route: China → {target_market}")
    
    # Get product cost for customs calculation
    if state.supplier_data and state.supplier_data.recommended_supplier:
        product_cost = state.supplier_data.recommended_supplier.price_per_unit
    elif state.supplier_data:
        product_cost = state.supplier_data.average_unit_cost
    else:
        product_cost = 10.0  # Fallback estimate
        print(f"   ⚠️ Using estimated product cost: ${product_cost}")
    
    # ==========================================
    # PHASE 1: DIMENSIONS
    # ==========================================
    
    dimensions = estimate_product_dimensions(product_name, category)
    
    # ==========================================
    # PHASE 2: FREIGHT OPTIONS
    # ==========================================
    
    print(f"\n   🚚 Calculating freight options...")
    
    # Assume initial order of 100 units (typical test order)
    freight_options = calculate_freight_options(
        dimensions=dimensions,
        origin="China",
        destination=target_market,
        quantity=100
    )
    
    # Find recommended option
    recommended = next((opt for opt in freight_options if opt.recommended), freight_options[0])
    
    print(f"      ✅ Found {len(freight_options)} shipping methods")
    for opt in freight_options:
        marker = "⭐" if opt.recommended else "  "
        print(f"      {marker} {opt.method}: ${opt.cost_per_unit:.2f}/unit ({opt.transit_days} days)")
    
    # ==========================================
    # PHASE 3: CUSTOMS DUTY
    # ==========================================
    
    print(f"\n   🏛️ Calculating customs & import fees...")
    
    customs = get_customs_duty_info(
        category=category,
        product_cost=product_cost,
        destination=target_market
    )
    
    print(f"      💰 Duty: ${customs.duty_cost_per_unit:.2f} ({customs.duty_rate_percentage*100:.1f}%)")
    print(f"      📋 Fees: ${customs.additional_fees:.2f}")
    print(f"      💵 Total Customs: ${customs.total_customs_cost:.2f}")
    
    # ==========================================
    # PHASE 4: PACKAGING
    # ==========================================
    
    packaging_cost = estimate_packaging_cost(dimensions)
    print(f"      📦 Packaging: ${packaging_cost:.2f}")
    
    # ==========================================
    # PHASE 5: TOTAL LANDED COST
    # ==========================================
    
    total_landed_cost = (
        product_cost +
        recommended.cost_per_unit +
        customs.total_customs_cost +
        packaging_cost
    )
    
    print(f"\n   💰 TOTAL LANDED COST BREAKDOWN:")
    print(f"      Product Cost:      ${product_cost:.2f}")
    print(f"      Shipping:          ${recommended.cost_per_unit:.2f}")
    print(f"      Customs & Fees:    ${customs.total_customs_cost:.2f}")
    print(f"      Packaging:         ${packaging_cost:.2f}")
    print(f"      ─────────────────────────────")
    print(f"      TOTAL:             ${total_landed_cost:.2f}")
    
    # ==========================================
    # PHASE 6: DELIVERY ESTIMATE
    # ==========================================
    
    delivery_estimate = f"{recommended.transit_days}-{recommended.transit_days + 5} business days"
    
    # ==========================================
    # PHASE 7: COMPILE RESULTS
    # ==========================================
    
    assumptions = [
        f"Product weight: {dimensions.weight_kg} kg",
        f"Volumetric weight: {dimensions.volumetric_weight_kg} kg",
        f"Shipping method: {recommended.method}",
        f"Origin: China (can adjust for other countries)",
        f"Destination: {target_market}",
        f"Order quantity: 100 units (affects sea freight viability)",
        customs.notes,
        f"Packaging: Standard e-commerce packaging"
    ]
    
    logistics_data = LogisticsData(
        product_dimensions=dimensions,
        origin_country="China",
        destination_country=target_market,
        freight_options=freight_options,
        recommended_freight=recommended,
        customs_duty=customs,
        packaging_cost_per_unit=packaging_cost,
        total_landed_cost_per_unit=round(total_landed_cost, 2),
        estimated_delivery_time=delivery_estimate,
        logistics_assumptions=assumptions
    )
    
    state.logistics_data = logistics_data
    
    print(f"\n   ✅ Logistics Analysis Complete!")
    
    return state