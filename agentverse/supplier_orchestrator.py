from uuid import uuid4
from uagents import Agent, Context, Protocol
from dotenv import load_dotenv
import os
from datetime import datetime
from typing import Dict, List, Any, Tuple
from supabase import create_client, Client
import google.generativeai as genai


from uagents_core.contrib.protocols.chat import (
    ChatAcknowledgement,
    ChatMessage,
    TextContent,
    EndSessionContent,
    chat_protocol_spec,
)

# Import compliance, financial, demand, and logistics models
from models.compliance import ComplianceRequest, ComplianceResponse
from models.financial import FinancialRequest, FinancialResponse
from models.demand import DemandRequest, DemandResponse
from models.logistics import LogisticsRequest, LogisticsResponse
from models.find_supplier import FindSupplierRequest, FindSupplierResponse

# Import test utilities
from test_func.orchestrator_helpers import (
    verify_orchestrator_connection,
    log_message_transmission,
    validate_request_before_sending,
)

# Import instruction prompts for handling irrelevant queries
from prompts.instructions import (
    INSTRUCTIONS_PROMPT,
    MONITOR_FEATURE_PROMPT,
    MONITOR_FEATURE_HELP,
    GREETING_RESPONSE,
    MONITOR_WITHOUT_SUPPLIER_RESPONSE,
    HELP_RESPONSE,
)

load_dotenv()

SUPPLIER_ORCHESTRATOR_SEED = os.getenv("SUPPLIER_ORCHESTRATOR_SEED")

# Initialize Gemini for personalized responses - OPTIMIZED: Limited to 300 chars
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel(
    "gemini-2.5-flash",
    generation_config={
        "max_output_tokens": 500,  # ~300 chars (4 chars per token)
        "temperature": 0.5,  # Balanced for friendly greetings
    },
)


def generate_personalized_response(
    user_message: str,
    query_type: str,
    fallback_response: str,
    supplier_name: str = None,
) -> str:
    """
    Generate a personalized response using Google Gemini LLM.

    Args:
        user_message: The user's original message
        query_type: Type of query (greeting, help, monitor_no_supplier, monitor_help, irrelevant)
        fallback_response: Static response to use if LLM fails
        supplier_name: Name of the supplier (if user has one)

    Returns:
        Personalized response string
    """
    try:
        # OPTIMIZED: Shorter prompts to save input tokens
        # Special handling for monitor_help - user has a supplier and needs guidance
        if query_type == "monitor_help" and supplier_name:
            prompt = f"User has supplier {supplier_name}. They said: '{user_message}'. Guide them to say 'monitor my supplier'. Be brief."
        else:
            # Build concise context based on query type
            context_map = {
                "greeting": "Greet warmly. Introduce ChainGuard AI briefly.",
                "help": "Explain ChainGuard AI's features: find suppliers, monitor suppliers.",
                "monitor_no_supplier": "Tell user to find a supplier first before monitoring.",
                "irrelevant": "Politely redirect to ChainGuard AI features.",
            }

            context = context_map.get(query_type, "Guide user on ChainGuard AI usage.")

            # Shorter prompt format
            prompt = (
                f"{context} User said: '{user_message}'. Respond in max 2 sentences."
            )

        # Call Gemini for personalized response
        response = gemini_model.generate_content(prompt)

        # FIXED: Check if response has valid parts before accessing .text
        # This prevents errors when finish_reason = MAX_TOKENS (2)
        if not response.candidates:
            return fallback_response

        candidate = response.candidates[0]

        # Check finish_reason: 1=STOP (success), 2=MAX_TOKENS, 3=SAFETY, 4=RECITATION
        if candidate.finish_reason != 1:  # Not a natural completion
            # Use fallback if response was blocked or cut off
            return fallback_response

        # Safely extract text
        if not candidate.content or not candidate.content.parts:
            return fallback_response

        generated_response = candidate.content.parts[0].text.strip()

        # Validate the response isn't empty or too short
        if generated_response and len(generated_response) > 20:
            return generated_response
        else:
            return fallback_response

    except Exception as e:
        # If Gemini fails, use the fallback static response
        return fallback_response


def determine_product_category(user_query: str) -> str:
    """
    Determine the product category from the user query.

    Maps business keywords to product categories:
    - food: coffee, pizza, bakery, restaurant, tea, chocolate, food, grocery, cafe
    - clothing: clothing, apparel, fashion, textile, garment, shoes, accessories
    - electronics: technology, software, electronics, computer, phone, hardware

    Args:
        user_query: The user's search query

    Returns:
        Product category: "food", "clothing", or "electronics"
    """
    query_lower = user_query.lower()

    # Food category keywords
    food_keywords = [
        "coffee",
        "pizza",
        "bakery",
        "restaurant",
        "tea",
        "chocolate",
        "food",
        "grocery",
        "cafe",
        "catering",
        "beverage",
        "organic",
        "farm",
        "produce",
        "meat",
        "dairy",
        "snack",
        "juice",
        "wine",
        "beer",
        "bread",
        "pasta",
        "sauce",
        "spice",
        "ingredient",
        "water",
        "chicken",
        "beef",
        "pork",
        "fish",
        "seafood",
        "fruit",
        "vegetable",
        "grain",
        "rice",
        "flour",
        "sugar",
        "oil",
        "milk",
        "cheese",
        "egg",
        "honey",
        "nut",
        "candy",
        "ice cream",
        "alcohol",
    ]

    # Clothing category keywords
    clothing_keywords = [
        "clothing",
        "apparel",
        "fashion",
        "textile",
        "garment",
        "shoes",
        "accessories",
        "wear",
        "fabric",
        "leather",
        "denim",
        "cotton",
        "wool",
        "silk",
        "uniform",
        "sportswear",
        "footwear",
        "bag",
        "hat",
    ]

    # Electronics category keywords
    electronics_keywords = [
        "technology",
        "software",
        "electronics",
        "computer",
        "phone",
        "hardware",
        "device",
        "gadget",
        "tech",
        "digital",
        "smart",
        "appliance",
        "machine",
        "equipment",
        "component",
        "chip",
    ]

    # Check for matches
    for keyword in food_keywords:
        if keyword in query_lower:
            return "food"

    for keyword in clothing_keywords:
        if keyword in query_lower:
            return "clothing"

    for keyword in electronics_keywords:
        if keyword in query_lower:
            return "electronics"

    # Default to food if no match
    return "food"


def clear_session_data(ctx: Context) -> None:
    """Clear all session data to start fresh with a new supplier search"""
    ctx.storage.set("selected_supplier", None)
    ctx.storage.set("selected_product_category", None)
    ctx.storage.set("active_sessions", {})
    ctx.storage.set("pending_responses", {})
    ctx.storage.set("approved_suppliers", [])
    ctx.logger.info("Session data cleared")


def is_reset_command(user_query: str) -> bool:
    """
    Check if the user is explicitly requesting a session reset.

    Recognized commands:
    - "reset"
    - "start over"
    - "new search"
    - "clear"
    - "clear session"
    - "new supplier"
    - "find new supplier"
    - "start fresh"
    """
    query_lower = user_query.lower().strip()

    reset_patterns = [
        "reset",
        "start over",
        "new search",
        "clear",
        "clear session",
        "new supplier",
        "find new supplier",
        "start fresh",
        "fresh start",
        "begin again",
        "restart",
    ]

    for pattern in reset_patterns:
        if pattern in query_lower:
            return True

    return False


def analyze_query_relevance(
    user_query: str, has_selected_supplier: bool = False
) -> Tuple[bool, str, str]:
    """
    Analyze whether the user query is relevant to ChainGuard AI's capabilities.

    This function uses semantic analysis to determine if the query is:
    1. A valid supplier search request
    2. A valid monitoring request
    3. A greeting or help request
    4. A monitor help request (user has supplier but asks how to monitor)
    5. A reset/new search request
    6. An irrelevant/off-topic query

    Args:
        user_query: The user's input message
        has_selected_supplier: Whether the user already has a selected supplier

    Returns:
        Tuple of (is_relevant, query_type, response_message)
        - is_relevant: True if this is a valid ChainGuard request
        - query_type: "find_supplier", "monitor", "greeting", "help", "monitor_help", "reset", or "irrelevant"
        - response_message: Pre-built response for non-actionable queries
    """
    query_lower = user_query.lower().strip()

    # Check for explicit reset command first
    if is_reset_command(user_query):
        return True, "reset", ""

    # Empty or very short queries
    if len(query_lower) < 2:
        if has_selected_supplier:
            return False, "monitor_help", MONITOR_FEATURE_HELP
        return False, "irrelevant", GREETING_RESPONSE

    # SUPPLIER SEARCH INDICATORS
    # Look for patterns that indicate a supplier search intent
    # If user mentions "supplier" with any intent pattern, it's a valid search
    find_supplier_patterns = [
        "find",
        "search",
        "looking for",
        "look for",
        "need a",
        "want a",
        "want to find",
        "get me",
        "get a",
        "show me",
        "i need",
        "i want",
    ]

    # Strong supplier intent indicators - these alone indicate a supplier search
    supplier_intent_keywords = [
        "supplier",
        "vendor",
        "source",
        "b corp",
        "b-corp",
        "bcorp",
    ]

    # Business owner patterns - if combined with any product, it's a supplier search
    business_owner_patterns = [
        "i'm a",
        "i am a",
        "im a",
        "business owner",
        "business",
        "company",
        "shop",
        "store",
        "owner",
    ]

    # MONITORING INDICATORS
    monitoring_patterns = [
        "monitor",
        "monitoring",
        "update",
        "status",
        "check on",
        "check up",
        "how is",
        "report on",
        "track",
        "tracking",
        "logistics",
        "inventory",
        "demand",
    ]

    # MONITOR HELP INDICATORS - user asking how to monitor
    monitor_help_patterns = [
        "how do i monitor",
        "how can i monitor",
        "how to monitor",
        "what now",
        "what next",
        "what do i do",
        "what can i do now",
        "now what",
        "what's next",
        "whats next",
        "next step",
        "next steps",
    ]

    # GREETING INDICATORS
    greeting_patterns = [
        "hi",
        "hello",
        "hey",
        "good morning",
        "good afternoon",
        "good evening",
        "greetings",
        "howdy",
        "what's up",
        "whats up",
        "yo",
    ]

    # HELP/INFO INDICATORS
    help_patterns = [
        "help",
        "how do",
        "how does",
        "how to",
        "what is",
        "what's",
        "what are",
        "explain",
        "tell me about",
        "can you",
        "what can",
        "features",
        "capabilities",
        "instructions",
        "tutorial",
        "guide",
    ]

    # IMPORTANT: Check for MONITORING intent FIRST before supplier search
    # Reason: "monitor my supplier" contains both "monitor" AND "supplier"
    # We need to prioritize the more specific intent (monitoring) over generic supplier mentions
    has_monitoring_pattern = any(
        pattern in query_lower for pattern in monitoring_patterns
    )

    if has_monitoring_pattern:
        if has_selected_supplier:
            return True, "monitor", ""
        else:
            # User wants to monitor but hasn't selected a supplier yet
            return False, "monitor_no_supplier", MONITOR_WITHOUT_SUPPLIER_RESPONSE

    # Check for supplier search intent - FLEXIBLE DETECTION
    # Any of these combinations indicates a supplier search:
    # 1. Contains "supplier", "vendor", or "b corp" (strong intent)
    # 2. Contains a find pattern + business owner pattern (e.g., "I'm a UK business owner find me a water supplier")
    # 3. Contains find pattern + "supplier" keyword

    has_supplier_keyword = any(kw in query_lower for kw in supplier_intent_keywords)
    has_find_pattern = any(pattern in query_lower for pattern in find_supplier_patterns)
    has_business_pattern = any(
        pattern in query_lower for pattern in business_owner_patterns
    )

    # Strong supplier intent - if they mention "supplier", "vendor", or "b corp", it's a search
    if has_supplier_keyword:
        return True, "find_supplier", ""

    # Business owner with find intent - "I'm a UK business owner looking for X"
    if has_find_pattern and has_business_pattern:
        return True, "find_supplier", ""

    # Check if user is asking HOW to monitor (they have a supplier)
    if has_selected_supplier:
        has_monitor_help_pattern = any(
            pattern in query_lower for pattern in monitor_help_patterns
        )
        if has_monitor_help_pattern:
            return False, "monitor_help", MONITOR_FEATURE_HELP

    # Check for greetings (simple hi, hello, etc.)
    # Only match if the query is primarily a greeting (short queries)
    if len(query_lower.split()) <= 3:
        is_greeting = any(
            query_lower.startswith(g) or query_lower == g for g in greeting_patterns
        )
        if is_greeting:
            if has_selected_supplier:
                # User has supplier and says hi - guide them to monitor
                return False, "monitor_help", MONITOR_FEATURE_HELP
            return False, "greeting", GREETING_RESPONSE

    # Check for help/info requests
    has_help_pattern = any(pattern in query_lower for pattern in help_patterns)
    if has_help_pattern:
        # If user has supplier and asks for help, guide them to monitor
        if has_selected_supplier:
            monitor_keywords = ["monitor", "supplier", "track", "check"]
            if any(kw in query_lower for kw in monitor_keywords):
                return False, "monitor_help", MONITOR_FEATURE_HELP

        # Check if they're asking about ChainGuard specifically
        chainguard_mentions = [
            "chainguard",
            "chain guard",
            "this agent",
            "this platform",
            "you",
        ]
        if any(mention in query_lower for mention in chainguard_mentions):
            return False, "help", HELP_RESPONSE
        # General help request
        return False, "help", HELP_RESPONSE

    # If user has a supplier and sends something irrelevant, guide them to monitor
    if has_selected_supplier:
        return False, "monitor_help", MONITOR_FEATURE_HELP

    # If none of the above, it's likely irrelevant
    # Generate a friendly response guiding them back to ChainGuard features
    return False, "irrelevant", GREETING_RESPONSE


def extract_user_country(user_query: str) -> str:
    """
    Extract the user's country from their query.

    Looks for patterns like:
    - "I'm a United Kingdom business owner"
    - "I'm in Germany"
    - "I'm from Spain"
    - "I am a UK business"
    - "I'm a German company"

    Args:
        user_query: The user's search query

    Returns:
        Country name (default: "United States")
    """
    query_lower = user_query.lower()

    # List of countries to look for (with variations)
    countries = {
        # UK variations
        "united kingdom": "United Kingdom",
        "uk": "United Kingdom",
        "british": "United Kingdom",
        "britain": "United Kingdom",
        "england": "United Kingdom",
        "scotland": "United Kingdom",
        "wales": "United Kingdom",
        # US variations
        "united states": "United States",
        "usa": "United States",
        "us": "United States",
        "american": "United States",
        "america": "United States",
        # EU countries
        "germany": "Germany",
        "german": "Germany",
        "france": "France",
        "french": "France",
        "spain": "Spain",
        "spanish": "Spain",
        "italy": "Italy",
        "italian": "Italy",
        "netherlands": "Netherlands",
        "dutch": "Netherlands",
        "holland": "Netherlands",
        "belgium": "Belgium",
        "belgian": "Belgium",
        "portugal": "Portugal",
        "portuguese": "Portugal",
        "austria": "Austria",
        "austrian": "Austria",
        "ireland": "Ireland",
        "irish": "Ireland",
        "sweden": "Sweden",
        "swedish": "Sweden",
        "norway": "Norway",
        "norwegian": "Norway",
        "denmark": "Denmark",
        "danish": "Denmark",
        "finland": "Finland",
        "finnish": "Finland",
        "poland": "Poland",
        "polish": "Poland",
        "greece": "Greece",
        "greek": "Greece",
        "czech": "Czech Republic",
        "hungary": "Hungary",
        "hungarian": "Hungary",
        "romania": "Romania",
        "romanian": "Romania",
        # Other developed economies
        "canada": "Canada",
        "canadian": "Canada",
        "australia": "Australia",
        "australian": "Australia",
        "new zealand": "New Zealand",
        "japan": "Japan",
        "japanese": "Japan",
        "south korea": "South Korea",
        "korean": "South Korea",
        "singapore": "Singapore",
        "singaporean": "Singapore",
        "switzerland": "Switzerland",
        "swiss": "Switzerland",
        # Latin America
        "mexico": "Mexico",
        "mexican": "Mexico",
        "brazil": "Brazil",
        "brazilian": "Brazil",
        "argentina": "Argentina",
        "argentinian": "Argentina",
        "chile": "Chile",
        "chilean": "Chile",
        "colombia": "Colombia",
        "colombian": "Colombia",
        "peru": "Peru",
        "peruvian": "Peru",
        # Asia
        "china": "China",
        "chinese": "China",
        "india": "India",
        "indian": "India",
        "taiwan": "Taiwan",
        "taiwanese": "Taiwan",
        "thailand": "Thailand",
        "thai": "Thailand",
        "vietnam": "Vietnam",
        "vietnamese": "Vietnam",
        "indonesia": "Indonesia",
        "indonesian": "Indonesia",
        "malaysia": "Malaysia",
        "malaysian": "Malaysia",
        "philippines": "Philippines",
        "filipino": "Philippines",
        # Middle East / Africa
        "israel": "Israel",
        "israeli": "Israel",
        "united arab emirates": "United Arab Emirates",
        "uae": "United Arab Emirates",
        "dubai": "United Arab Emirates",
        "saudi arabia": "Saudi Arabia",
        "saudi": "Saudi Arabia",
        "south africa": "South Africa",
        "south african": "South Africa",
        "egypt": "Egypt",
        "egyptian": "Egypt",
        "morocco": "Morocco",
        "moroccan": "Morocco",
        "kenya": "Kenya",
        "kenyan": "Kenya",
        "nigeria": "Nigeria",
        "nigerian": "Nigeria",
        # Other
        "turkey": "Turkey",
        "turkish": "Turkey",
        "russia": "Russia",
        "russian": "Russia",
        "ukraine": "Ukraine",
        "ukrainian": "Ukraine",
    }

    # Check for country mentions (longer matches first to avoid partial matches)
    # Sort by length descending to check longer matches first
    sorted_countries = sorted(countries.keys(), key=len, reverse=True)

    for country_key in sorted_countries:
        if country_key in query_lower:
            return countries[country_key]

    # Default to United States if no country found
    return "United States"


# Supabase configuration for monitoring data storage
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Global Supabase client (initialized at startup)
# Note: We use a global variable because ctx.storage only accepts JSON-serializable data
supabase_client: Client | None = None

supplier_orchestrator = Agent(
    name="supplier_orchestrator",
    seed=SUPPLIER_ORCHESTRATOR_SEED,
    port=8000,
    mailbox=True,  # Required for ASI:1 communication through Agentverse
)

chat_proto = Protocol(name="chat_protocol", spec=chat_protocol_spec)

COMPLIANCE_AGENT_ADDRESS = os.getenv(
    "COMPLIANCE_AGENT_ADDRESS",
)
FINANCIAL_AGENT_ADDRESS = os.getenv(
    "FINANCIAL_AGENT_ADDRESS",
)
DEMAND_AGENT_ADDRESS = os.getenv(
    "DEMAND_AGENT_ADDRESS",
)
LOGISTICS_AGENT_ADDRESS = os.getenv(
    "LOGISTICS_AGENT_ADDRESS",
)
FIND_SUPPLIER_AGENT_ADDRESS = os.getenv(
    "FIND_SUPPLIER_ADDRESS",
)

orchestrator_protocol = Protocol(name="supplier_orchestrator_protocol", version="1.0")


@supplier_orchestrator.on_event("startup")
async def startup(ctx: Context):
    """Initialize orchestrator on startup"""
    ctx.logger.info("Supplier Orchestrator starting up")

    # Initialize Supabase client for monitoring data storage
    global supabase_client
    if SUPABASE_URL and SUPABASE_KEY:
        try:
            supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
            ctx.logger.info("Supabase client initialized")
        except Exception as e:
            ctx.logger.error(f"Failed to initialize Supabase: {e}")
            supabase_client = None
    else:
        supabase_client = None

    # Initialize storage
    ctx.storage.set("active_sessions", {})
    ctx.storage.set("approved_suppliers", [])
    ctx.storage.set("selected_supplier", None)
    ctx.storage.set("selected_product_category", None)
    ctx.storage.set("pending_responses", {})
    ctx.storage.set(
        "message_trace",
        {
            "received_from_asi": [],
            "sent_to_compliance": [],
            "received_from_compliance": [],
            "sent_to_asi": [],
        },
    )

    # Verify connection on startup
    conn_status = verify_orchestrator_connection(ctx)
    ctx.logger.info(f"Connection Status: {conn_status['status']}")


@supplier_orchestrator.on_event("shutdown")
async def shutdown(ctx: Context):
    """Clean up on shutdown"""
    ctx.logger.info("Supplier Orchestrator shutting down")


@chat_proto.on_message(ChatMessage)
async def handle_chat_message(ctx: Context, sender: str, msg: ChatMessage):
    """Handle chat messages from user via ASI:1"""

    # Step 0: Verify and log message received from ASI:1
    log_message_transmission(
        ctx, "RECEIVED", "ChatMessage", str(msg.msg_id), {"sender": sender}
    )

    # Step 1: Send acknowledgment immediately
    ack = ChatAcknowledgement(
        timestamp="",
        acknowledged_msg_id=msg.msg_id,
    )
    await ctx.send(sender, ack)

    # Step 2: Extract user query from message content
    user_query = None

    for content_item in msg.content:
        if isinstance(content_item, TextContent):
            user_query = content_item.text
            break

    if not user_query:
        ctx.logger.warning("No text content in message")
        error_response = ChatMessage(
            timestamp="",
            msg_id=uuid4(),
            content=[
                TextContent(
                    type="text",
                    text="Error: No query text provided in message",
                )
            ],
        )
        await ctx.send(sender, error_response)
        log_message_transmission(
            ctx, "SENT", "ChatMessage", str(error_response.msg_id), {"type": "error"}
        )
        return

    # Step 3: Analyze query relevance to determine how to handle it
    selected_supplier = ctx.storage.get("selected_supplier")
    has_supplier = selected_supplier is not None and selected_supplier != ""

    is_relevant, query_type, instruction_response = analyze_query_relevance(
        user_query, has_selected_supplier=has_supplier
    )

    # Handle reset command - clear session and confirm
    if query_type == "reset":
        clear_session_data(ctx)

        reset_response = ChatMessage(
            timestamp="",
            msg_id=uuid4(),
            content=[
                TextContent(
                    type="text",
                    text='Session Reset Complete!\n\nYour previous supplier search has been cleared. You can now start a fresh search.\n\nTo find a new supplier, tell me:\n- What type of business you have (e.g., coffee shop, restaurant, clothing store)\n- Your location/country\n\nExample: "I\'m a UK business owner looking for a coffee supplier"',
                ),
                EndSessionContent(type="end-session"),
            ],
        )
        await ctx.send(sender, reset_response)
        log_message_transmission(
            ctx,
            "SENT",
            "ChatMessage",
            str(reset_response.msg_id),
            {"type": "reset_confirmation"},
        )
        return

    # If this is a new supplier search and we already have a supplier, auto-reset
    if query_type == "find_supplier" and has_supplier:
        clear_session_data(ctx)
        has_supplier = False

    # Handle irrelevant or informational queries with personalized responses
    if not is_relevant:

        # For monitor_help, include supplier name in the fallback response
        fallback = instruction_response
        if query_type == "monitor_help" and selected_supplier:
            fallback = MONITOR_FEATURE_HELP.format(supplier_name=selected_supplier)

        # Generate a personalized response using Gemini
        personalized_response = generate_personalized_response(
            user_message=user_query,
            query_type=query_type,
            fallback_response=fallback,
            supplier_name=selected_supplier if has_supplier else None,
        )

        instruction_message = ChatMessage(
            timestamp="",
            msg_id=uuid4(),
            content=[
                TextContent(
                    type="text",
                    text=personalized_response.strip(),
                ),
                EndSessionContent(type="end-session"),
            ],
        )
        await ctx.send(sender, instruction_message)
        log_message_transmission(
            ctx,
            "SENT",
            "ChatMessage",
            str(instruction_message.msg_id),
            {"type": "personalized_instruction", "query_type": query_type},
        )
        return

    # Determine if this is a monitoring request based on query_type
    is_monitoring_request = query_type == "monitor"

    # Step 4: Store session information
    msg_id = str(msg.msg_id)
    active_sessions = ctx.storage.get("active_sessions") or {}
    active_sessions[msg_id] = {
        "sender": sender,
        "query": user_query,
        "mode": "monitor" if is_monitoring_request else "find",
    }
    ctx.storage.set("active_sessions", active_sessions)

    # Step 5: Route based on mode
    if is_monitoring_request:
        # MONITORING MODE: Forward to Demand and Logistics Agents

        # Get the selected supplier and product category (from the find_supplier phase)
        selected_supplier = ctx.storage.get("selected_supplier")
        product_category = ctx.storage.get("selected_product_category") or "food"

        if not selected_supplier:
            ctx.logger.warning("No selected supplier available for monitoring")
            error_response = ChatMessage(
                timestamp="",
                msg_id=uuid4(),
                content=[
                    TextContent(
                        type="text",
                        text="Error: No supplier selected for monitoring. Please run the find_supplier feature first to select a supplier.",
                    )
                ],
            )
            await ctx.send(sender, error_response)
            log_message_transmission(
                ctx,
                "SENT",
                "ChatMessage",
                str(error_response.msg_id),
                {"type": "error"},
            )
            return

        # Initialize pending responses for monitoring (2 monitoring agents: demand, logistics)
        pending_responses = ctx.storage.get("pending_responses") or {}
        pending_responses[msg_id] = {
            "demand_response": None,
            "logistics_response": None,
            "sender": sender,
            "user_query": user_query,
            "mode": "monitor",
        }
        ctx.storage.set("pending_responses", pending_responses)

        try:
            # SEQUENTIAL: Send to Demand Agent FIRST

            demand_request = DemandRequest(
                request_id=msg_id,
                supplier_name=selected_supplier,
                product_category=product_category,
                timestamp="",
            )

            await ctx.send(DEMAND_AGENT_ADDRESS, demand_request)

        except Exception as e:
            ctx.logger.error(f"Error sending to monitoring agents: {e}")

            # Clean up pending responses
            pending_responses.pop(msg_id, None)
            ctx.storage.set("pending_responses", pending_responses)

            # Send error response to user
            error_response = ChatMessage(
                timestamp="",
                msg_id=uuid4(),
                content=[
                    TextContent(
                        type="text",
                        text=f"Error forwarding monitoring request: {str(e)}",
                    )
                ],
            )
            await ctx.send(sender, error_response)
            log_message_transmission(
                ctx,
                "SENT",
                "ChatMessage",
                str(error_response.msg_id),
                {"type": "error"},
            )

        return

    # Step 6: FIND SUPPLIER MODE - First forward to find_supplier agent

    # Initialize pending responses tracking for this request
    pending_responses = ctx.storage.get("pending_responses") or {}
    pending_responses[msg_id] = {
        "find_supplier_response": None,
        "compliance_response": None,
        "financial_response": None,
        "sender": sender,
        "user_query": user_query,
        "mode": "find",
    }
    ctx.storage.set("pending_responses", pending_responses)

    # Extract business category from user query
    user_query_lower = user_query.lower()
    business_category = "general"

    # Try to identify business category
    category_keywords = {
        "coffee": "coffee",
        "pizza": "pizza",
        "chocolate": "chocolate",
        "food": "food",
        "restaurant": "restaurant",
        "clothing": "clothing",
        "apparel": "clothing",
        "fashion": "fashion",
        "tea": "tea",
        "bakery": "bakery",
        "furniture": "furniture",
        "technology": "technology",
        "software": "software",
    }

    for keyword, category in category_keywords.items():
        if keyword in user_query_lower:
            business_category = category
            break

    try:
        # Send to Find Supplier Agent first
        find_supplier_request = FindSupplierRequest(
            request_id=msg_id,
            user_query=user_query,
            business_category=business_category,
            timestamp="",
        )

        await ctx.send(FIND_SUPPLIER_AGENT_ADDRESS, find_supplier_request)

        log_message_transmission(
            ctx,
            "SENT",
            "FindSupplierRequest",
            msg_id,
            {
                "user_query": user_query,
                "business_category": business_category,
            },
        )

    except Exception as e:
        ctx.logger.error(f"Error sending to find_supplier agent: {e}")

        # Clean up pending responses
        pending_responses.pop(msg_id, None)
        ctx.storage.set("pending_responses", pending_responses)

        # Send error response to user
        error_response = ChatMessage(
            timestamp="",
            msg_id=uuid4(),
            content=[
                TextContent(
                    type="text",
                    text=f"Error forwarding request to find_supplier agent: {str(e)}",
                )
            ],
        )
        await ctx.send(sender, error_response)
        log_message_transmission(
            ctx, "SENT", "ChatMessage", str(error_response.msg_id), {"type": "error"}
        )


@chat_proto.on_message(ChatAcknowledgement)
async def handle_acknowledgement(ctx: Context, sender: str, msg: ChatAcknowledgement):
    """Handle acknowledgment messages from user"""
    ctx.logger.info(
        f"Acknowledgment received from {sender} for message: {msg.acknowledged_msg_id}"
    )


compliance_protocol = Protocol(name="compliance_response_protocol", version="1.0")


@compliance_protocol.on_message(model=ComplianceResponse)
async def handle_compliance_response(
    ctx: Context, sender: str, msg: ComplianceResponse
):
    """Handle compliance response and trigger financial agent (sequential)"""
    ctx.logger.info(
        f"Received compliance response for {msg.supplier_name}: {msg.compliance_score}/100"
    )

    log_message_transmission(
        ctx,
        "RECEIVED",
        "ComplianceResponse",
        msg.request_id,
        {"supplier_name": msg.supplier_name, "compliance_score": msg.compliance_score},
    )

    try:
        pending_responses = ctx.storage.get("pending_responses") or {}

        if msg.request_id not in pending_responses:
            ctx.logger.warning(
                f"No pending response tracking for request {msg.request_id}"
            )
            return

        pending_responses[msg.request_id]["compliance_response"] = msg.model_dump()
        ctx.storage.set("pending_responses", pending_responses)

        # Get supplier info from stored data
        best_supplier_data = pending_responses[msg.request_id].get("best_supplier")
        user_query = pending_responses[msg.request_id].get("user_query")

        if not best_supplier_data:
            ctx.logger.error(
                "No supplier data found, cannot proceed to financial agent"
            )
            return

        # Extract user's country from their query
        user_country = extract_user_country(user_query or "")

        financial_request = FinancialRequest(
            request_id=msg.request_id,
            supplier_name=best_supplier_data.get("company_name"),
            industry=best_supplier_data.get("industry", "general"),
            b_corp_profile_url=best_supplier_data.get("b_corp_profile_url", ""),
            user_country=user_country,
            timestamp="",
        )

        await ctx.send(FINANCIAL_AGENT_ADDRESS, financial_request)

        log_message_transmission(
            ctx,
            "SENT",
            "FinancialRequest",
            msg.request_id,
            {
                "supplier_name": financial_request.supplier_name,
                "user_country": user_country,
            },
        )

    except Exception as e:
        ctx.logger.error(f"Error handling compliance response: {e}")
        import traceback

        traceback.print_exc()


# Add Financial Response Protocol
financial_protocol = Protocol(name="financial_response_protocol", version="1.0")

# Add Demand Response Protocol
demand_protocol = Protocol(name="demand_response_protocol", version="1.0")

# Add Logistics Response Protocol
logistics_protocol = Protocol(name="logistics_response_protocol", version="1.0")

# Add Find Supplier Response Protocol
find_supplier_protocol = Protocol(name="find_supplier_response_protocol", version="1.0")


@find_supplier_protocol.on_message(model=FindSupplierResponse)
async def handle_find_supplier_response(
    ctx: Context, sender: str, msg: FindSupplierResponse
):
    """Handle find supplier response and forward to analysis agents"""

    log_message_transmission(
        ctx,
        "RECEIVED",
        "FindSupplierResponse",
        msg.request_id,
        {
            "success": msg.success,
            "search_category": msg.search_category,
            "total_results": msg.total_results_found,
        },
    )

    try:
        # Get pending response tracking
        pending_responses = ctx.storage.get("pending_responses") or {}

        if msg.request_id not in pending_responses:
            ctx.logger.warning(
                f"No pending response tracking for request {msg.request_id}"
            )
            return

        # Store find_supplier response
        pending_responses[msg.request_id]["find_supplier_response"] = msg.model_dump()
        ctx.storage.set("pending_responses", pending_responses)

        user_sender = pending_responses[msg.request_id].get("sender")
        user_query = pending_responses[msg.request_id].get("user_query")

        if not user_sender:
            ctx.logger.error(f"No sender found for request {msg.request_id}")
            return

        # Check if search was successful
        if not msg.success:
            ctx.logger.error(f"Find supplier search failed: {msg.error_message}")

            # Send error message to user
            error_response = ChatMessage(
                timestamp="",
                msg_id=uuid4(),
                content=[
                    TextContent(
                        type="text",
                        text=f"Supplier Search Failed\n\nCategory: {msg.search_category}\nError: {msg.error_message}\n\nPlease try a different search term.",
                    ),
                    EndSessionContent(type="end-session"),
                ],
            )

            await ctx.send(user_sender, error_response)

            # Clean up
            pending_responses.pop(msg.request_id, None)
            ctx.storage.set("pending_responses", pending_responses)

            return

        # Search was successful
        best_supplier = msg.best_supplier

        if not best_supplier:
            ctx.logger.warning("No supplier found in successful response")

            no_results_response = ChatMessage(
                timestamp="",
                msg_id=uuid4(),
                content=[
                    TextContent(
                        type="text",
                        text=f"No Suppliers Found\n\nCategory: {msg.search_category}\nWe couldn't find any B Corporation certified companies matching your search.\n\nPlease try a different category or search term.",
                    ),
                    EndSessionContent(type="end-session"),
                ],
            )

            await ctx.send(user_sender, no_results_response)

            # Clean up
            pending_responses.pop(msg.request_id, None)
            ctx.storage.set("pending_responses", pending_responses)

            return

        # Supplier found successfully - Store it and forward to analysis agents
        ctx.logger.info(f"Found supplier: {best_supplier.company_name}")

        # Store the selected supplier for monitoring
        ctx.storage.set("selected_supplier", best_supplier.company_name)

        # Determine and store product category from user query
        product_category = determine_product_category(user_query)
        ctx.storage.set("selected_product_category", product_category)

        # Store supplier info for response handlers
        pending_responses[msg.request_id]["best_supplier"] = best_supplier.model_dump()
        pending_responses[msg.request_id]["user_query"] = user_query
        ctx.storage.set("pending_responses", pending_responses)

        try:
            # SEQUENTIAL: Send ONLY to Compliance Agent first

            compliance_request = ComplianceRequest(
                request_id=msg.request_id,
                supplier_name=best_supplier.company_name,
                industry=best_supplier.industry or "general",
                company_values=user_query,
                b_corp_profile_url=best_supplier.b_corp_profile_url or "",
                timestamp="",
            )
            await ctx.send(COMPLIANCE_AGENT_ADDRESS, compliance_request)
            log_message_transmission(
                ctx,
                "SENT",
                "ComplianceRequest",
                msg.request_id,
                {
                    "supplier_name": compliance_request.supplier_name,
                    "industry": compliance_request.industry,
                },
            )

        except Exception as e:
            ctx.logger.error(f"Error sending to analysis agents: {e}")
            import traceback

            traceback.print_exc()

            # Clean up pending responses
            pending_responses.pop(msg.request_id, None)
            ctx.storage.set("pending_responses", pending_responses)

            # Send error response to user
            error_response = ChatMessage(
                timestamp="",
                msg_id=uuid4(),
                content=[
                    TextContent(
                        type="text",
                        text=f"Error forwarding supplier to analysis agents: {str(e)}",
                    ),
                    EndSessionContent(type="end-session"),
                ],
            )
            await ctx.send(user_sender, error_response)
            log_message_transmission(
                ctx,
                "SENT",
                "ChatMessage",
                str(error_response.msg_id),
                {"type": "error"},
            )

    except Exception as e:
        ctx.logger.error(f"Error handling find supplier response: {e}")
        import traceback

        traceback.print_exc()


@financial_protocol.on_message(model=FinancialResponse)
async def handle_financial_response(ctx: Context, sender: str, msg: FinancialResponse):
    """Handle financial response and combine results (final step)"""
    ctx.logger.info(
        f"Received financial response for {msg.supplier_name}: {msg.financial_score}/100"
    )

    log_message_transmission(
        ctx,
        "RECEIVED",
        "FinancialResponse",
        msg.request_id,
        {"supplier_name": msg.supplier_name, "financial_score": msg.financial_score},
    )

    try:
        pending_responses = ctx.storage.get("pending_responses") or {}

        if msg.request_id not in pending_responses:
            ctx.logger.warning(
                f"No pending response tracking for request {msg.request_id}"
            )
            return

        pending_responses[msg.request_id]["financial_response"] = msg.model_dump()
        ctx.storage.set("pending_responses", pending_responses)

        # Combine and send final response
        await check_and_send_combined_response(ctx, msg.request_id)

    except Exception as e:
        ctx.logger.error(f"Error handling financial response: {e}")
        import traceback

        traceback.print_exc()


@demand_protocol.on_message(model=DemandResponse)
async def handle_demand_response(ctx: Context, sender: str, msg: DemandResponse):
    """Handle demand response and trigger logistics agent (sequential)"""
    ctx.logger.info(f"Received demand response for {msg.supplier_name}")

    log_message_transmission(
        ctx,
        "RECEIVED",
        "DemandResponse",
        msg.request_id,
        {
            "supplier_name": msg.supplier_name,
            "overall_performance": msg.overall_performance,
        },
    )

    try:
        pending_responses = ctx.storage.get("pending_responses") or {}

        if msg.request_id not in pending_responses:
            ctx.logger.warning(
                f"No pending response tracking for request {msg.request_id}"
            )
            return

        pending_responses[msg.request_id]["demand_response"] = msg.model_dump()
        ctx.storage.set("pending_responses", pending_responses)

        # SEQUENTIAL: Now trigger Logistics Agent
        supplier_name = msg.supplier_name
        product_category = ctx.storage.get("selected_product_category") or "food"

        logistics_request = LogisticsRequest(
            request_id=msg.request_id,
            supplier_name=supplier_name,
            product_category=product_category,
            timestamp="",
        )

        await ctx.send(LOGISTICS_AGENT_ADDRESS, logistics_request)

        log_message_transmission(
            ctx,
            "SENT",
            "LogisticsRequest",
            msg.request_id,
            {
                "supplier_name": logistics_request.supplier_name,
                "product_category": product_category,
            },
        )

    except Exception as e:
        ctx.logger.error(f"Error handling demand response: {e}")
        import traceback

        traceback.print_exc()


@logistics_protocol.on_message(model=LogisticsResponse)
async def handle_logistics_response(ctx: Context, sender: str, msg: LogisticsResponse):
    """Handle logistics response and combine results (sequential - final step)"""
    ctx.logger.info(f"Received logistics response for {msg.supplier_name}")

    log_message_transmission(
        ctx,
        "RECEIVED",
        "LogisticsResponse",
        msg.request_id,
        {
            "supplier_name": msg.supplier_name,
            "overall_logistics_status": msg.overall_logistics_status,
        },
    )

    try:
        pending_responses = ctx.storage.get("pending_responses") or {}

        if msg.request_id not in pending_responses:
            ctx.logger.warning(
                f"No pending response tracking for request {msg.request_id}"
            )
            return

        pending_responses[msg.request_id]["logistics_response"] = msg.model_dump()
        ctx.storage.set("pending_responses", pending_responses)

        # Combine and send final monitoring response
        await check_and_send_monitoring_response(ctx, msg.request_id)

    except Exception as e:
        ctx.logger.error(f"Error handling logistics response: {e}")
        import traceback

        traceback.print_exc()

async def store_supplier_data_in_supabase(
    ctx: Context,
    supplier_name: str,
    compliance_response: Dict[str, Any],
    financial_response: Dict[str, Any],
    supplier_info: str,
) -> bool:
    """
    Store supplier analysis results in Supabase database.

    Args:
        ctx: Agent context
        supplier_name: Name of the supplier
        compliance_response: Compliance agent response data
        financial_response: Financial agent response data
        supplier_info: Recommendation text (Approved/Not Approved)

    Returns:
        bool: True if successful, False otherwise
    """
    global supabase_client

    try:
        if not supabase_client:
            ctx.logger.warning(
                "Supabase client not initialized - skipping data storage"
            )
            return False

        # Format compliance text
        supplier_compliance = f"""Compliance Score: {compliance_response.get('compliance_score')}/100

Ethics & Worker Treatment:
{compliance_response.get('ethics_info', 'N/A')}

Sustainability Practices:
{compliance_response.get('sustainability_info', 'N/A')}

Violations: {', '.join(compliance_response.get('violations', [])) if compliance_response.get('violations') else 'None'}"""

        # Format financial text
        supplier_finance = f"""Financial Score: {financial_response.get('financial_score')}/100

{financial_response.get('financial_details', 'N/A')}

Risk Factors: {', '.join(financial_response.get('risk_factors', [])) if financial_response.get('risk_factors') else 'None'}

Trade Route: {financial_response.get('user_country', 'N/A')} → {financial_response.get('supplier_country', 'N/A')}"""

        # Prepare data for insertion
        supplier_data = {
            "supplier_name": supplier_name,
            "supplier_compliance": supplier_compliance,
            "supplier_finance": supplier_finance,
            "supplier_info": supplier_info,
        }

        ctx.logger.info("Storing supplier data in Supabase...")
        ctx.logger.info(f"   Supplier: {supplier_name}")
        ctx.logger.info(f"   Recommendation: {supplier_info}")

        # Insert data into Find_supplier table
        result = supabase_client.table("Find_supplier").insert(supplier_data).execute()

        ctx.logger.info("Successfully stored supplier data in Supabase")
        ctx.logger.info(
            f"   Record ID: {result.data[0]['id'] if result.data else 'N/A'}"
        )

        return True

    except Exception as e:
        ctx.logger.error(f"Error storing supplier data in Supabase: {e}")
        import traceback

        traceback.print_exc()
        return False
        
async def store_monitoring_data_in_supabase(
    ctx: Context,
    supplier_name: str,
    demand_response: Dict[str, Any],
    logistics_response: Dict[str, Any],
) -> bool:
    """
    Store monitoring agent responses in Supabase database.

    Args:
        ctx: Agent context
        supplier_name: Name of the supplier being monitored
        demand_response: Demand agent response data
        logistics_response: Logistics agent response data

    Returns:
        bool: True if successful, False otherwise
    """
    global supabase_client

    try:
        if not supabase_client:
            ctx.logger.warning(
                "Supabase client not initialized - skipping data storage"
            )
            return False

        # Determine overall status based on both agent responses
        demand_performance = demand_response.get("overall_performance", "UNKNOWN")
        logistics_status = logistics_response.get("overall_logistics_status", "UNKNOWN")

        # Create overall status summary
        overall_status = (
            f"Performance: {demand_performance} | Logistics: {logistics_status}"
        )

        # Prepare data for insertion
        monitoring_data = {
            "supplier_name": supplier_name,
            "demand_content": demand_response,
            "logistics_content": logistics_response,
            "overall_status": overall_status,
        }

        ctx.logger.info("Storing monitoring data in Supabase...")
        ctx.logger.info(f"   Supplier: {supplier_name}")
        ctx.logger.info(f"   Overall Status: {overall_status}")

        # Insert data into monitoring_conversations table
        result = (
            supabase_client.table("monitoring_conversations")
            .insert(monitoring_data)
            .execute()
        )

        ctx.logger.info("Successfully stored monitoring data in Supabase")
        ctx.logger.info(
            f"   Record ID: {result.data[0]['id'] if result.data else 'N/A'}"
        )

        return True

    except Exception as e:
        ctx.logger.error(f"Error storing monitoring data in Supabase: {e}")
        import traceback

        traceback.print_exc()
        return False


async def check_and_send_monitoring_response(ctx: Context, request_id: str):
    """Check if all 2 monitoring responses are received, combine them, and send to user"""
    pending_responses = ctx.storage.get("pending_responses") or {}

    if request_id not in pending_responses:
        ctx.logger.warning(f"No pending response for request {request_id}")
        return

    response_data = pending_responses[request_id]
    demand_response = response_data.get("demand_response")
    logistics_response = response_data.get("logistics_response")

    # Check if we have BOTH monitoring responses
    if demand_response is None or logistics_response is None:
        ctx.logger.info(f"Still waiting for monitoring responses...")
        ctx.logger.info(f"Demand: {'RECEIVED' if demand_response else 'PENDING'}")
        ctx.logger.info(f"Logistics: {'RECEIVED' if logistics_response else 'PENDING'}")
        return

    # Store monitoring data in Supabase database
    supplier_name = demand_response.get("supplier_name", "Unknown")
    await store_monitoring_data_in_supabase(
        ctx,
        supplier_name,
        demand_response,
        logistics_response,
    )

    user_sender = response_data.get("sender")

    if not user_sender:
        ctx.logger.error(f"No sender found for request {request_id}")
        return

    # Build combined monitoring report
    demand_alerts_text = ""
    if demand_response.get("alerts"):
        demand_alerts_text = "\n".join(
            [f"  - {alert}" for alert in demand_response.get("alerts", [])]
        )
    else:
        demand_alerts_text = "  - No active alerts"

    logistics_alerts_text = ""
    if logistics_response.get("alerts"):
        logistics_alerts_text = "\n".join(
            [f"  - {alert}" for alert in logistics_response.get("alerts", [])]
        )
    else:
        logistics_alerts_text = "  - No active alerts"

    response_text = f"""
SUPPLIER MONITORING UPDATE

Supplier: {demand_response.get('supplier_name', 'N/A')}
Overall Performance: {demand_response.get('overall_performance', 'N/A')}
Logistics Status: {logistics_response.get('overall_logistics_status', 'N/A')}

===== OPERATIONAL METRICS (DEMAND FORECAST) =====

DELIVERY & DELAYS
Status: {demand_response.get('delay_status', 'N/A')}
{demand_response.get('delay_details', 'N/A')}

---

SHIPPING & LOGISTICS
Status: {demand_response.get('shipping_status', 'N/A')}
{demand_response.get('shipping_details', 'N/A')}

---

QUALITY TRACKING
Status: {demand_response.get('quality_status', 'N/A')}
{demand_response.get('quality_details', 'N/A')}

---

DEMAND ALERTS ({len(demand_response.get('alerts', []))})
{demand_alerts_text}

===== INVENTORY & LOGISTICS MANAGEMENT =====

CURRENT INVENTORY LEVEL
Status: {logistics_response.get('current_inventory_level', 'N/A')}
Current Units: {logistics_response.get('current_inventory_units', 0)}
{logistics_response.get('inventory_details', 'N/A')}

---

INVENTORY FORECAST
Predicted Level: {logistics_response.get('predicted_inventory_level', 'N/A')}
Predicted Units: {logistics_response.get('predicted_inventory_units', 0)}
{logistics_response.get('inventory_forecast', 'N/A')}

---

RESTOCKING STATUS
Status: {logistics_response.get('restocking_status', 'N/A')}
{logistics_response.get('restocking_details', 'N/A')}

---

LOGISTICS ALERTS ({len(logistics_response.get('alerts', []))})
{logistics_alerts_text}

---

This monitoring update combines operational metrics and inventory management to give you a complete view of your supplier's performance. Request another update anytime to see the latest conditions.
    """

    # Send combined monitoring response back to user
    response = ChatMessage(
        timestamp="",
        msg_id=uuid4(),
        content=[
            TextContent(type="text", text=response_text.strip()),
            EndSessionContent(type="end-session"),
        ],
    )

    await ctx.send(user_sender, response)

    log_message_transmission(
        ctx,
        "SENT",
        "ChatMessage",
        str(response.msg_id),
        {
            "type": "monitoring_update",
            "performance": demand_response.get("overall_performance"),
        },
    )

    # Clean up session and pending responses
    active_sessions = ctx.storage.get("active_sessions") or {}
    active_sessions.pop(request_id, None)
    ctx.storage.set("active_sessions", active_sessions)

    pending_responses.pop(request_id, None)
    ctx.storage.set("pending_responses", pending_responses)

    ctx.logger.info(f"Session cleaned up for request {request_id}")
    ctx.logger.info("=" * 70)


async def check_and_send_combined_response(ctx: Context, request_id: str):
    """Check if both responses are received, combine them, and send to user"""
    pending_responses = ctx.storage.get("pending_responses") or {}

    if request_id not in pending_responses:
        ctx.logger.warning(f"No pending response for request {request_id}")
        return

    response_data = pending_responses[request_id]
    compliance_response = response_data.get("compliance_response")
    financial_response = response_data.get("financial_response")

    # Check if we have BOTH responses
    if compliance_response is None or financial_response is None:
        ctx.logger.info(f"Still waiting for responses...")
        ctx.logger.info(
            f"Compliance: {'RECEIVED' if compliance_response else 'PENDING'}"
        )
        ctx.logger.info(f"Financial: {'RECEIVED' if financial_response else 'PENDING'}")
        return

    # We have both responses! Combine them
    user_sender = response_data.get("sender")
    user_query = response_data.get("user_query")

    if not user_sender:
        ctx.logger.error(f"No sender found for request {request_id}")
        return

    # Determine overall approval status - BOTH must pass (60 is the approval threshold)
    compliance_passed = compliance_response.get("compliance_score", 0) >= 60
    financial_passed = financial_response.get("financial_score", 0) >= 60
    overall_approved = compliance_passed and financial_passed

    # Build dynamic supplier requirements status
    passed_requirements = []
    failed_requirements = []

    if compliance_passed:
        passed_requirements.append("compliance")
    else:
        failed_requirements.append("compliance")

    if financial_passed:
        passed_requirements.append("financial")
    else:
        failed_requirements.append("financial")

    if overall_approved:
        requirements_text = (
            f"Supplier meets all requirements: {', '.join(passed_requirements)}"
        )
        # Store the approved supplier for monitoring
        ctx.storage.set(
            "selected_supplier", compliance_response.get("supplier_name", "")
        )
    elif len(passed_requirements) > 0:
        requirements_text = f"Supplier meets {', '.join(passed_requirements)} but does not meet {', '.join(failed_requirements)}"
    else:
        requirements_text = "Supplier does not meet any requirements"

    # Build dynamic recommendation summary
    if overall_approved:
        approval_header = "APPROVED FOR PARTNERSHIP"
        recommendation_summary = f"Based on comprehensive analysis across compliance and financial factors, this supplier demonstrates strong alignment with your business values and meets all required thresholds for partnership consideration."
    else:
        approval_header = "NOT APPROVED FOR PARTNERSHIP"
        if len(failed_requirements) == 2:
            recommendation_summary = "This supplier requires significant improvements across both evaluation areas (compliance and financial) before being considered for partnership."
        else:
            failed_area = failed_requirements[0]
            recommendation_summary = f"The supplier shows strength in {passed_requirements[0]}, but {failed_area} concerns must be mitigated to proceed with partnership."

    # Get supplier name and related data
    supplier_name = compliance_response.get("supplier_name", "Unknown Supplier")
    supplier_country = financial_response.get("supplier_country", "Unknown")
    user_country = financial_response.get("user_country", "United States")

    # Extract industry/establishment from user query
    query_lower = (user_query or "").lower()
    establishment = "business"
    if "pizza" in query_lower:
        establishment = "pizza"
    elif "coffee" in query_lower:
        establishment = "coffee"
    elif "restaurant" in query_lower:
        establishment = "restaurant"
    elif "food" in query_lower:
        establishment = "food"
    elif "bakery" in query_lower:
        establishment = "bakery"
    elif "cafe" in query_lower:
        establishment = "cafe"
    elif "retail" in query_lower:
        establishment = "retail"

    # Format financial details cleanly without bullet points
    financial_details = financial_response.get("financial_details", "N/A")
    # Remove "LLM Analysis:" prefix if present
    financial_details = financial_details.replace(
        "LLM Analysis:", f"{supplier_name} Financial Overview:"
    )

    # Format risk factors - include even small ones
    risk_factors = financial_response.get("risk_factors", [])
    if risk_factors:
        risk_factors_text = "\n".join([f"  - {factor}" for factor in risk_factors])
    else:
        risk_factors_text = "  - No significant financial risks identified"

    # Format violations - include even if minimal
    violations = compliance_response.get("violations", [])
    violations_count = len(violations)
    if violations:
        violations_text = "\n".join([f"  - {v}" for v in violations])
    else:
        violations_text = "  - None identified"

    # Build brief summary based on approval status
    compliance_score = compliance_response.get("compliance_score", 0)
    financial_score = financial_response.get("financial_score", 0)

    if overall_approved:
        brief_summary = f"""
Why {supplier_name} is a good fit for your {establishment} business:

{supplier_name} demonstrates strong alignment with your business needs. With a compliance score of {compliance_score}/100, they show commitment to ethical practices and sustainability. Their financial stability (score: {financial_score}/100) suggests reliable partnership potential, especially for {user_country}-{supplier_country} trade relations, making them a dependable supplier choice for your {establishment} establishment.
"""
    else:
        areas_of_concern = []
        if not compliance_passed:
            areas_of_concern.append(f"compliance ({compliance_score}/100)")
        if not financial_passed:
            areas_of_concern.append(f"financial ({financial_score}/100)")

        brief_summary = f"""
Why {supplier_name} may not be ideal for your {establishment} business at this time:

While {supplier_name} shows some positive attributes, there are concerns in {', '.join(areas_of_concern)} that should be addressed. For your {establishment} business operating from {user_country}, these factors could impact reliability and partnership success. We recommend exploring alternative suppliers or requesting additional information from {supplier_name} to address these concerns before proceeding.
"""

    response_text = f"""
{approval_header} with your {establishment} business

Supplier Analysis Report for your {establishment} business

{supplier_name} Status: {'Approved' if overall_approved else 'Not Approved'}

Evaluation Summary: {requirements_text}

---

{supplier_name} Compliance Overview

Overall Compliance Score: {compliance_response.get('compliance_score')}/100

Ethics & Worker Treatment:
{compliance_response.get('ethics_info', 'N/A')}

Sustainability Practices:
{compliance_response.get('sustainability_info', 'N/A')}

Violations Found: {violations_count}
{violations_text}

---

{supplier_name} Financial Analysis

Financial Score: {financial_response.get('financial_score')}/100 (Higher = Lower Risk)

Operating & Cost Assessment:
{financial_details}

Financial Risk Factors Identified: {len(risk_factors)}
{risk_factors_text}

---

Recommendation

{recommendation_summary}

---

Summary
{brief_summary}
    """

    # Send combined response back to user via chat
    response = ChatMessage(
        timestamp="",
        msg_id=uuid4(),
        content=[
            TextContent(type="text", text=response_text.strip()),
            EndSessionContent(type="end-session"),
        ],
    )

    await ctx.send(user_sender, response)

    log_message_transmission(
        ctx,
        "SENT",
        "ChatMessage",
        str(response.msg_id),
        {"type": "combined_result", "approved": overall_approved},
    )

    # Clean up session and pending responses
    active_sessions = ctx.storage.get("active_sessions") or {}
    active_sessions.pop(request_id, None)
    ctx.storage.set("active_sessions", active_sessions)

    pending_responses.pop(request_id, None)
    ctx.storage.set("pending_responses", pending_responses)

    ctx.logger.info(f"Session cleaned up for request {request_id}")
    ctx.logger.info("=" * 70)


@chat_proto.on_message(ChatAcknowledgement)
async def handle_acknowledgement(ctx: Context, sender: str, msg: ChatAcknowledgement):
    ctx.logger.info(
        f"Received acknowledgement from {sender} for message: {msg.acknowledged_msg_id}"
    )


supplier_orchestrator.include(chat_proto, publish_manifest=True)
supplier_orchestrator.include(find_supplier_protocol, publish_manifest=True)
supplier_orchestrator.include(compliance_protocol, publish_manifest=True)
supplier_orchestrator.include(financial_protocol, publish_manifest=True)
supplier_orchestrator.include(demand_protocol, publish_manifest=True)
supplier_orchestrator.include(logistics_protocol, publish_manifest=True)

if __name__ == "__main__":
    supplier_orchestrator.run()
