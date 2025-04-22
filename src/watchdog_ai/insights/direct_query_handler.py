"""
Direct Query Handler

This module implements the logic for handling direct queries to extract specific metrics from data.
It specifically addresses the fix for extracting days_to_close values for specific sales reps.
"""

import re
import re
import pandas as pd
import streamlit as st
import logging
import datetime
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple, Set, Callable, NamedTuple
from functools import lru_cache
from Levenshtein import distance
from scipy import stats
from sklearn.linear_model import LinearRegression
from scipy.spatial.distance import cosine
from ..models.query_models import QueryContext, QueryResult, IntentSchema, TimeRange
from ..insights.context import InsightExecutionContext
from .utils import sanitize_text, format_numeric_value
logger = logging.getLogger(__name__)

# Enhanced regex patterns for entity extraction
SALES_REP_PATTERNS = [
    r'(?:sales\s*(?:rep|representative|agent|person|consultant))?\s*(?:named|called)?\s*["\']?([A-Z][a-z]+\s+[A-Z][a-z]+)["\']?',
    r'(?:for|by)\s+["\']?([A-Z][a-z]+\s+[A-Z][a-z]+)["\']?(?:\s*(?:\'s|s\'|s))?(?:\s*(?:sales|performance|numbers|metrics|results))?',
    r'["\']([A-Z][a-z]+\s+[A-Z][a-z]+)["\'](?:\s*(?:\'s|s\'|s))?(?:\s*(?:sales|performance|numbers|metrics|results))?'
]

# Product entity patterns
PRODUCT_PATTERNS = [
    r'(?:product|item|offering|solution)?\s*(?:named|called)?\s*["\']?([A-Za-z0-9\s\-\.]+)["\']?',
    r'(?:for|about)\s+(?:product|item)?\s*["\']?([A-Za-z0-9\s\-\.]+)["\']?',
    r'(?:product|item)\s*(?:line|category|group)?\s*["\']?([A-Za-z0-9\s\-\.]+)["\']?'
]

# Region entity patterns
REGION_PATTERNS = [
    r'(?:region|area|territory|zone|market)?\s*(?:of|in|for)?\s*["\']?([A-Za-z\s\-\.]+)["\']?',
    r'(?:in|for|across)\s+(?:the)?\s*(?:region|area|territory|zone|market)?\s*(?:of)?\s*["\']?([A-Za-z\s\-\.]+)["\']?',
    r'(?:regional|territorial)\s*(?:data|performance|metrics|results)?\s*(?:for|in)?\s*["\']?([A-Za-z\s\-\.]+)["\']?'
]

# Customer segment patterns
CUSTOMER_SEGMENT_PATTERNS = [
    r'(?:customer|client|consumer)?\s*(?:segment|group|category|type)?\s*["\']?([A-Za-z\s\-\.]+)["\']?',
    r'(?:for|among|in)\s+(?:the)?\s*(?:customer|client|consumer)?\s*(?:segment|group|category|type)?\s*["\']?([A-Za-z\s\-\.]+)["\']?',
    r'(?:segmented|categorized)\s*(?:data|performance|metrics|results)?\s*(?:for|among)?\s*["\']?([A-Za-z\s\-\.]+)["\']?'
]

# Temporal patterns for date extraction
TEMPORAL_PATTERNS = {
    'absolute_date': [
        r'(?:on|at|for|during)\s+([A-Za-z]+\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4})',  # January 1st, 2023
        r'(?:on|at|for|during)\s+(\d{1,2}/\d{1,2}/\d{2,4})',  # MM/DD/YYYY
        r'(?:on|at|for|during)\s+(\d{4}-\d{2}-\d{2})'  # YYYY-MM-DD
    ],
    'relative_date': [
        r'(today|yesterday|tomorrow)',
        r'(last|this|next)\s+(week|month|quarter|year)',
        r'(last|previous|past)\s+(\d+)\s+(day|week|month|year)s?',
        r'(\d+)\s+(day|week|month|year)s?\s+ago'
    ],
    'date_range': [
        r'(?:from|between)\s+([^\s]+)\s+(?:to|and|through|until)\s+([^\s]+)',
        r'(?:in|during|for|over)\s+(?:the\s+)?(last|past|previous|next|coming)\s+(\d+)\s+(day|week|month|year)s?',
        r'(?:year|month|quarter)\s+to\s+date',
        r'(ytd|mtd|qtd)',
        r'(q[1-4])',
        r'(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)'
    ]
}

# Entity type patterns dictionary
ENTITY_PATTERNS = {
    'sales_rep': SALES_REP_PATTERNS,
    'product': PRODUCT_PATTERNS,
    'region': REGION_PATTERNS,
    'customer_segment': CUSTOMER_SEGMENT_PATTERNS
}
# Metric type patterns
METRIC_PATTERNS = {
    # Basic metrics (existing)
    'days_to_close': [
        r'(?:average\s*)?days?\s*(?:to|til|until|for)?\s*(?:close|closing)',
        r'(?:average\s*)?(?:close|closing)\s*(?:time|days|period)',
        r'how\s*(?:long|many)\s*days?(?:\s+(?:does|did)\s+it\s+take)?',
        r'how\s*(?:long|many\s*days).*?to\s*close'
    ],
    'total_sales': [
        r'(?:total|overall|all)\s*sales',
        r'number\s*of\s*(?:total\s*)?sales',
        r'sales\s*(?:amount|number|count|total|volume)'
    ],
    'conversion_rate': [
        r'(?:lead\s*)?(?:conversion|close)\s*(?:rate|percentage|ratio)',
        r'percentage\s*of\s*(?:leads|prospects)?\s*(?:converted|closed)'
    ],
    'revenue': [
        r'(?:total|overall)?\s*revenue',
        r'(?:total|overall)?\s*(?:income|earnings)',
        r'how\s*much\s*(?:revenue|money|income)'
    ],
    
    # Trend analysis patterns (new)
    'trend': [
        r'(?:sales|revenue|performance)\s*trend',
        r'(?:growth|decline)\s*(?:pattern|rate|trend)',
        r'(?:how|what)(?:\s+is|\s+has\s+been)?\s+(?:the)?\s*(?:trend|trajectory|direction)',
        r'(?:upward|downward)\s*(?:trend|movement|pattern)',
        r'(?:over\s+time|time\s+series)\s*(?:analysis|pattern|trend)',
        r'(?:historical|past)\s*(?:performance|pattern|trend)'
    ],
    'growth_rate': [
        r'(?:growth|increase|decrease)\s*rate',
        r'(?:percentage|rate)\s*(?:growth|increase|decrease|change)',
        r'(?:how\s+fast|how\s+quickly|at\s+what\s+rate)',
        r'(?:year[\s-]over[\s-]year|month[\s-]over[\s-]month|quarter[\s-]over[\s-]quarter)\s*(?:growth|change)',
        r'(?:annual|monthly|quarterly)\s*(?:growth|rate\s+of\s+change)'
    ],
    
    # Ratio-based metrics (new)
    'profit_margin': [
        r'(?:profit|gross|net)\s*margin',
        r'(?:margin|percentage|ratio)\s*(?:of|on)\s*(?:profit|revenue|sales)',
        r'(?:how\s+profitable|what\s+percentage\s+of\s+profit)',
        r'(?:return\s+on)\s*(?:sales|revenue)',
        r'(?:profit\s+to\s+revenue|revenue\s+to\s+profit)\s*(?:ratio|percentage)'
    ],
    'cost_ratio': [
        r'(?:cost|expense)\s*(?:ratio|percentage)',
        r'(?:cost\s+to\s+revenue|revenue\s+to\s+cost)\s*(?:ratio|percentage)',
        r'(?:cost\s+as\s+a\s+percentage\s+of|percentage\s+of\s+cost\s+to)\s*(?:sales|revenue)',
        r'(?:operating|overhead)\s*(?:cost|expense)\s*(?:ratio|percentage)',
        r'(?:cost|expense)\s*(?:efficiency|effectiveness)'
    ],
    
    # Comparative metrics (new)
    'comparison': [
        r'(?:compare|comparison|versus|vs\.?|against)',
        r'(?:difference|gap|delta)\s*(?:between|from|to)',
        r'(?:how\s+does|how\s+is)\s+.+\s+(?:compare|stack\s+up|match\s+up|different)',
        r'(?:better|worse|higher|lower|more|less)\s+than',
        r'(?:outperform|underperform|exceed|fall\s+short\s+of)'
    ],
    'year_over_year': [
        r'(?:year[\s-]over[\s-]year|yoy|year\s+to\s+year)',
        r'(?:compared\s+to|versus|against)\s+(?:last|previous)\s+year',
        r'(?:annual\s+comparison|yearly\s+comparison)',
        r'(?:how\s+does)\s+.+\s+(?:compare)\s+(?:to|with)\s+last\s+year',
        r'(?:this\s+year)\s+(?:versus|compared\s+to)\s+(?:last|previous)\s+year'
    ],
    
    # Statistical metrics (new)
    'variance': [
        r'(?:variance|variation|variability)',
        r'(?:how\s+(?:variable|consistent|stable|volatile))',
        r'(?:spread|distribution|range)',
        r'(?:statistical\s+variation|statistical\s+spread)',
        r'(?:deviation\s+from|difference\s+from)\s+(?:average|mean|median)'
    ],
    'standard_deviation': [
        r'(?:standard\s+deviation|std\s+dev|stdev)',
        r'(?:measure\s+of\s+spread|measure\s+of\s+variability)',
        r'(?:dispersion|scatter|spread)\s+(?:of|in)\s+(?:the\s+data|values|results)',
        r'(?:how\s+(?:spread\s+out|dispersed|scattered))',
        r'(?:statistical\s+dispersion|statistical\s+spread)'
    ],
    
    # Composite metrics (new)
    'performance_index': [
        r'(?:performance|productivity|efficiency)\s*(?:index|score|rating)',
        r'(?:overall|composite|aggregate)\s*(?:performance|score|rating)',
        r'(?:kpi|key\s+performance\s+indicator)',
        r'(?:performance\s+metric|performance\s+measure)',
        r'(?:how\s+(?:well|poorly|effectively))\s+(?:performing|doing)'
    ],
    'efficiency_score': [
        r'(?:efficiency|productivity)\s*(?:score|rating|level)',
        r'(?:how\s+efficient|efficiency\s+level)',
        r'(?:output\s+per\s+input|input\s+to\s+output\s+ratio)',
        r'(?:resource\s+utilization|resource\s+efficiency)',
        r'(?:time\s+efficiency|cost\s+efficiency)'
    ]
}

# Metric relationships and compatibility
METRIC_RELATIONSHIPS = {
    'revenue': {
        'compatible_with': ['profit_margin', 'cost_ratio', 'growth_rate', 'trend', 'comparison', 'year_over_year', 'variance', 'standard_deviation'],
        'derived_metrics': {
            'profit': lambda df, rev_col, cost_col: df[rev_col] - df[cost_col] if cost_col in df.columns else None,
            'profit_margin': lambda df, rev_col, cost_col: ((df[rev_col] - df[cost_col]) / df[rev_col]) * 100 if cost_col in df.columns else None
        },
        'compatible_entities': ['sales_rep', 'product', 'region', 'customer_segment'],
        'benchmark_type': 'higher_better'
    },
    'days_to_close': {
        'compatible_with': ['trend', 'comparison', 'variance', 'standard_deviation'],
        'derived_metrics': {
            'efficiency': lambda df, days_col: 100 / df[days_col] if days_col in df.columns else None
        },
        'compatible_entities': ['sales_rep', 'product', 'region', 'customer_segment'],
        'benchmark_type': 'lower_better'
    },
    'total_sales': {
        'compatible_with': ['trend', 'growth_rate', 'comparison', 'year_over_year', 'variance'],
        'derived_metrics': {
            'average_sale_value': lambda df, sales_col, rev_col: df[rev_col] / df[sales_col] if sales_col in df.columns and rev_col in df.columns else None
        },
        'compatible_entities': ['sales_rep', 'product', 'region', 'customer_segment'],
        'benchmark_type': 'higher_better'
    },
    'conversion_rate': {
        'compatible_with': ['trend', 'comparison', 'year_over_year'],
        'derived_metrics': {},
        'compatible_entities': ['sales_rep', 'product', 'region', 'customer_segment'],
        'benchmark_type': 'higher_better'
    },
    'trend': {
        'compatible_with': ['growth_rate', 'year_over_year'],
        'derived_metrics': {},
        'requires_time_data': True,
        'compatible_entities': ['sales_rep', 'product', 'region', 'customer_segment'],
        'benchmark_type': 'direction'
    },
    'growth_rate': {
        'compatible_with': ['trend', 'year_over_year', 'comparison'],
        'derived_metrics': {},
        'requires_time_data': True,
        'compatible_entities': ['sales_rep', 'product', 'region', 'customer_segment'],
        'benchmark_type': 'higher_better'
    }
}

# Metric validation functions
def validate_metric_entity_compatibility(metric_type: str, entity_type: str) -> bool:
    """
    Validate if a metric type is compatible with an entity type.
    
    Args:
        metric_type: The type of metric to validate
        entity_type: The type of entity to validate against
        
    Returns:
        Boolean indicating if the metric is compatible with the entity
    """
    # Default to True for basic metrics
    if metric_type not in METRIC_RELATIONSHIPS:
        return True
        
    relationship = METRIC_RELATIONSHIPS.get(metric_type, {})
    compatible_entities = relationship.get('compatible_entities', [])
    
    # If no specific compatible entities are listed, assume it's compatible with all
    if not compatible_entities:
        return True
        
    return entity_type in compatible_entities

def validate_metric_data_availability(metric_type: str, filtered_data: pd.DataFrame) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Validate if the data needed for a metric is available in the DataFrame.
    
    Args:
        metric_type: The type of metric to validate
        filtered_data: The filtered DataFrame to check
        
    Returns:
        Tuple of (is_valid, message, required_columns)
    """
    if filtered_data.empty:
        return False, "No data available for analysis", {}
        
    # Map metric types to required columns
    metric_columns = {
        'days_to_close': ['days_to_close', 'close_time', 'closing_time', 'days', 'cycle_time', 'sales_cycle'],
        'total_sales': ['sales_count', 'num_sales', 'number_of_sales', 'deals_closed', 'transactions'],
        'conversion_rate': ['conversion_rate', 'close_rate', 'success_rate', 'win_rate', 'conversion_percentage'],
        'revenue': ['revenue', 'sales', 'income', 'earnings', 'amount', 'sales_amount', 'deal_value'],
        'profit_margin': ['profit', 'margin', 'profit_margin', 'gross_margin', 'net_margin'],
        'cost_ratio': ['cost', 'expense', 'expenses', 'cost_ratio', 'cost_percentage'],
        'trend': ['date', 'time', 'created_at', 'timestamp', 'period'],
        'growth_rate': ['date', 'time', 'created_at', 'timestamp', 'period'],
'year_over_year': ['date', 'time', 'created_at'],

def handle_direct_query_ui():
    """Handle direct queries in the Streamlit UI."""
    st.header("Direct Data Analysis")
    st.write("Ask questions about your data in natural language.")
    
    # Check if data is loaded
    if 'validated_data' not in st.session_state:
        st.warning("Please upload data in the Data Upload tab first.")
        return
    
    # Query input
    query = st.text_input("Enter your question:", 
                         placeholder="Example: What is the average days to close for sales rep Karen Davis?")
    
    if query:
        try:
            # Create execution context
            context = InsightExecutionContext(
                df=st.session_state.validated_data,
                query=query,
                user_role=st.session_state.get('user_role', 'analyst')
            )
            
            # Create query context
            query_context = QueryContext(
                query=query,
                insight_context=context
            )
            
            # Process query
            result = process_query(query_context)
            
            # Display results
            if result.success:
                st.success(result.message)
                
                # Display metrics for each entity
                for entity, metric_data in result.metrics.items():
                    st.metric(
                        label=f"{entity} - {metric_data['metric_type'].replace('_', ' ').title()}", 
                        value=metric_data['formatted']
                    )
            else:
                st.error(result.message)
                
            # Add to query history
            if query not in st.session_state.query_history:
                st.session_state.query_history.append(query)
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}", exc_info=True)
            st.error(f"Error processing query: {str(e)}")
    
    # Show query history
    if st.session_state.query_history:
        with st.expander("Query History"):
            for past_query in st.session_state.query_history:
                st.write(past_query)

def extract_temporal_context(query: str) -> Optional[TimeRange]:
    """
    Extract temporal context (dates, date ranges, relative time periods) from the query.
    
    Args:
        query: The user query string
        
    Returns:
        TimeRange object if temporal information is found, None otherwise
    """
    time_range = TimeRange()
    
    # Check for absolute dates
    for pattern in TEMPORAL_PATTERNS['absolute_date']:
        matches = re.finditer(pattern, query, re.IGNORECASE)
        for match in matches:
            date_str = match.group(1).strip()
            try:
                # Try multiple date formats
                for fmt in ['%B %d, %Y', '%B %dst, %Y', '%B %dnd, %Y', '%B %drd, %Y', '%B %dth, %Y', 
                           '%m/%d/%Y', '%m/%d/%y', '%Y-%m-%d']:
                    try:
                        parsed_date = datetime.datetime.strptime(date_str, fmt)
                        if not time_range.start_date:
                            time_range.start_date = parsed_date
                        elif not time_range.end_date:
                            time_range.end_date = parsed_date
                        break
                    except ValueError:
                        continue
            except Exception as e:
                logger.warning(f"Failed to parse date '{date_str}': {str(e)}")
    
    # Check for relative dates
    for pattern in TEMPORAL_PATTERNS['relative_date']:
        matches = re.finditer(pattern, query, re.IGNORECASE)
        for match in matches:
            if match.group(1).lower() in ['today', 'yesterday', 'tomorrow']:
                period = match.group(1).lower()
                time_range.period = period
            elif match.group(1).lower() in ['last', 'this', 'next']:
                period_type = match.group(1).lower()
                period_unit = match.group(2).lower()
                time_range.period = f"{period_type}_{period_unit}"
            elif match.groups()[-1] in ['day', 'week', 'month', 'year']:
                # Handle "X days/weeks/months/years ago" or "last X days/weeks/months/years"
                quantity = match.group(2) if len(match.groups()) > 2 else match.group(1)
                unit = match.groups()[-1]
                time_range.period = f"last_{quantity}_{unit}s"
    
    # Check for date ranges
    for pattern in TEMPORAL_PATTERNS['date_range']:
        matches = re.finditer(pattern, query, re.IGNORECASE)
        for match in matches:
            if 'from' in pattern or 'between' in pattern:
                # Handle explicit ranges like "from X to Y"
                if len(match.groups()) >= 2:
                    time_range.period = "custom_range"
                    # Actual dates would be processed in absolute_date patterns
            elif 'year to date' in query.lower() or 'ytd' in query.lower():
                time_range.period = "ytd"
            elif 'month to date' in query.lower() or 'mtd' in query.lower():
                time_range.period = "mtd"
            elif 'quarter to date' in query.lower() or 'qtd' in query.lower():
                time_range.period = "qtd"
            elif match.group(1).lower() in ['q1', 'q2', 'q3', 'q4']:
                time_range.period = match.group(1).lower()
            elif match.group(1).lower() in ['jan', 'january', 'feb', 'february', 'mar', 'march',
                                          'apr', 'april', 'may', 'jun', 'june', 'jul', 'july',
                                          'aug', 'august', 'sep', 'sept', 'september', 'oct', 'october',
                                          'nov', 'november', 'dec', 'december']:
                month = match.group(1).lower()
                # Normalize month names
                if month.startswith('jan'): month = 'january'
                elif month.startswith('feb'): month = 'february'
                elif month.startswith('mar'): month = 'march'
                elif month.startswith('apr'): month = 'april'
                elif month.startswith('jun'): month = 'june'
                elif month.startswith('jul'): month = 'july'
                elif month.startswith('aug'): month = 'august'
                elif month.startswith('sep'): month = 'september'
                elif month.startswith('oct'): month = 'october'
                elif month.startswith('nov'): month = 'november'
                elif month.startswith('dec'): month = 'december'
                time_range.period = month
    
    # If we have any temporal information, return the TimeRange
    if time_range.start_date or time_range.end_date or time_range.period:
        return time_range
    
    return None

@lru_cache(maxsize=100)
def get_cached_entity_list(entity_type: str, df: pd.DataFrame) -> Set[str]:
    """
    Get a cached list of valid entity values from the dataframe.
    
    Args:
        entity_type: The type of entity (e.g., 'sales_rep', 'product')
        df: The dataframe containing the data
        
    Returns:
        Set of valid entity values
    """
    if df is None or df.empty:
        return set()
    
    # Map entity types to potential column names
    entity_columns = {
        'sales_rep': ['sales_rep', 'rep', 'agent', 'salesperson', 'representative', 'sales_representative', 'employee'],
        'product': ['product', 'item', 'product_name', 'sku', 'offering', 'solution', 'product_id'],
        'region': ['region', 'area', 'territory', 'zone', 'market', 'location', 'geography'],
        'customer_segment': ['segment', 'customer_segment', 'customer_type', 'client_type', 'category', 'customer_category']
    }
    
    # Find matching columns
    entity_cols = []
    if entity_type in entity_columns:
        col_patterns = entity_columns[entity_type]
        entity_cols = [col for col in df.columns if any(pat.lower() in col.lower() for pat in col_patterns)]
    
    if not entity_cols:
        return set()
    
    # Get unique values
    values = set()
    for col in entity_cols:
        values.update(df[col].dropna().astype(str).unique())
    
    return values
def fuzzy_match_entity(
    candidate: str, 
    entity_type: str, 
    df: pd.DataFrame, 
    threshold: float = 0.7
) -> Tuple[Optional[str], float]:
    """
    Find the closest match for a candidate entity using fuzzy matching.
    
    Args:
        candidate: The candidate entity string to match
        entity_type: The type of entity (e.g., 'sales_rep', 'product')
        df: The dataframe containing valid entities
        threshold: Minimum similarity threshold (0-1) to consider a match
        
    Returns:
        Tuple of (best_match, confidence_score) or (None, 0) if no good match
    """
    if not candidate or df is None or df.empty:
        return None, 0.0
    
    # Get valid entities from cache
    valid_entities = get_cached_entity_list(entity_type, df)
    if not valid_entities:
        return None, 0.0
    
    # Convert candidate to lowercase for case-insensitive matching
    candidate_lower = candidate.lower()
    
    # Perfect match check
    for entity in valid_entities:
        if entity.lower() == candidate_lower:
            return entity, 1.0
    
    # Calculate similarity for each entity
    best_match = None
    best_score = 0.0
    
    for entity in valid_entities:
        # Levenshtein distance is lower for more similar strings
        # Convert to similarity score (0-1) where 1 is perfect match
        max_len = max(len(candidate), len(entity))
        if max_len == 0:  # Handle empty strings
            continue
            
        lev_dist = distance(candidate_lower, entity.lower())
        similarity = 1.0 - (lev_dist / max_len)
        
        if similarity > best_score:
            best_score = similarity
            best_match = entity
    
    # Only return a match if it meets the threshold
    if best_score >= threshold:
        return best_match, best_score
    
    return None, 0.0

def extract_entities(query: str, context: Optional[InsightExecutionContext] = None) -> Dict[str, List[Dict[str, Any]]]:
    """
    Extract entity names from the query using enhanced regex patterns with fuzzy matching.
    
    Args:
        query: The user query string
        context: Optional insight context containing dataframe for entity validation
        
    Returns:
        Dictionary of entity types and their extracted values with confidence scores
    """
    # Initialize empty result with all supported entity types
    entities = {
        "sales_rep": [],
        "product": [],
        "region": [],
        "customer_segment": []
    }
    
    # Extract time range if available
    time_range = extract_temporal_context(query)
    
    # For each entity type, extract matches using patterns
    for entity_type, patterns in ENTITY_PATTERNS.items():
        for pattern in patterns:
            matches = re.finditer(pattern, query, re.IGNORECASE)
            for match in matches:
                entity_value = match.group(1).strip()
                
                # Skip if empty or too short
                if not entity_value or len(entity_value) < 2:
                    continue
                
                # Basic validation to skip common false positives
                if entity_value.lower() in ['the', 'for', 'and', 'with', 'by', 'from', 'to']:
                    continue
                
                # Check if this entity already exists in the results
                entity_exists = any(e['value'].lower() == entity_value.lower() for e in entities[entity_type])
                if entity_exists:
                    continue
                
                entity_info = {
                    "value": entity_value,
                    "confidence": 0.8,  # Base confidence from regex match
                    "validated": False
                }
                
                # If context is provided, attempt fuzzy matching against known entities
                if context and context.df is not None and not context.df.empty:
                    matched_entity, confidence = fuzzy_match_entity(
                        entity_value, 
                        entity_type, 
                        context.df
                    )
                    
                    if matched_entity and confidence > 0.7:
                        entity_info["value"] = matched_entity
                        entity_info["confidence"] = confidence
                        entity_info["validated"] = True
                        entity_info["original_value"] = entity_value  # Keep track of what was actually mentioned
                    else:
                        # Downgrade confidence if the entity couldn't be validated
                        entity_info["confidence"] *= 0.7
                
                entities[entity_type].append(entity_info)
    
    # Remove empty entity types
    entities = {k: v for k, v in entities.items() if v}
    
    # If no entities were found but we have a dataframe, we can try more aggressive fuzzy matching
    if context and context.df is not None and not any(entities.values()):
        # Extract potential entity mentions by looking for capitalized words or quoted text
        potential_entities = re.findall(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)|(?:"([^"]+)")|(?:\'([^\']+)\')', query)
        for match in potential_entities:
            for match_part in match:  # Handle the different capture groups
                if match_part:
                    # Try to match this against each entity type
                    for entity_type in ENTITY_PATTERNS.keys():
                        matched_entity, confidence = fuzzy_match_entity(
                            match_part, 
                            entity_type, 
                            context.df,
                            threshold=0.8  # Higher threshold for this fallback method
                        )
                        
                        if matched_entity and confidence > 0.8:
                            entity_info = {
                                "value": matched_entity,
                                "confidence": confidence * 0.9,  # Slightly lower confidence since this is a fallback
                                "validated": True,
                                "original_value": match_part
                            }
                            entities.setdefault(entity_type, []).append(entity_info)
    
    return entities

def identify_metric_type(query: str) -> Optional[str]:
    """
    Identify the type of metric being requested in the query.
    
    Args:
        query: The user query string
        
    Returns:
        The identified metric type or None if no match
    """
    for metric_type, patterns in METRIC_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return metric_type
                
    return None

def extract_days_to_close(data: pd.DataFrame, sales_rep: str) -> Optional[float]:
    """
    Extract the days_to_close value for a specific sales rep.
    
    Args:
        data: The dataframe containing sales data
        sales_rep: The name of the sales rep
        
    Returns:
        The days_to_close value for the specified sales rep or None if not found
    """
    if data is None or data.empty:
        return None
        
    # Fix for proper case-insensitive comparison for names like "Karen Davis"
    sales_rep_lower = sales_rep.lower()
    
    # Handle potential column name variations
    rep_columns = [col for col in data.columns if any(x in col.lower() for x in ['rep', 'agent', 'salesperson', 'representative'])]
    
    if not rep_columns:
        logger.warning(f"No sales rep column found in data with columns: {data.columns}")
        return None
        
    rep_column = rep_columns[0]
    
    # Ensure we have days_to_close column
    days_columns = [col for col in data.columns if any(x in col.lower() for x in ['days', 'days_to_close', 'closing_time'])]
    
    if not days_columns:
        logger.warning(f"No days_to_close column found in data with columns: {data.columns}")
        return None
        
    days_column = days_columns[0]
    
    # Apply case-insensitive filtering for the specific sales rep
    filtered_data = data[data[rep_column].str.lower() == sales_rep_lower]
    
    if filtered_data.empty:
        logger.warning(f"No data found for sales rep '{sales_rep}'")
        return None
        
    # Calculate the average days_to_close for the specific sales rep
    days_value = filtered_data[days_column].mean()
    
    return days_value
def process_metric_for_entity(
    metric_type: str,
    entity_type: str,
    entity_value: str,
    context: InsightExecutionContext
) -> Tuple[Optional[float], Optional[str]]:
    """
    Process a specific metric for a given entity.
    
    Args:
        metric_type: The type of metric to extract
        entity_type: The type of entity (e.g., 'sales_rep')
        entity_value: The specific entity value (e.g., 'Karen Davis')
        context: The insight context containing data
        
    Returns:
        Tuple of (metric_value, formatted_value)
    """
    if context.df is None or not metric_type:
        return None, None
        
    data = context.df
    
    # Map entity types to potential column names
    entity_columns = {
        'sales_rep': ['sales_rep', 'rep', 'agent', 'salesperson', 'representative', 'sales_representative', 'employee'],
        'product': ['product', 'item', 'product_name', 'sku', 'offering', 'solution', 'product_id'],
        'region': ['region', 'area', 'territory', 'zone', 'market', 'location', 'geography'],
        'customer_segment': ['segment', 'customer_segment', 'customer_type', 'client_type', 'category', 'customer_category']
    }
    
    # Map metric types to potential column names
    metric_columns = {
        'days_to_close': ['days_to_close', 'close_time', 'closing_time', 'days', 'cycle_time', 'sales_cycle'],
        'total_sales': ['sales_count', 'num_sales', 'number_of_sales', 'deals_closed', 'transactions'],
        'conversion_rate': ['conversion_rate', 'close_rate', 'success_rate', 'win_rate', 'conversion_percentage'],
        'revenue': ['revenue', 'sales', 'income', 'earnings', 'amount', 'sales_amount', 'deal_value']
    }
    
    # Find matching columns for the entity type
    entity_cols = []
    if entity_type in entity_columns:
        col_patterns = entity_columns[entity_type]
        entity_cols = [col for col in data.columns if any(pat.lower() in col.lower() for pat in col_patterns)]
    
    if not entity_cols:
        logger.warning(f"No {entity_type} column found in data with columns: {data.columns}")
        return None, None
    
    entity_column = entity_cols[0]
    
    # Case-insensitive filtering for the specific entity
    entity_value_lower = entity_value.lower()
    filtered_data = data[data[entity_column].str.lower() == entity_value_lower]
    
    if filtered_data.empty:
        logger.warning(f"No data found for {entity_type} '{entity_value}'")
        return None, None
    
    # Handle temporal filtering if time_range is in the query context
    if hasattr(context, 'time_range') and context.time_range:
        try:
            date_columns = [col for col in filtered_data.columns if any(x in col.lower() for x in 
                          ['date', 'time', 'created', 'closed', 'timestamp'])]
            
            if date_columns:
                date_col = date_columns[0]
                
                # Ensure date column is in datetime format
                if pd.api.types.is_string_dtype(filtered_data[date_col]):
                    try:
                        filtered_data[date_col] = pd.to_datetime(filtered_data[date_col])
                    except Exception as e:
                        logger.warning(f"Failed to convert {date_col} to datetime: {str(e)}")
                
                # Apply date filters
                if context.time_range.start_date:
                    filtered_data = filtered_data[filtered_data[date_col] >= context.time_range.start_date]
                if context.time_range.end_date:
                    filtered_data = filtered_data[filtered_data[date_col] <= context.time_range.end_date]
                
                # Handle predefined periods
                if context.time_range.period:
                    period = context.time_range.period.lower()
                    today = datetime.datetime.now()
                    
                    if period == 'today':
                        start_date = today.replace(hour=0, minute=0, second=0, microsecond=0)
                        filtered_data = filtered_data[filtered_data[date_col] >= start_date]
                    elif period == 'yesterday':
                        start_date = (today - datetime.timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
                        end_date = today.replace(hour=0, minute=0, second=0, microsecond=0)
                        filtered_data = filtered_data[(filtered_data[date_col] >= start_date) & (filtered_data[date_col] < end_date)]
                    elif period == 'this_week':
                        start_date = (today - datetime.timedelta(days=today.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)
                        filtered_data = filtered_data[filtered_data[date_col] >= start_date]
                    elif period == 'last_week':
                        start_date = (today - datetime.timedelta(days=today.weekday() + 7)).replace(hour=0, minute=0, second=0, microsecond=0)
                        end_date = (today - datetime.timedelta(days=today.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)
                        filtered_data = filtered_data[(filtered_data[date_col] >= start_date) & (filtered_data[date_col] < end_date)]
                    elif period == 'this_month':
                        start_date = today.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
                        filtered_data = filtered_data[filtered_data[date_col] >= start_date]
                    elif period == 'last_month':
                        last_month = today.replace(day=1) - datetime.timedelta(days=1)
                        start_date = last_month.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
                        end_date = today.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
                        filtered_data = filtered_data[(filtered_data[date_col] >= start_date) & (filtered_data[date_col] < end_date)]
                    elif period == 'ytd' or period == 'year_to_date':
                        start_date = today.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
                        filtered_data = filtered_data[filtered_data[date_col] >= start_date]
                    # Add more period handling as needed
        
        except Exception as e:
            logger.warning(f"Error during temporal filtering: {str(e)}")
    
    # If filtered data is empty after temporal filtering
    if filtered_data.empty:
        logger.warning(f"No data found for {entity_type} '{entity_value}' after temporal filtering")
        return None, None
    
    # Process the metric based on metric type
    try:
        if metric_type == 'days_to_close':
            # Find the relevant column for days_to_close
            days_cols = [col for col in filtered_data.columns if any(pat.lower() in col.lower() for pat in metric_columns['days_to_close'])]
            
            if not days_cols:
                logger.warning(f"No days_to_close column found in data")
                return None, None
                
            days_column = days_cols[0]
            
            # Calculate the metric
            days_mean = filtered_data[days_column].mean()
            days_median = filtered_data[days_column].median()
            
            # Use the mean as the primary metric, but include median in formatted output
            metric_value = days_mean
            formatted_value = f"{format_numeric_value(days_mean, decimals=1)} days (median: {format_numeric_value(days_median, decimals=1)})"
            
            return metric_value, formatted_value
            
        elif metric_type == 'total_sales':
            # Check if we have sales count column
            sales_cols = [col for col in filtered_data.columns if any(pat.lower() in col.lower() for pat in metric_columns['total_sales'])]
            
            if sales_cols:
                sales_column = sales_cols[0]
                count_value = filtered_data[sales_column].sum()
                return count_value, format_numeric_value(count_value, decimals=0)
            else:
                # If no direct column, try to count rows (assuming each row is a sale)
                count_value = len(filtered_data)
                return count_value, format_numeric_value(count_value, decimals=0)
                
        elif metric_type == 'conversion_rate':
            # Find conversion rate column
            conv_cols = [col for col in filtered_data.columns if any(pat.lower() in col.lower() for pat in metric_columns['conversion_rate'])]
            
            if conv_cols:
                # Direct conversion rate column
                conv_column = conv_cols[0]
                conv_value = filtered_data[conv_column].mean()
                # Convert to percentage if not already
                if conv_value <= 1:
                    conv_value *= 100
                return conv_value, f"{format_numeric_value(conv_value, decimals=1)}%"
            else:
                # Try to calculate from opportunities and closed deals
                opp_cols = [col for col in filtered_data.columns if any(x in col.lower() for x in ['opportunity', 'opportunities', 'leads'])]
                win_cols = [col for col in filtered_data.columns if any(x in col.lower() for x in ['won', 'closed', 'success'])]
                
                if opp_cols and win_cols:
                    opp_col = opp_cols[0]
                    win_col = win_cols[0]
                    
                    # Check column types and calculate accordingly
                    if pd.api.types.is_numeric_dtype(filtered_data[opp_col]) and pd.api.types.is_numeric_dtype(filtered_data[win_col]):
                        total_opps = filtered_data[opp_col].sum()
                        total_wins = filtered_data[win_col].sum()
                        
                        if total_opps > 0:
                            conv_value = (total_wins / total_opps) * 100
                            return conv_value, f"{format_numeric_value(conv_value, decimals=1)}%"
                
                return None, None
                
        elif metric_type == 'revenue':
            # Find revenue column
            rev_cols = [col for col in filtered_data.columns if any(pat.lower() in col.lower() for pat in metric_columns['revenue'])]
            
            if not rev_cols:
                logger.warning(f"No revenue column found in data")
                return None, None
                
            rev_column = rev_cols[0]
            total_revenue = filtered_data[rev_column].sum()
            avg_revenue = filtered_data[rev_column].mean()
            
            metric_value = total_revenue
            formatted_value = f"${format_numeric_value(total_revenue, decimals=2)} (avg: ${format_numeric_value(avg_revenue, decimals=2)})"
            
            return metric_value, formatted_value
        
        # Handle other metric types here
        else:
            logger.warning(f"Unsupported metric type: {metric_type}")
            return None, None
            
    except Exception as e:
        logger.error(f"Error processing {metric_type} for {entity_type} '{entity_value}': {str(e)}", exc_info=True)
        return None, None

def find_similar_entities(
    entities: Dict[str, List[Dict[str, Any]]],
    metrics: Dict[str, Dict[str, Any]],
    metric_type: str,
    context: InsightExecutionContext
) -> List[Dict[str, Any]]:
    """
    Find entities similar to the queried entities based on metric performance.
    
    Args:
        entities: Dictionary of entities extracted from the query
        metrics: Dictionary of metrics calculated for the entities
        metric_type: The type of metric being analyzed
        context: The insight execution context
        
    Returns:
        List of similar entities with comparison information
    """
    if not entities or not metrics or not metric_type or context.df is None or context.df.empty:
        return []
    
    similar_entities = []
    data = context.df
    
    for entity_type, entity_list in entities.items():
        # Get entity column
        entity_cols = get_entity_columns(entity_type, data)
        if not entity_cols:
            continue
            
        entity_column = entity_cols[0]
        
        # Get metric column
        metric_cols = get_metric_columns(metric_type, data)
        if not metric_cols:
            continue
            
        metric_column = metric_cols[0]
        
        # Get all unique entities of this type
        all_entities = data[entity_column].dropna().unique()
        
        # For each entity in the query, find similar entities
        for entity_info in entity_list:
            entity_value = entity_info["value"]
            entity_key = f"{entity_type}:{entity_value}"
            
            if entity_key not in metrics:
                continue
                
            # Get the metric value for this entity
            entity_metric_value = metrics[entity_key]["value"]
            
            # Calculate metrics for all other entities of the same type
            other_entities_metrics = {}
            for other_entity in all_entities:
                if other_entity == entity_value:
                    continue
                    
                # Filter data for this entity
                other_entity_data = data[data[entity_column] == other_entity]
                if other_entity_data.empty:
                    continue
                    
                # Calculate metric value
                if metric_type == 'days_to_close':
                    other_metric_value = other_entity_data[metric_column].mean()
                elif metric_type == 'revenue':
                    other_metric_value = other_entity_data[metric_column].sum()
                elif metric_type == 'total_sales':
                    other_metric_value = len(other_entity_data)
                elif metric_type == 'conversion_rate':
                    other_metric_value = other_entity_data[metric_column].mean()
                else:
                    # Default to mean for unknown metrics
                    other_metric_value = other_entity_data[metric_column].mean()
                    
                if pd.isna(other_metric_value):
                    continue
                    
                other_entities_metrics[other_entity] = other_metric_value
            
            # Find the most similar entities based on metric value
            similar_entity_list = []
            benchmark_type = METRIC_RELATIONSHIPS.get(metric_type, {}).get('benchmark_type', 'higher_better')
            
            for other_entity, other_metric_value in other_entities_metrics.items():
                # Calculate similarity based on the metric difference
                # For values where higher is better, smaller differences to higher values are more similar
                # For values where lower is better, smaller differences to lower values are more similar
                if benchmark_type == 'higher_better':
                    if other_metric_value > entity_metric_value:
                        similarity = 1.0 - min(1.0, (other_metric_value - entity_metric_value) / max(1.0, entity_metric_value))
                        comparison = "better"
                    else:
                        similarity = 1.0 - min(1.0, (entity_metric_value - other_metric_value) / max(1.0, entity_metric_value))
                        comparison = "worse"
                elif benchmark_type == 'lower_better':
                    if other_metric_value < entity_metric_value:
                        similarity = 1.0 - min(1.0, (entity_metric_value - other_metric_value) / max(1.0, entity_metric_value))
                        comparison = "better"
                    else:
                        similarity = 1.0 - min(1.0, (other_metric_value - entity_metric_value) / max(1.0, entity_metric_value))
                        comparison = "worse"
                else:
                    # Default to absolute difference for other types
                    abs_diff = abs(entity_metric_value - other_metric_value)
                    similarity = 1.0 - min(1.0, abs_diff / max(1.0, abs(entity_metric_value)))
                    comparison = "different" if abs_diff > 0.1 * abs(entity_metric_value) else "similar"
                
                if similarity >= 0.7:  # Only include sufficiently similar entities
                    similar_entity_list.append({
                        "entity_type": entity_type,
                        "entity_value": other_entity,
                        "similarity_score": similarity,
                        "metric_value": other_metric_value,
                        "formatted_value": format_metric_value(other_metric_value, metric_type),
                        "comparison": comparison
                    })
            
            # Sort by similarity score and take top 3
            similar_entity_list.sort(key=lambda x: x["similarity_score"], reverse=True)
            similar_entities.extend(similar_entity_list[:3])
    
    return similar_entities

def calculate_statistical_significance(
    metrics: Dict[str, Dict[str, Any]],
    context: InsightExecutionContext,
    entities: Dict[str, List[Dict[str, Any]]]
) -> Dict[str, Any]:
    """
    Calculate statistical significance of differences between entities.
    
    Args:
        metrics: Dictionary of metrics calculated for the entities
        context: The insight execution context
        entities: Dictionary of entities extracted from the query
        
    Returns:
        Dictionary with statistical significance results
    """
    if not metrics or len(metrics) < 2 or context.df is None or context.df.empty:
        return {}
    
    significance_results = {}
    data = context.df
    
    # Group metrics by entity type
    entity_type_metrics = {}
    for entity_key, metric_data in metrics.items():
        entity_type, entity_value = entity_key.split(":", 1)
        if entity_type not in entity_type_metrics:
            entity_type_metrics[entity_type] = {}
        entity_type_metrics[entity_type][entity_value] = metric_data
    
    # For each entity type with multiple entities, calculate significance
    for entity_type, entity_metrics in entity_type_metrics.items():
        if len(entity_metrics) < 2:
            continue
            
        # Get entity column
        entity_cols = get_entity_columns(entity_type, data)
        if not entity_cols:
            continue
            
        entity_column = entity_cols[0]
        
        # Get the metric values and distributions for each entity
        entity_distributions = {}
        metric_type = next(iter(entity_metrics.values()))["metric_type"]
        metric_cols = get_metric_columns(metric_type, data)
        
        if not metric_cols:
            continue
            
        metric_column = metric_cols[0]
        
        for entity_value, metric_data in entity_metrics.items():
            # Get the raw data points for this entity
            entity_data = data[data[entity_column] == entity_value]
            if entity_data.empty or metric_column not in entity_data.columns:
                continue
                
            # Get the distribution of values
            value_distribution = entity_data[metric_column].dropna().tolist()
            if len(value_distribution) < 2:  # Need at least 2 points for t-test
                continue
                
            entity_distributions[entity_value] = value_distribution
        
        # Calculate pairwise t-tests
        if len(entity_distributions) < 2:
            continue
            
        pairwise_tests = {}
        entity_values = list(entity_distributions.keys())
        
        for i in range(len(entity_values)):
            for j in range(i+1, len(entity_values)):
                entity1 = entity_values[i]
                entity2 = entity_values[j]
                
                dist1 = entity_distributions[entity1]
                dist2 = entity_distributions[entity2]
                
                # Calculate t-test
                t_stat, p_value = stats.ttest_ind(dist1, dist2, equal_var=False)
                
                # Calculate effect size (Cohen's d)
                mean1, mean2 = np.mean(dist1), np.mean(dist2)
                std1, std2 = np.std(dist1), np.std(dist2)
                pooled_std = np.sqrt(((len(dist1) - 1) * std1**2 + (len(dist2) - 1) * std2**2) / 
                                    (len(dist1) + len(dist2) - 2))
                effect_size = abs(mean1 - mean2) / pooled_std if pooled_std > 0 else 0
                
                # Calculate confidence intervals
                ci1 = stats.t.interval(0.95, len(dist1)-1, loc=mean1, scale=stats.sem(dist1))
                ci2 = stats.t.interval(0.95, len(dist2)-1, loc=mean2, scale=stats.sem(dist2))
                
                # Determine if the difference is significant
                is_significant = p_value < 0.05
                
                # Add comparison info based on the benchmark type
                benchmark_type = METRIC_RELATIONSHIPS.get(metric_type, {}).get('benchmark_type', 'higher_better')
                if benchmark_type == 'higher_better':
                    better_entity = entity1 if mean1 > mean2 else entity2
                    comparison = f"{better_entity} performs better than {'entity2' if better_entity == entity1 else 'entity1'}"
                elif benchmark_type == 'lower_better':
                    better_entity = entity1 if mean1 < mean2 else entity2
                    comparison = f"{better_entity} performs better than {'entity2' if better_entity == entity1 else 'entity1'}"
                else:
                    comparison = f"{entity1} and {entity2} perform differently"
                
                test_key = f"{entity1}_vs_{entity2}"
                pairwise_tests[test_key] = {
                    "entity1": entity1,
                    "entity2": entity2,
                    "p_value": p_value,
                    "t_statistic": t_stat,
                    "effect_size": effect_size,
                    "is_significant": is_significant,
                    "confidence_interval_1": ci1,
                    "confidence_interval_2": ci2,
                    "comparison": comparison if is_significant else f"No significant difference between {entity1} and {entity2}"
                }
        
        significance_results[entity_type] = pairwise_tests
    
    return significance_results

def generate_benchmarks(
    entities: Dict[str, List[Dict[str, Any]]],
    metrics: Dict[str, Dict[str, Any]],
    metric_type: str,
    context: InsightExecutionContext
) -> Dict[str, Any]:
    """
    Generate benchmark information for the metrics and entities.
    
    Args:
        entities: Dictionary of entities extracted from the query
        metrics: Dictionary of metrics calculated for the entities
        metric_type: The type of metric being analyzed
        context: The insight execution context
        
    Returns:
        Dictionary with benchmark information
    """
    if not entities or not metrics or not metric_type or context.df is None or context.df.empty:
        return {}
    
    benchmarks = {}
    data = context.df
    
    # For each entity type in the query, calculate benchmarks
    for entity_type, entity_list in entities.items():
        # Get entity column
        entity_cols = get_entity_columns(entity_type, data)
        if not entity_cols:
            continue
            
        entity_column = entity_cols[0]
        
        # Get metric column
        metric_cols = get_metric_columns(metric_type, data)
        if not metric_cols:
            continue
            
        metric_column = metric_cols[0]
        
        # Calculate global statistics for this metric
        if metric_column not in data.columns:
            continue
            
        all_values = data[metric_column].dropna()
        
        if len(all_values) < 1:
            continue
            
        # Calculate global statistics for benchmarking
        global_stats = {
            "mean": float(all_values.mean()),
            "median": float(all_values.median()),
            "percentiles": {
                "10th": float(np.percentile(all_values, 10)),
                "25th": float(np.percentile(all_values, 25)),
                "75th": float(np.percentile(all_values, 75)),
                "90th": float(np.percentile(all_values, 90))
            },
            "min": float(all_values.min()),
            "max": float(all_values.max())
        }
        
        # Calculate historical benchmarks if we have date data
        historical_benchmarks = {}
        date_cols = [col for col in data.columns if any(x in col.lower() for x in 
                   ['date', 'time', 'created', 'closed', 'timestamp'])]
                   
        if date_cols and metric_type not in ['trend', 'growth_rate']:  # Skip for trend metrics
            date_col = date_cols[0]
            
            # Ensure date column is datetime
            if pd.api.types.is_string_dtype(data[date_col]):
                try:
                    data[date_col] = pd.to_datetime(data[date_col])
                except Exception as e:
                    logger.warning(f"Failed to convert {date_col} to datetime: {str(e)}")
            
            # If it's a datetime, calculate historical benchmarks
            if pd.api.types.is_datetime64_any_dtype(data[date_col]):
                # Get current year and month
                max_date = data[date_col].max()
                current_year = max_date.year
                current_month = max_date.month
                
                # Previous year data
                prev_year_data = data[data[date_col].dt.year == current_year - 1]
                if not prev_year_data.empty and metric_column in prev_year_data.columns:
                    historical_benchmarks["previous_year"] = {
                        "mean": float(prev_year_data[metric_column].mean()),
                        "median": float(prev_year_data[metric_column].median())
                    }
                
                # Previous quarter data
                current_quarter = (current_month - 1) // 3 + 1
                quarter_start_month = 3 * (current_quarter - 1) + 1
                if current_quarter > 1:
                    prev_quarter_start = datetime.datetime(current_year, quarter_start_month - 3, 1)
                    prev_quarter_end = datetime.datetime(current_year, quarter_start_month, 1) - datetime.timedelta(days=1)
                    prev_quarter_data = data[(data[date_col] >= prev_quarter_start) & (data[date_col] <= prev_quarter_end)]
                    
                    if not prev_quarter_data.empty and metric_column in prev_quarter_data.columns:
                        historical_benchmarks["previous_quarter"] = {
                            "mean": float(prev_quarter_data[metric_column].mean()),
                            "median": float(prev_quarter_data[metric_column].median())
                        }
        
        # Calculate peer group benchmarks
        peer_benchmarks = {}
        
        for entity_info in entity_list:
            entity_value = entity_info["value"]
            entity_key = f"{entity_type}:{entity_value}"
            
            if entity_key not in metrics:
                continue
                
            # Get all other entities of the same type
            peer_group = data[data[entity_column] != entity_value]
            if peer_group.empty or metric_column not in peer_group.columns:
                continue
                
            peer_metric_values = peer_group[metric_column].dropna()
            if len(peer_metric_values) < 1:
                continue
                
            # Calculate peer statistics
            peer_stats = {
                "mean": float(peer_metric_values.mean()),
                "median": float(peer_metric_values.median()),
                "percentile_rank": calculate_percentile_rank(
                    metrics[entity_key]["value"], 
                    peer_metric_values
                )
            }
            
            # Calculate performance indicator based on benchmark type
            benchmark_type = METRIC_RELATIONSHIPS.get(metric_type, {}).get('benchmark_type', 'higher_better')
            entity_value = metrics[entity_key]["value"]
            
            if benchmark_type == 'higher_better':
                if entity_value > peer_stats["mean"]:
                    performance = "above_average"
                elif entity_value < peer_stats["mean"]:
                    performance = "below_average"
                else:
                    performance = "average"
            elif benchmark_type == 'lower_better':
                if entity_value < peer_stats["mean"]:
                    performance = "above_average"
                elif entity_value > peer_stats["mean"]:
                    performance = "below_average"
                else:
                    performance = "average"
            else:
                performance = "neutral"
                
            peer_stats["performance_indicator"] = performance
            peer_benchmarks[entity_value] = peer_stats
        
        # Add trend-based benchmarks if we have date data
        trend_benchmarks = {}
        if date_cols and len(data) >= 5:  # Need at least 5 data points for meaningful trend
            date_col = date_cols[0]
            
            # Ensure date column is datetime
            if pd.api.types.is_datetime64_any_dtype(data[date_col]):
                # For each entity, calculate trend benchmark
                for entity_info in entity_list:
                    entity_value = entity_info["value"]
                    entity_key = f"{entity_type}:{entity_value}"
                    
                    if entity_key not in metrics:
                        continue
                    
                    # Get data for this entity sorted by date
                    entity_data = data[data[entity_column] == entity_value].sort_values(by=date_col)
                    if entity_data.empty or len(entity_data) < 5 or metric_column not in entity_data.columns:
                        continue
                        
                    # Calculate trend using simple linear regression
                    try:
                        X = np.array(range(len(entity_data))).reshape(-1, 1)
                        y = entity_data[metric_column].values
                        
                        model = LinearRegression()
                        model.fit(X, y)
                        
                        # Calculate trend metrics
                        slope = model.coef_[0]
                        r_squared = model.score(X, y)
                        
                        # Determine trend direction
                        if abs(slope) < 0.01 * np.mean(y):  # Slope is very small relative to mean
                            direction = "stable"
                        elif slope > 0:
                            direction = "increasing"
                        else:
                            direction = "decreasing"
                            
                        # Calculate annualized growth rate for easier interpretation
                        if len(entity_data) >= 30:  # Only if we have enough data
                            first_value = entity_data[metric_column].iloc[0]
                            last_value = entity_data[metric_column].iloc[-1]
                            days_span = (entity_data[date_col].iloc[-1] - entity_data[date_col].iloc[0]).days
                            
                            if days_span > 0 and first_value != 0:
                                annual_growth = ((last_value / first_value) ** (365 / days_span) - 1) * 100
                            else:
                                annual_growth = None
                        else:
                            annual_growth = None
                        
                        trend_benchmarks[entity_value] = {
                            "direction": direction,
                            "slope": float(slope),
                            "r_squared": float(r_squared),
                            "annual_growth_rate": annual_growth
                        }
                    except Exception as e:
                        logger.warning(f"Error calculating trend benchmark: {str(e)}")
        
        # Compile all benchmarks into a single result for this entity type
        entity_benchmarks = {
            "global_stats": global_stats,
            "historical_benchmarks": historical_benchmarks,
            "peer_benchmarks": peer_benchmarks,
            "trend_benchmarks": trend_benchmarks
        }
        
        benchmarks[entity_type] = entity_benchmarks
    
    return benchmarks

def calculate_percentile_rank(value: float, distribution: pd.Series) -> float:
    """
    Calculate the percentile rank of a value within a distribution.
    
    Args:
        value: The value to calculate the rank for
        distribution: The distribution to compare against
        
    Returns:
        Percentile rank (0-100)
    """
    if distribution.empty:
        return 50.0  # Default to median if no distribution
        
    # Count values below the current value
    below_count = (distribution < value).sum()
    equal_count = (distribution == value).sum()
    
    # Calculate percentile rank
    percentile = 100 * (below_count + 0.5 * equal_count) / len(distribution)
    
    return float(percentile)

def get_entity_columns(entity_type: str, data: pd.DataFrame) -> List[str]:
    """
    Get columns that match a given entity type.
    
    Args:
        entity_type: The type of entity to find columns for
        data: The dataframe containing the columns
        
    Returns:
        List of column names that match the entity type
    """
    entity_columns = {
        'sales_rep': ['sales_rep', 'rep', 'agent', 'salesperson', 'representative', 'sales_representative', 'employee'],
        'product': ['product', 'item', 'product_name', 'sku', 'offering', 'solution', 'product_id'],
        'region': ['region', 'area', 'territory', 'zone', 'market', 'location', 'geography'],
        'customer_segment': ['segment', 'customer_segment', 'customer_type', 'client_type', 'category', 'customer_category']
    }
    
    if entity_type not in entity_columns or data is None or data.empty:
        return []
        
    col_patterns = entity_columns[entity_type]
    entity_cols = [col for col in data.columns if any(pat.lower() in col.lower() for pat in col_patterns)]
    
    return entity_cols

def get_metric_columns(metric_type: str, data: pd.DataFrame) -> List[str]:
    """
    Get columns that match a given metric type.
    
    Args:
        metric_type: The type of metric to find columns for
        data: The dataframe containing the columns
        
    Returns:
        List of column names that match the metric type
    """
    metric_columns = {
        'days_to_close': ['days_to_close', 'close_time', 'closing_time', 'days', 'cycle_time', 'sales_cycle'],
        'total_sales': ['sales_count', 'num_sales', 'number_of_sales', 'deals_closed', 'transactions'],
        'conversion_rate': ['conversion_rate', 'close_rate', 'success_rate', 'win_rate', 'conversion_percentage'],
        'revenue': ['revenue', 'sales', 'income', 'earnings', 'amount', 'sales_amount', 'deal_value'],
        'profit_margin': ['profit', 'margin', 'profit_margin', 'gross_margin', 'net_margin'],
        'cost_ratio': ['cost', 'expense', 'expenses', 'cost_ratio', 'cost_percentage']
    }
    
    if metric_type not in metric_columns or data is None or data.empty:
        return []
        
    col_patterns = metric_columns[metric_type]
    metric_cols = [col for col in data.columns if any(pat.lower() in col.lower() for pat in col_patterns)]
    
    return metric_cols

def format_metric_value(value: float, metric_type: str) -> str:
    """
    Format a metric value based on the metric type.
    
    Args:
        value: The numeric value to format
        metric_type: The type of metric
        
    Returns:
        Formatted string representation of the value
    """
    if pd.isna(value):
        return "N/A"
        
    if metric_type in ['conversion_rate', 'profit_margin', 'cost_ratio']:
        # Format as percentage
        if value <= 1:
            value *= 100  # Convert to percentage if it's a decimal
        return f"{value:.1f}%"
    elif metric_type == 'revenue':
        # Format as currency
        return f"${value:,.2f}"
    elif metric_type == 'days_to_close':
        # Format with 1 decimal place
        return f"{value:.1f} days"
    elif metric_type == 'total_sales':
        # Format as integer
        return f"{int(value):,}"
    else:
        # Default formatting
        if value.is_integer():
            return f"{int(value):,}"
        else:
            return f"{value:,.2f}"

def generate_visualization_data(
    entities: Dict[str, List[Dict[str, Any]]],
    metric_type: str,
    context: InsightExecutionContext
) -> Dict[str, Any]:
    """
    Generate data for visualizations based on the query results.
    
    Args:
        entities: Dictionary of entities extracted from the query
        metric_type: The type of metric being analyzed
        context: The insight execution context
        
    Returns:
        Dictionary with visualization data
    """
    if not entities or not metric_type or context.df is None or context.df.empty:
        return {}
    
    visualization_data = {
        "time_series": {},
        "comparative": {},
        "distribution": {},
        "benchmarks": {},
        "confidence_intervals": {},
        "visualization_hints": {}
    }
    
    data = context.df
    
    # Check if we have date data for time series
    date_cols = [col for col in data.columns if any(x in col.lower() for x in 
               ['date', 'time', 'created', 'closed', 'timestamp'])]
    
    if date_cols:
        date_col = date_cols[0]
        
        # Ensure date column is in datetime format
        if pd.api.types.is_string_dtype(data[date_col]):
            try:
                data[date_col] = pd.to_datetime(data[date_col])
            except Exception as e:
                logger.warning(f"Failed to convert {date_col} to datetime: {str(e)}")
        
        # Only proceed if
    query = query_context.query
    context = query_context.insight_context
    
    # Extract entities from the query with context for validation
    entities = extract_entities(query, context)
    
    # Extract temporal context
    time_range = extract_temporal_context(query)
    
    # Identify the metric type
    metric_type = identify_metric_type(query)
    
    # Create intent schema
    intent = IntentSchema(
        intent="direct_query",
        metric=metric_type,
        time_range=time_range,
        metric_confidence=0.9 if metric_type else 0.0
    )
    
    # Initialize result with enhanced fields
    result = QueryResult(
        query=query,
        success=False,
        message="Unable to extract requested metric",
        metrics={},
        entities=entities,
        intent=intent,
        confidence_score=0.7 if metric_type and any(entities.values()) else 0.3
    )
    
    # Process metrics for all entity types
    metrics = {}
    processed_any = False
    
    for entity_type, entity_list in entities.items():
        if not entity_list:
            continue
            
        for entity_info in entity_list:
            entity_value = entity_info["value"]
            metric_value, formatted_value = process_metric_for_entity(
                metric_type, 
                entity_type,
                entity_value,
                context
            )
            
            if metric_value is not None:
                entity_key = f"{entity_type}:{entity_value}"
                metrics[entity_key] = {
                    "value": metric_value,
                    "formatted": formatted_value,
                    "metric_type": metric_type,
                    "entity_type": entity_type,
                    "confidence": entity_info.get("confidence", 0.8)
                }
                processed_any = True
    
    if processed_any:
        result.success = True
        result.metrics = metrics
        result.message = "Successfully extracted metrics"
        result.confidence_score = 0.9
        
        # Add related insights and similar entities
        similar_entities = find_similar_entities(
            entities, 
            metrics, 
            metric_type, 
            context
        )
        if similar_entities:
            result.related_insights.extend([
                {
                    "type": "similar_entity",
                    "entity_type": similar["entity_type"],
                    "entity_value": similar["entity_value"],
                    "similarity_score": similar["similarity_score"],
                    "metric_value": similar["metric_value"],
                    "formatted_value": similar["formatted_value"],
                    "comparison": similar["comparison"]
                } for similar in similar_entities
            ])
                
        # Add statistical significance indicators
        if len(metrics) > 1:
            # Calculate if differences between entities are statistically significant
            significance_results = calculate_statistical_significance(
                metrics, 
                context, 
                entities
            )
            if significance_results:
                result.statistical_significance = significance_results
                
        # Generate benchmarks
        benchmarks = generate_benchmarks(
            entities,
            metrics,
            metric_type,
            context
        )
        if benchmarks:
            result.benchmarks = benchmarks
            
        # Add visualization metadata
        result.historical_context = generate_visualization_data(
            entities,
            metric_type,
            context
        )
    
    return result
