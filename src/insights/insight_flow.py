"""
Contextual intelligence module for guiding users through insights with smart follow-up prompts.
"""

import streamlit as st
from typing import Dict, List, Any, Optional
import re
from dataclasses import dataclass
import json
from Levenshtein import distance
from datetime import datetime
from fallback_renderer import FallbackRenderer, ErrorCode, ErrorContext

@dataclass
class FollowUpPrompt:
    """Structure for follow-up prompts with metadata."""
    text: str
    category: str
    priority: int
    context_vars: Dict[str, str] = None
    
    def __post_init__(self):
        if self.context_vars is None:
            self.context_vars = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "prompt": self.text,
            "entity": self.text,
            "timeframe": self.text,
            "priority": self.priority
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FollowUpPrompt':
        """Create from dictionary."""
        return cls(
            text=data["prompt"],
            category=data["prompt"],
            priority=data.get("priority", 1)
        )

class PromptGenerator:
    """Generates contextual follow-up prompts based on insight content."""
    
    def __init__(self, schema: Dict[str, str] = None):
        """
        Initialize the prompt generator with a schema.
        
        Args:
            schema: Dictionary mapping entity types to their descriptions (optional)
        """
        self.schema = schema or {}  # Use empty dict if schema is None
        self.comparison_templates = [
            "How does {entity} compare to {timeframe}?",
            "What's the trend in {entity} over the last {timeframe}?",
            "Is {entity} improving or declining compared to {timeframe}?",
            "Show me the {entity} breakdown by {dimension}."
        ]
        self.dig_deeper_templates = [
            "What factors contributed to {entity}?",
            "What are the key drivers behind {entity}?",
            "Can you explain why {entity} changed?",
            "What recommendations do you have for improving {entity}?"
        ]
        self.context_templates = [
            "How does {entity} relate to {related_entity}?",
            "What's the impact of {entity} on {related_entity}?",
            "Can you analyze {entity} in the context of {related_entity}?"
        ]
        self.entity_aliases = self._build_entity_aliases()
        self.levenshtein_threshold = 0.8  # Similarity threshold for fuzzy matching
        
        # Initialize session state for storing generated prompts
        if 'suggested_prompts' not in st.session_state:
            st.session_state['suggested_prompts'] = []
    
    def _build_entity_aliases(self) -> Dict[str, List[str]]:
        """Build dictionary of entity aliases for fuzzy matching."""
        aliases = {}
        for entity, description in self.schema.items():
            # Extract key terms from description
            terms = re.findall(r'\b\w+\b', description.lower())
            aliases[entity] = list(set(terms))  # Remove duplicates
        return aliases
    
    def _fuzzy_match_entity(self, text: str) -> Optional[str]:
        """
        Find the best matching entity using Levenshtein distance.
        
        Args:
            text: Text to match against entities
            
        Returns:
            Matched entity or None if no good match found
        """
        text = text.lower()
        best_match = None
        best_score = 0
        
        for entity, aliases in self.entity_aliases.items():
            # Check direct match
            if text == entity.lower():
                return entity
            
            # Check aliases
            for alias in aliases:
                score = 1 - (distance(text, alias) / max(len(text), len(alias)))
                if score > best_score and score >= self.levenshtein_threshold:
                    best_score = score
                    best_match = entity
        
        return best_match
    
    def _extract_entities(self, text: str) -> List[Dict[str, str]]:
        """
        Extract entities and timeframes from text.
        
        Args:
            text: Text to extract entities from
            
        Returns:
            List of dictionaries with entity and timeframe
        """
        entities = []
        
        # Extract timeframes
        timeframes = re.findall(r'(?:last|past|previous|this)\s+(?:week|month|quarter|year)', text.lower())
        if not timeframes:
            timeframes = ['current period']  # Default timeframe
        
        # Extract potential entities
        words = re.findall(r'\b\w+\b', text.lower())
        for word in words:
            entity = self._fuzzy_match_entity(word)
            if entity:
                for timeframe in timeframes:
                    entities.append({
                        "entity": entity,
                        "timeframe": timeframe
                    })
        
        return entities
    
    def _deduplicate_prompts(self, prompts: List[FollowUpPrompt]) -> List[FollowUpPrompt]:
        """
        Remove duplicate prompts based on entity/timeframe combinations.
        
        Args:
            prompts: List of prompts to deduplicate
            
        Returns:
            Deduplicated list of prompts
        """
        seen = set()
        unique_prompts = []
        
        for prompt in prompts:
            key = (prompt.text, prompt.text)
            if key not in seen:
                seen.add(key)
                unique_prompts.append(prompt)
        
        return unique_prompts
    
    def extract_entities(self, summary: str) -> List[str]:
        """
        Extract relevant entities from the summary text.
        
        Args:
            summary: The insight summary text
            
        Returns:
            List of extracted entity names
        """
        entities = []
        for entity_type, description in self.schema.items():
            # Look for entity mentions in the summary
            pattern = rf'\b{re.escape(entity_type)}\b'
            if re.search(pattern, summary, re.IGNORECASE):
                entities.append(entity_type)
            else:
                # Check for terms from the description
                terms = re.findall(r'\b\w+\b', description.lower())
                for term in terms:
                    if re.search(rf'\b{re.escape(term)}\b', summary.lower()):
                        entities.append(entity_type)
                        break
        return list(set(entities))  # Remove duplicates
    
    def extract_metrics(self, summary: str) -> List[str]:
        """
        Extract metrics and values from the summary text.
        
        Args:
            summary: The insight summary text
            
        Returns:
            List of extracted metrics
        """
        # Pattern for currency, percentages, and numbers with optional signs
        metric_pattern = r'(?:[\+\-\$]?\d+(?:,\d{3})*(?:\.\d+)?%?)'
        metrics = re.findall(metric_pattern, summary)
        
        # Filter out standalone single digits that might be part of text
        metrics = [m for m in metrics if not (len(m) == 1 and m.isdigit())]
        
        return list(set(metrics))  # Remove duplicates
    
    def extract_timeframes(self, summary: str) -> List[str]:
        """
        Extract time-related phrases from the summary text.
        
        Args:
            summary: The insight summary text
            
        Returns:
            List of extracted timeframes
        """
        timeframes = []
        
        # Common timeframe patterns
        patterns = [
            r'(?:last|past|previous|this)\s+(?:week|month|quarter|year)',
            r'Q[1-4]',
            r'(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|'
            r'Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)',
            r'\d{4}',  # Years
            r'year[\s-](?:to|over|on)[\s-]year',
            r'month[\s-](?:to|over|on)[\s-]month'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, summary, re.IGNORECASE)
            timeframes.extend(matches)
        
        # Clean and normalize timeframes
        timeframes = [tf.lower().strip() for tf in timeframes]
        return list(set(timeframes))  # Remove duplicates
    
    def generate_comparison_prompts(self, entities: List[str], timeframes: List[str]) -> List[FollowUpPrompt]:
        """
        Generate comparison-based follow-up prompts.
        
        Args:
            entities: List of entities from the summary
            timeframes: List of timeframes from the summary
            
        Returns:
            List of comparison prompts
        """
        prompts = []
        
        # If we have entities but no timeframes, suggest time-based comparisons
        if entities and not timeframes:
            for entity in entities:
                prompts.append(FollowUpPrompt(
                    text=f"How does {entity} compare to last month?",
                    category="comparison",
                    priority=1,
                    context_vars={"entity": entity, "timeframe": "last month"}
                ))
        
        # If we have both entities and timeframes, suggest dimension-based comparisons
        elif entities and timeframes:
            for entity in entities:
                for timeframe in timeframes:
                    prompts.append(FollowUpPrompt(
                        text=f"Show me the {entity} breakdown by category compared to {timeframe}",
                        category="comparison",
                        priority=2,
                        context_vars={"entity": entity, "timeframe": timeframe, "dimension": "category"}
                    ))
        
        return prompts
    
    def generate_dig_deeper_prompts(self, entities: List[str], metrics: List[str]) -> List[FollowUpPrompt]:
        """
        Generate prompts for digging deeper into insights.
        
        Args:
            entities: List of entities from the summary
            metrics: List of metrics from the summary
            
        Returns:
            List of dig deeper prompts
        """
        prompts = []
        
        for entity in entities:
            prompts.append(FollowUpPrompt(
                text=f"What factors contributed to the {entity} changes?",
                category="analysis",
                priority=1,
                context_vars={"entity": entity}
            ))
            
            if metrics:
                prompts.append(FollowUpPrompt(
                    text=f"What recommendations do you have for improving {entity}?",
                    category="recommendation",
                    priority=2,
                    context_vars={"entity": entity}
                ))
        
        return prompts
    
    def generate_context_prompts(self, entities: List[str]) -> List[FollowUpPrompt]:
        """
        Generate prompts for adding context from related entities.
        
        Args:
            entities: List of entities from the summary
            
        Returns:
            List of context prompts
        """
        prompts = []
        
        # Find related entities from the schema
        for entity in entities:
            for related_entity, description in self.schema.items():
                if related_entity != entity:
                    prompts.append(FollowUpPrompt(
                        text=f"How does {entity} relate to {related_entity}?",
                        category="context",
                        priority=1,
                        context_vars={"entity": entity, "related_entity": related_entity}
                    ))
        
        return prompts
    
    def generate_follow_up_prompts(self, summary: str) -> List[FollowUpPrompt]:
        """
        Generate a comprehensive list of follow-up prompts based on the insight summary.
        
        Args:
            summary: The insight summary text
            
        Returns:
            List of follow-up prompts
        """
        # Extract relevant information from the summary
        entities = self._extract_entities(summary)
        metrics = self.extract_metrics(summary)
        timeframes = self.extract_timeframes(summary)
        
        # Generate different types of prompts
        comparison_prompts = self.generate_comparison_prompts(entities, timeframes)
        dig_deeper_prompts = self.generate_dig_deeper_prompts(entities, metrics)
        context_prompts = self.generate_context_prompts(entities)
        
        # Combine all prompts
        all_prompts = comparison_prompts + dig_deeper_prompts + context_prompts
        
        # Sort by priority (higher priority first)
        all_prompts.sort(key=lambda x: x.priority, reverse=True)
        
        # Deduplicate prompts
        unique_prompts = self._deduplicate_prompts(all_prompts)
        
        # Store in session state
        st.session_state['suggested_prompts'] = [p.to_dict() for p in unique_prompts]
        
        return unique_prompts
    
    def get_stored_prompts(self) -> List[FollowUpPrompt]:
        """Retrieve stored prompts from session state."""
        stored = st.session_state.get('suggested_prompts', [])
        return [FollowUpPrompt.from_dict(p) for p in stored]

    def generate_prompt(self, system_prompt: str, user_query: str, validation_context: Dict[str, Any] = None) -> str:
        """
        Generate a formatted prompt for LLM consumption.
        
        Args:
            system_prompt: The system instruction prompt text
            user_query: The user's question
            validation_context: Dictionary with context information (optional)
            
        Returns:
            Complete formatted prompt for the LLM
        """
        # Default validation context if not provided
        if validation_context is None:
            validation_context = {}
            
        # Build context section
        context_parts = []
        
        # Add data shape if available
        if 'data_shape' in validation_context:
            rows, cols = validation_context['data_shape']
            context_parts.append(f"Dataset contains {rows} rows and {cols} columns.")
        
        # Add column information if available
        if 'columns' in validation_context:
            columns = validation_context['columns']
            context_parts.append(f"Available columns:\n{', '.join(columns)}")
        
        # Add data type information if available
        if 'data_types' in validation_context:
            data_types = validation_context['data_types']
            context_parts.append("Column data types:")
            for col, dtype in data_types.items():
                context_parts.append(f"- {col}: {dtype}")
        
        # Add basic statistics if available
        if 'basic_stats' in validation_context:
            context_parts.append("Basic statistics for numeric columns:")
            stats = validation_context['basic_stats']
            for col, metrics in stats.items():
                if isinstance(metrics, dict):
                    stats_str = "\n  ".join([f"{k}: {v}" for k, v in metrics.items() if k in ['mean', 'min', 'max', 'count']])
                    context_parts.append(f"- {col}:\n  {stats_str}")
        
        # Add lead source breakdown if available
        if 'lead_source_breakdown' in validation_context:
            context_parts.append("\nLead Source Breakdown:")
            for source, count in validation_context['lead_source_breakdown'].items():
                context_parts.append(f"- {source}: {count} deals")
        
        # Format the context section with more line breaks for readability
        context_section = "\n\n".join(context_parts)
        
        # Add structured output format guidance
        response_format_guidance = """
\nRESPONSE FORMAT GUIDANCE:
Your response should follow a structured format like:
{
  "summary": "A concise 1-2 sentence overview of the key finding",
  "value_insights": [
    "Specific insight point with relevant metrics and business impact",
    "Another specific insight with supporting data"
  ],
  "actionable_flags": [
    "Recommended action based on the analysis",
    "Another suggestion for business improvement"
  ],
  "confidence": "high/medium/low"
}

Ensure each insight uses clear, concise markdown formatting with bullet points where appropriate.
"""
        
        # Combine all parts
        full_prompt = f"""
{system_prompt}

CONTEXT INFORMATION:
{context_section}
{response_format_guidance}

USER QUERY:
{user_query}
"""
        return full_prompt

def render_follow_up_suggestions(prompts: List[FollowUpPrompt]) -> None:
    """
    Render follow-up suggestions in the UI.
    
    Args:
        prompts: List of FollowUpPrompt objects
    """
    if not prompts:
        st.info("No follow-up suggestions available.")
        return
    
    st.markdown("### 💡 Suggested Follow-up Questions")
    for prompt in prompts:
        if st.button(prompt.text):
            st.session_state['current_prompt'] = prompt.text
            st.rerun()

def generate_llm_prompt(selected_prompt: str, context_vars: Dict[str, str], previous_insights: List[Dict[str, Any]] = None) -> str:
    """
    Generate a structured LLM prompt based on the selected follow-up prompt.
    
    Args:
        selected_prompt: The selected follow-up prompt text
        context_vars: Variables to include in the prompt
        previous_insights: List of previous insights to include for context
        
    Returns:
        Structured LLM prompt
    """
    # Base prompt structure
    prompt = {
        "query": selected_prompt,
        "context": context_vars,
        "previous_insights": []
    }
    
    # Add previous insights if available
    if previous_insights:
        for insight in previous_insights[-3:]:  # Include up to 3 previous insights
            if 'summary' in insight:
                prompt["previous_insights"].append({
                    "summary": insight["summary"],
                    "timestamp": insight.get("timestamp", "")
                })
    
    # Convert to JSON for structured prompt
    return json.dumps(prompt, indent=2)

def render_insight_flow(response: Dict[str, Any], schema: Dict[str, str], previous_insights: List[Dict[str, Any]] = None) -> Optional[str]:
    """
    Render the insight flow with fallback error handling.
    
    Args:
        response: The insight response data
        schema: Dictionary mapping entity types to their descriptions
        previous_insights: List of previous insights for context
        
    Returns:
        Optional[str]: The rendered insight or None if there was an error
    """
    try:
        # Initialize the fallback renderer
        fallback_renderer = FallbackRenderer()
        
        # Check if the response indicates an error
        if response.get("type") == "error":
            error_context = ErrorContext(
                error_code=ErrorCode(response.get("error_code", "system_error")),
                error_message=response.get("error_message", "Unknown error occurred"),
                details=response.get("error_details", {}),
                timestamp=datetime.now().isoformat(),
                user_query=response.get("user_query"),
                affected_columns=response.get("affected_columns"),
                stack_trace=response.get("stack_trace")
            )
            return fallback_renderer.render_fallback(error_context)
        
        # Process the insight response
        if not response.get("content"):
            error_context = ErrorContext(
                error_code=ErrorCode.NO_MATCHING_DATA,
                error_message="No insight content generated",
                details={"criteria": "insight content"},
                timestamp=datetime.now().isoformat(),
                user_query=response.get("user_query")
            )
            return fallback_renderer.render_fallback(error_context)
        
        # Generate follow-up prompts
        prompt_generator = PromptGenerator(schema)
        prompts = prompt_generator.generate_follow_up_prompts(response["content"])
        
        # Render the insight and prompts
        st.write(response["content"])
        render_follow_up_suggestions(prompts)
        
        return response["content"]
        
    except Exception as e:
        # Handle unexpected errors
        error_context = ErrorContext(
            error_code=ErrorCode.SYSTEM_ERROR,
            error_message=str(e),
            details={"error_details": str(e)},
            timestamp=datetime.now().isoformat(),
            stack_trace=getattr(e, "__traceback__", None)
        )
        return fallback_renderer.render_fallback(error_context) 