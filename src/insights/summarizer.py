"""
LLM-based insight summarization system.

Provides natural language summaries of statistical insights using
templated prompts and LLM generation.
"""

import os
import json
from typing import Dict, Any, Optional
import logging
from jinja2 import Environment, FileSystemLoader, Template
import sentry_sdk
from ..utils.errors import InsightGenerationError, ValidationError
from .prompt_tuner import PromptTuner

# Configure logger
logger = logging.getLogger(__name__)

class SummarizationError(InsightGenerationError):
    """Raised when summarization fails validation."""
    pass

class Summarizer:
    """
    Generates natural language summaries from statistical insights.
    
    Uses templated prompts and LLM generation to create executive-friendly
    narratives from complex statistical data.
    """
    
    def __init__(self, llm_client: Any, prompts_path: str = "src/insights/prompts"):
        """
        Initialize the summarizer.
        
        Args:
            llm_client: LLM client instance for text generation
            prompts_path: Path to directory containing prompt templates
        """
        self.llm_client = llm_client
        self.prompts_path = prompts_path
        
        # Set up Jinja environment
        self.env = Environment(
            loader=FileSystemLoader(prompts_path),
            trim_blocks=True,
            lstrip_blocks=True
        )
        
        # Cache for loaded templates
        self._template_cache: Dict[str, Template] = {}
        
        # Validation patterns for common metrics
        self.validation_patterns = {
            'gross_profit': ['gross', 'profit', '$'],
            'deals': ['deal', 'volume', 'count'],
            'days_in_stock': ['days', 'aging', 'aged'],
            'trends': ['trend', 'increase', 'decrease', 'change'],
            'recommendations': ['recommend', 'should', 'consider']
        }
        
        # Add prompt tuner
        self.prompt_tuner = PromptTuner()
    
    def load_template(self, template_name: str) -> Template:
        """
        Load and cache a prompt template.
        
        Args:
            template_name: Name of the template file (e.g. 'sales_performance_prompt.tpl')
            
        Returns:
            Loaded Jinja template
            
        Raises:
            ValidationError: If template file doesn't exist
        """
        if template_name not in self._template_cache:
            try:
                self._template_cache[template_name] = self.env.get_template(template_name)
            except Exception as e:
                raise ValidationError(f"Failed to load template {template_name}: {str(e)}")
        
        return self._template_cache[template_name]
    
    def format_prompt(self, template_name: str, **context) -> str:
        """
        Format a prompt template with context variables.
        
        Args:
            template_name: Name of the template file
            **context: Context variables for template rendering
            
        Returns:
            Formatted prompt string
            
        Raises:
            ValidationError: If required context variables are missing
        """
        required_vars = {'entity_name', 'date_range', 'metrics_table'}
        missing_vars = required_vars - set(context.keys())
        
        if missing_vars:
            raise ValidationError(f"Missing required context variables: {missing_vars}")
        
        template = self.load_template(template_name)
        return template.render(**context)
    
    def validate_summary(self, summary: str, context: Dict[str, Any]) -> None:
        """
        Validate generated summary contains expected elements.
        
        Args:
            summary: Generated summary text
            context: Context used to generate the summary
            
        Raises:
            SummarizationError: If validation fails
        """
        # Check for entity name
        if context['entity_name'].lower() not in summary.lower():
            raise SummarizationError("Summary does not mention the analyzed entity")
        
        # Check for key metric patterns
        found_patterns = 0
        for pattern_list in self.validation_patterns.values():
            if any(p.lower() in summary.lower() for p in pattern_list):
                found_patterns += 1
        
        # Require at least 3 metric pattern types
        if found_patterns < 3:
            raise SummarizationError("Summary lacks sufficient metric coverage")
        
        # Check for recommendations
        if not any(p.lower() in summary.lower() for p in self.validation_patterns['recommendations']):
            raise SummarizationError("Summary lacks actionable recommendations")
    
    def summarize(self, template_name: str, **context) -> str:
        """
        Generate a natural language summary using the LLM.
        
        Args:
            template_name: Name of the template file
            **context: Context variables for template rendering
            
        Returns:
            Generated summary text
            
        Raises:
            SummarizationError: If generation or validation fails
        """
        try:
            # Add Sentry breadcrumb
            sentry_sdk.add_breadcrumb(
                category='summarization',
                message=f'Generating summary using {template_name}',
                level='info'
            )
            
            # Format the base prompt
            prompt = self.format_prompt(template_name, **context)
            
            # Get feedback for this template type
            feedback = []
            if 'feedback_manager' in context:
                feedback = context['feedback_manager'].get_feedback(
                    template_name=template_name
                )
            
            # Tune the prompt if we have feedback
            tuned_prompt = self.prompt_tuner.tune_prompt(prompt, feedback)
            
            # Generate summary
            summary = self.llm_client.generate(tuned_prompt)
            
            # Validate the output
            self.validate_summary(summary, context)
            
            # Add Sentry tags
            sentry_sdk.set_tag("summarization_template", template_name)
            sentry_sdk.set_tag("prompt_tuned", prompt != tuned_prompt)
            
            return summary
            
        except Exception as e:
            logger.error(f"Summarization failed: {str(e)}")
            raise SummarizationError(f"Failed to generate summary: {str(e)}")