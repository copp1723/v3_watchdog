"""
Insight Formatter Engine for Watchdog AI.

This module provides functionality for formatting insights based on user personas
and templates.
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import yaml
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)

class InsightType(Enum):
    """Types of insights that can be formatted."""
    METRIC = "metric"          # Single metric or KPI
    TREND = "trend"            # Time-series data
    COMPARISON = "comparison"  # Comparative analysis
    ANOMALY = "anomaly"        # Anomaly detection
    PREDICTION = "prediction"  # Forecast or prediction
    RECOMMENDATION = "recommendation"  # Action recommendation

@dataclass
class FormattingTemplate:
    """Template for formatting insights."""
    name: str
    persona: str
    insight_type: InsightType
    sections: List[str]
    max_length: int
    required_elements: List[str]
    optional_elements: List[str]
    style_guide: Dict[str, Any]

class InsightFormatter:
    """
    Formats insights based on user personas and templates.
    """
    
    def __init__(self, template_file: str):
        """
        Initialize the formatter with templates.
        
        Args:
            template_file: Path to YAML template file
        """
        self.template_file = template_file
        self.templates: Dict[str, FormattingTemplate] = {}
        self._load_templates()
    
    def _load_templates(self) -> None:
        """Load formatting templates from file."""
        try:
            with open(self.template_file, 'r') as f:
                template_data = yaml.safe_load(f)
            
            for template in template_data:
                self.templates[template['name']] = FormattingTemplate(
                    name=template['name'],
                    persona=template['persona'],
                    insight_type=InsightType(template['insight_type']),
                    sections=template['sections'],
                    max_length=template['max_length'],
                    required_elements=template['required_elements'],
                    optional_elements=template['optional_elements'],
                    style_guide=template['style_guide']
                )
        except Exception as e:
            logger.error(f"Error loading templates: {e}")
            raise
    
    def format_insight(
        self,
        insight: Dict[str, Any],
        persona: str,
        insight_type: InsightType
    ) -> Dict[str, Any]:
        """
        Format an insight for a specific persona.
        
        Args:
            insight: Raw insight data
            persona: User persona (executive, analyst, manager)
            insight_type: Type of insight
            
        Returns:
            Formatted insight
        """
        # Find matching template
        template = self._find_template(persona, insight_type)
        if not template:
            logger.warning(f"No template found for {persona} {insight_type}")
            return insight
        
        # Create formatted insight
        formatted = {
            "type": insight_type.value,
            "persona": persona,
            "sections": {},
            "metadata": {
                "template": template.name,
                "formatting_timestamp": datetime.now().isoformat()
            }
        }
        
        # Format each section
        for section in template.sections:
            if section in insight:
                formatted["sections"][section] = self._format_section(
                    insight[section],
                    template,
                    section
                )
        
        # Add required elements
        for element in template.required_elements:
            if element not in formatted["sections"]:
                formatted["sections"][element] = self._get_default_element(element)
        
        # Add optional elements if present
        for element in template.optional_elements:
            if element in insight:
                formatted["sections"][element] = self._format_section(
                    insight[element],
                    template,
                    element
                )
        
        return formatted
    
    def _find_template(
        self,
        persona: str,
        insight_type: InsightType
    ) -> Optional[FormattingTemplate]:
        """Find matching template for persona and insight type."""
        for template in self.templates.values():
            if (template.persona == persona and
                template.insight_type == insight_type):
                return template
        return None
    
    def _format_section(
        self,
        content: Any,
        template: FormattingTemplate,
        section: str
    ) -> Any:
        """Format a section according to template rules."""
        if isinstance(content, str):
            # Apply length limits
            if len(content) > template.max_length:
                content = content[:template.max_length] + "..."
            
            # Apply style guide
            style = template.style_guide.get(section, {})
            if style.get("capitalize", False):
                content = content.capitalize()
            if style.get("bullet_points", False):
                content = self._format_bullet_points(content)
        
        return content
    
    def _format_bullet_points(self, content: str) -> str:
        """Format content as bullet points."""
        lines = content.split('\n')
        return '\n'.join(f"• {line.strip()}" for line in lines if line.strip())
    
    def _get_default_element(self, element: str) -> Any:
        """Get default content for required elements."""
        defaults = {
            "summary": "No summary available.",
            "metrics": [],
            "trends": [],
            "recommendations": []
        }
        return defaults.get(element, "") 