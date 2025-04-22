"""
Feedback Prompt Generator for Watchdog AI.

This module provides functionality for generating feedback prompts and UI controls
for collecting user feedback on insights.
"""

import json
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import uuid
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)

class FeedbackType(Enum):
    """Types of feedback that can be collected."""
    THUMBS = "thumbs"          # Thumbs up/down
    RATING = "rating"          # Numeric rating (1-5)
    COMMENT = "comment"        # Text comment
    STRUCTURED = "structured"  # Structured feedback with predefined options

@dataclass
class FeedbackPrompt:
    """Represents a feedback prompt with UI controls."""
    prompt_id: str
    insight_id: str
    question: str
    feedback_type: FeedbackType
    options: Optional[List[str]] = None
    required: bool = True
    metadata: Dict[str, Any] = None

class FeedbackPromptGenerator:
    """
    Generates feedback prompts and UI controls for insights.
    """
    
    def __init__(self):
        """Initialize the feedback prompt generator."""
        self.default_prompts = {
            "thumbs": "Was this insight helpful?",
            "rating": "How would you rate this insight?",
            "comment": "What would make this insight better?",
            "structured": "What aspects of this insight were most valuable?"
        }
        
        self.structured_options = {
            "executive": [
                "Clarity",
                "Actionability",
                "Strategic Value",
                "Timeliness"
            ],
            "analyst": [
                "Accuracy",
                "Depth of Analysis",
                "Methodology",
                "Data Quality",
                "Insightfulness"
            ],
            "manager": [
                "Practical Value",
                "Actionability",
                "Team Impact",
                "Implementation Ease",
                "Resource Requirements"
            ]
        }
    
    def generate_prompt(
        self,
        insight_id: str,
        feedback_type: FeedbackType,
        persona: Optional[str] = None,
        custom_question: Optional[str] = None
    ) -> FeedbackPrompt:
        """
        Generate a feedback prompt for an insight.
        
        Args:
            insight_id: ID of the insight
            feedback_type: Type of feedback to collect
            persona: User persona (optional)
            custom_question: Custom question text (optional)
            
        Returns:
            FeedbackPrompt object
        """
        prompt = FeedbackPrompt(
            prompt_id=str(uuid.uuid4()),
            insight_id=insight_id,
            question=custom_question or self.default_prompts[feedback_type.value],
            feedback_type=feedback_type,
            metadata={
                "persona": persona,
                "timestamp": datetime.now().isoformat()
            }
        )
        
        if feedback_type == FeedbackType.STRUCTURED and persona:
            prompt.options = self.structured_options.get(persona, [])
        
        return prompt
    
    def generate_feedback_ui(
        self,
        insight_id: str,
        persona: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate complete feedback UI for an insight.
        
        Args:
            insight_id: ID of the insight
            persona: User persona (optional)
            
        Returns:
            Dictionary containing feedback UI components
        """
        ui_components = {
            "insight_id": insight_id,
            "prompts": [],
            "metadata": {
                "persona": persona,
                "timestamp": datetime.now().isoformat()
            }
        }
        
        # Add thumbs prompt
        ui_components["prompts"].append(
            self.generate_prompt(
                insight_id,
                FeedbackType.THUMBS,
                persona
            ).__dict__
        )
        
        # Add rating prompt
        ui_components["prompts"].append(
            self.generate_prompt(
                insight_id,
                FeedbackType.RATING,
                persona
            ).__dict__
        )
        
        # Add comment prompt
        ui_components["prompts"].append(
            self.generate_prompt(
                insight_id,
                FeedbackType.COMMENT,
                persona,
                required=False
            ).__dict__
        )
        
        # Add structured prompt if persona is provided
        if persona:
            ui_components["prompts"].append(
                self.generate_prompt(
                    insight_id,
                    FeedbackType.STRUCTURED,
                    persona
                ).__dict__
            )
        
        return ui_components
    
    def get_prompt_by_id(self, prompt_id: str) -> Optional[FeedbackPrompt]:
        """
        Get a feedback prompt by ID.
        
        Args:
            prompt_id: ID of the prompt
            
        Returns:
            FeedbackPrompt object or None if not found
        """
        # In a real implementation, this would query a database
        # For now, we'll return None as prompts are not persisted
        return None 