"""
Insight Feedback Engine for Watchdog AI.

This module provides functionality for collecting, analyzing, and applying feedback
on insights to improve their accuracy and relevance.
"""

import os
import json
import logging
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
from threading import Lock

# Configure logging
logger = logging.getLogger(__name__)

class FeedbackType(Enum):
    """Types of feedback that can be collected."""
    RATING = "rating"  # Numeric rating (e.g., 1-5 stars)
    THUMBS = "thumbs"  # Thumbs up/down
    TEXT = "text"      # Free-form text feedback
    STRUCTURED = "structured"  # Structured feedback with predefined options

class UserPersona(Enum):
    """User personas for persona-based insight formatting."""
    EXECUTIVE = "executive"  # High-level summary, strategic focus
    MANAGER = "manager"      # Balanced detail, operational focus
    ANALYST = "analyst"      # Detailed analysis, technical focus
    OPERATOR = "operator"    # Task-oriented, action-focused

@dataclass
class InsightFeedback:
    """Represents feedback for a specific insight."""
    feedback_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    insight_id: str = ""
    user_id: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    feedback_type: FeedbackType = FeedbackType.RATING
    rating: Optional[int] = None
    thumbs_up: Optional[bool] = None
    text_feedback: Optional[str] = None
    structured_feedback: Optional[Dict[str, Any]] = None
    expected_vs_actual: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "feedback_id": self.feedback_id,
            "insight_id": self.insight_id,
            "user_id": self.user_id,
            "timestamp": self.timestamp,
            "feedback_type": self.feedback_type.value,
            "rating": self.rating,
            "thumbs_up": self.thumbs_up,
            "text_feedback": self.text_feedback,
            "structured_feedback": self.structured_feedback,
            "expected_vs_actual": self.expected_vs_actual,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'InsightFeedback':
        """Create from dictionary."""
        return cls(
            feedback_id=data.get("feedback_id", str(uuid.uuid4())),
            insight_id=data.get("insight_id", ""),
            user_id=data.get("user_id", ""),
            timestamp=data.get("timestamp", datetime.now().isoformat()),
            feedback_type=FeedbackType(data.get("feedback_type", "rating")),
            rating=data.get("rating"),
            thumbs_up=data.get("thumbs_up"),
            text_feedback=data.get("text_feedback"),
            structured_feedback=data.get("structured_feedback"),
            expected_vs_actual=data.get("expected_vs_actual"),
            metadata=data.get("metadata", {})
        )

@dataclass
class UserProfile:
    """User profile for persona-based insight formatting."""
    user_id: str
    name: str
    persona: UserPersona
    preferences: Dict[str, Any] = field(default_factory=dict)
    feedback_history: List[str] = field(default_factory=list)  # List of feedback IDs
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "user_id": self.user_id,
            "name": self.name,
            "persona": self.persona.value,
            "preferences": self.preferences,
            "feedback_history": self.feedback_history,
            "created_at": self.created_at,
            "last_updated": self.last_updated
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UserProfile':
        """Create from dictionary."""
        return cls(
            user_id=data.get("user_id", ""),
            name=data.get("name", ""),
            persona=UserPersona(data.get("persona", "executive")),
            preferences=data.get("preferences", {}),
            feedback_history=data.get("feedback_history", []),
            created_at=data.get("created_at", datetime.now().isoformat()),
            last_updated=data.get("last_updated", datetime.now().isoformat())
        )

class InsightFeedbackEngine:
    """
    Engine for collecting, analyzing, and applying feedback on insights.
    """
    
    def __init__(self, data_dir: str = None):
        """
        Initialize the feedback engine.
        
        Args:
            data_dir: Directory to store feedback data
        """
        self.data_dir = data_dir or os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "feedback")
        self.feedback_file = os.path.join(self.data_dir, "feedback.json")
        self.user_profiles_file = os.path.join(self.data_dir, "user_profiles.json")
        self.ab_test_results_file = os.path.join(self.data_dir, "ab_test_results.json")
        
        # Create directory if it doesn't exist
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Initialize data structures
        self.feedback: List[InsightFeedback] = []
        self.user_profiles: Dict[str, UserProfile] = {}
        self.ab_test_results: Dict[str, Dict[str, Any]] = {}
        
        # Load existing data
        self._load_data()
        
        # Lock for thread safety
        self.lock = Lock()
    
    def _load_data(self) -> None:
        """Load data from files."""
        # Load feedback
        if os.path.exists(self.feedback_file):
            try:
                with open(self.feedback_file, 'r') as f:
                    data = json.load(f)
                    self.feedback = [InsightFeedback.from_dict(item) for item in data]
            except Exception as e:
                logger.error(f"Error loading feedback: {e}")
        
        # Load user profiles
        if os.path.exists(self.user_profiles_file):
            try:
                with open(self.user_profiles_file, 'r') as f:
                    data = json.load(f)
                    self.user_profiles = {
                        user_id: UserProfile.from_dict(profile_data)
                        for user_id, profile_data in data.items()
                    }
            except Exception as e:
                logger.error(f"Error loading user profiles: {e}")
        
        # Load A/B test results
        if os.path.exists(self.ab_test_results_file):
            try:
                with open(self.ab_test_results_file, 'r') as f:
                    self.ab_test_results = json.load(f)
            except Exception as e:
                logger.error(f"Error loading A/B test results: {e}")
    
    def _save_data(self) -> None:
        """Save data to files."""
        # Save feedback
        try:
            with open(self.feedback_file, 'w') as f:
                json.dump(
                    [feedback.to_dict() for feedback in self.feedback],
                    f,
                    indent=2
                )
        except Exception as e:
            logger.error(f"Error saving feedback: {e}")
        
        # Save user profiles
        try:
            with open(self.user_profiles_file, 'w') as f:
                json.dump(
                    {user_id: profile.to_dict() for user_id, profile in self.user_profiles.items()},
                    f,
                    indent=2
                )
        except Exception as e:
            logger.error(f"Error saving user profiles: {e}")
        
        # Save A/B test results
        try:
            with open(self.ab_test_results_file, 'w') as f:
                json.dump(self.ab_test_results, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving A/B test results: {e}")
    
    def add_feedback(
        self,
        insight_id: str,
        user_id: str,
        feedback_type: FeedbackType,
        rating: Optional[int] = None,
        thumbs_up: Optional[bool] = None,
        text_feedback: Optional[str] = None,
        structured_feedback: Optional[Dict[str, Any]] = None,
        expected_vs_actual: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Add feedback for an insight.
        
        Args:
            insight_id: ID of the insight
            user_id: ID of the user providing feedback
            feedback_type: Type of feedback
            rating: Numeric rating (if feedback_type is RATING)
            thumbs_up: Thumbs up/down (if feedback_type is THUMBS)
            text_feedback: Text feedback (if feedback_type is TEXT)
            structured_feedback: Structured feedback (if feedback_type is STRUCTURED)
            expected_vs_actual: Expected vs actual results
            metadata: Additional metadata
            
        Returns:
            ID of the created feedback
        """
        with self.lock:
            feedback = InsightFeedback(
                insight_id=insight_id,
                user_id=user_id,
                feedback_type=feedback_type,
                rating=rating,
                thumbs_up=thumbs_up,
                text_feedback=text_feedback,
                structured_feedback=structured_feedback,
                expected_vs_actual=expected_vs_actual,
                metadata=metadata or {}
            )
            
            self.feedback.append(feedback)
            
            # Update user profile
            if user_id in self.user_profiles:
                self.user_profiles[user_id].feedback_history.append(feedback.feedback_id)
                self.user_profiles[user_id].last_updated = datetime.now().isoformat()
            
            self._save_data()
            
            return feedback.feedback_id
    
    def get_feedback(self, feedback_id: str) -> Optional[InsightFeedback]:
        """
        Get feedback by ID.
        
        Args:
            feedback_id: ID of the feedback
            
        Returns:
            Feedback object or None if not found
        """
        for feedback in self.feedback:
            if feedback.feedback_id == feedback_id:
                return feedback
        return None
    
    def get_insight_feedback(self, insight_id: str) -> List[InsightFeedback]:
        """
        Get all feedback for an insight.
        
        Args:
            insight_id: ID of the insight
            
        Returns:
            List of feedback objects
        """
        return [f for f in self.feedback if f.insight_id == insight_id]
    
    def get_user_feedback(self, user_id: str) -> List[InsightFeedback]:
        """
        Get all feedback from a user.
        
        Args:
            user_id: ID of the user
            
        Returns:
            List of feedback objects
        """
        return [f for f in self.feedback if f.user_id == user_id]
    
    def add_user_profile(
        self,
        user_id: str,
        name: str,
        persona: UserPersona,
        preferences: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add or update a user profile.
        
        Args:
            user_id: ID of the user
            name: Name of the user
            persona: User persona
            preferences: User preferences
        """
        with self.lock:
            if user_id in self.user_profiles:
                # Update existing profile
                profile = self.user_profiles[user_id]
                profile.name = name
                profile.persona = persona
                if preferences:
                    profile.preferences.update(preferences)
                profile.last_updated = datetime.now().isoformat()
            else:
                # Create new profile
                profile = UserProfile(
                    user_id=user_id,
                    name=name,
                    persona=persona,
                    preferences=preferences or {}
                )
                self.user_profiles[user_id] = profile
            
            self._save_data()
    
    def get_user_profile(self, user_id: str) -> Optional[UserProfile]:
        """
        Get a user profile by ID.
        
        Args:
            user_id: ID of the user
            
        Returns:
            User profile or None if not found
        """
        return self.user_profiles.get(user_id)
    
    def record_ab_test_result(
        self,
        test_id: str,
        variant: str,
        user_id: str,
        insight_id: str,
        metric: str,
        value: Any
    ) -> None:
        """
        Record a result from an A/B test.
        
        Args:
            test_id: ID of the A/B test
            variant: Test variant (e.g., 'A' or 'B')
            user_id: ID of the user
            insight_id: ID of the insight
            metric: Metric being measured (e.g., 'open_rate', 'click_through')
            value: Value of the metric
        """
        with self.lock:
            if test_id not in self.ab_test_results:
                self.ab_test_results[test_id] = {
                    "variants": {},
                    "metrics": {}
                }
            
            # Record variant
            if variant not in self.ab_test_results[test_id]["variants"]:
                self.ab_test_results[test_id]["variants"][variant] = []
            
            self.ab_test_results[test_id]["variants"][variant].append({
                "user_id": user_id,
                "insight_id": insight_id,
                "timestamp": datetime.now().isoformat()
            })
            
            # Record metric
            if metric not in self.ab_test_results[test_id]["metrics"]:
                self.ab_test_results[test_id]["metrics"][metric] = {}
            
            if variant not in self.ab_test_results[test_id]["metrics"][metric]:
                self.ab_test_results[test_id]["metrics"][metric][variant] = []
            
            self.ab_test_results[test_id]["metrics"][metric][variant].append({
                "user_id": user_id,
                "insight_id": insight_id,
                "value": value,
                "timestamp": datetime.now().isoformat()
            })
            
            self._save_data()
    
    def get_ab_test_results(self, test_id: str) -> Dict[str, Any]:
        """
        Get results from an A/B test.
        
        Args:
            test_id: ID of the A/B test
            
        Returns:
            Dictionary of test results
        """
        return self.ab_test_results.get(test_id, {})
    
    def get_insight_stats(self, insight_id: str) -> Dict[str, Any]:
        """
        Get statistics for an insight.
        
        Args:
            insight_id: ID of the insight
            
        Returns:
            Dictionary of insight statistics
        """
        feedback = self.get_insight_feedback(insight_id)
        
        stats = {
            "total_feedback": len(feedback),
            "rating_stats": {},
            "thumbs_stats": {},
            "feedback_types": {},
            "text_feedback": [],
            "structured_feedback": {}
        }
        
        # Calculate rating statistics
        ratings = [f.rating for f in feedback if f.rating is not None]
        if ratings:
            stats["rating_stats"] = {
                "average": sum(ratings) / len(ratings),
                "count": len(ratings),
                "distribution": {i: ratings.count(i) for i in range(1, 6)}
            }
        
        # Calculate thumbs statistics
        thumbs_up = [f.thumbs_up for f in feedback if f.thumbs_up is not None]
        if thumbs_up:
            stats["thumbs_stats"] = {
                "up_count": sum(1 for t in thumbs_up if t),
                "down_count": sum(1 for t in thumbs_up if not t),
                "up_percentage": sum(1 for t in thumbs_up if t) / len(thumbs_up) * 100
            }
        
        # Count feedback types
        feedback_types = [f.feedback_type.value for f in feedback]
        stats["feedback_types"] = {t: feedback_types.count(t) for t in set(feedback_types)}
        
        # Collect text feedback
        stats["text_feedback"] = [f.text_feedback for f in feedback if f.text_feedback]
        
        # Collect structured feedback
        structured_feedback = [f.structured_feedback for f in feedback if f.structured_feedback]
        if structured_feedback:
            # Combine all structured feedback
            combined = {}
            for sf in structured_feedback:
                for key, value in sf.items():
                    if key not in combined:
                        combined[key] = []
                    combined[key].append(value)
            
            # Count occurrences of each value
            for key, values in combined.items():
                combined[key] = {v: values.count(v) for v in set(values)}
            
            stats["structured_feedback"] = combined
        
        return stats
    
    def get_user_stats(self, user_id: str) -> Dict[str, Any]:
        """
        Get statistics for a user.
        
        Args:
            user_id: ID of the user
            
        Returns:
            Dictionary of user statistics
        """
        feedback = self.get_user_feedback(user_id)
        
        stats = {
            "total_feedback": len(feedback),
            "feedback_by_type": {},
            "recent_feedback": [],
            "insight_ids": list(set(f.insight_id for f in feedback))
        }
        
        # Count feedback by type
        feedback_types = [f.feedback_type.value for f in feedback]
        stats["feedback_by_type"] = {t: feedback_types.count(t) for t in set(feedback_types)}
        
        # Get recent feedback
        stats["recent_feedback"] = sorted(
            [f.to_dict() for f in feedback],
            key=lambda x: x["timestamp"],
            reverse=True
        )[:10]
        
        return stats
    
    def get_persona_based_formatting(self, user_id: str) -> Dict[str, Any]:
        """
        Get persona-based formatting preferences for a user.
        
        Args:
            user_id: ID of the user
            
        Returns:
            Dictionary of formatting preferences
        """
        profile = self.get_user_profile(user_id)
        
        if not profile:
            # Default to executive persona
            return self._get_default_formatting(UserPersona.EXECUTIVE)
        
        return self._get_default_formatting(profile.persona)
    
    def _get_default_formatting(self, persona: UserPersona) -> Dict[str, Any]:
        """
        Get default formatting preferences for a persona.
        
        Args:
            persona: User persona
            
        Returns:
            Dictionary of formatting preferences
        """
        if persona == UserPersona.EXECUTIVE:
            return {
                "tone": "concise",
                "verbosity": "low",
                "detail_level": "high-level",
                "chart_types": ["summary", "trend"],
                "max_bullets": 3,
                "max_charts": 1,
                "focus": "strategic"
            }
        elif persona == UserPersona.MANAGER:
            return {
                "tone": "balanced",
                "verbosity": "medium",
                "detail_level": "balanced",
                "chart_types": ["summary", "trend", "comparison"],
                "max_bullets": 5,
                "max_charts": 2,
                "focus": "operational"
            }
        elif persona == UserPersona.ANALYST:
            return {
                "tone": "detailed",
                "verbosity": "high",
                "detail_level": "detailed",
                "chart_types": ["summary", "trend", "comparison", "distribution"],
                "max_bullets": 8,
                "max_charts": 3,
                "focus": "analytical"
            }
        else:  # OPERATOR
            return {
                "tone": "action-oriented",
                "verbosity": "medium",
                "detail_level": "task-focused",
                "chart_types": ["summary", "action"],
                "max_bullets": 4,
                "max_charts": 1,
                "focus": "actionable"
            }
    
    def generate_feedback_prompt(self, insight_id: str) -> Dict[str, Any]:
        """
        Generate a prompt for collecting feedback on an insight.
        
        Args:
            insight_id: ID of the insight
            
        Returns:
            Dictionary containing feedback prompt components
        """
        return {
            "rating_prompt": "How helpful was this insight?",
            "thumbs_prompt": "Was this insight what you expected?",
            "text_prompt": "What would make this insight better?",
            "structured_prompt": "What aspects of this insight were most valuable?",
            "structured_options": [
                "Accuracy",
                "Relevance",
                "Actionability",
                "Clarity",
                "Completeness"
            ]
        }
    
    def format_insight_for_persona(
        self,
        insight: Dict[str, Any],
        user_id: str
    ) -> Dict[str, Any]:
        """
        Format an insight for a specific user persona.
        
        Args:
            insight: Original insight
            user_id: ID of the user
            
        Returns:
            Formatted insight
        """
        formatting = self.get_persona_based_formatting(user_id)
        
        # Create a copy of the insight to avoid modifying the original
        formatted_insight = insight.copy()
        
        # Apply formatting based on persona preferences
        if "summary" in formatted_insight:
            # Adjust summary length based on verbosity
            if formatting["verbosity"] == "low":
                # Truncate to first paragraph or ~100 words
                summary = formatted_insight["summary"]
                paragraphs = summary.split("\n\n")
                if len(paragraphs) > 1:
                    formatted_insight["summary"] = paragraphs[0]
                else:
                    words = summary.split()
                    if len(words) > 100:
                        formatted_insight["summary"] = " ".join(words[:100]) + "..."
            
            elif formatting["verbosity"] == "high":
                # Ensure full detail is included
                pass  # No changes needed
        
        # Adjust bullet points
        if "bullets" in formatted_insight:
            bullets = formatted_insight["bullets"]
            if len(bullets) > formatting["max_bullets"]:
                # Prioritize bullets based on importance
                # This is a simplified approach - in a real implementation,
                # you would use more sophisticated prioritization
                formatted_insight["bullets"] = bullets[:formatting["max_bullets"]]
        
        # Adjust charts
        if "charts" in formatted_insight:
            charts = formatted_insight["charts"]
            if len(charts) > formatting["max_charts"]:
                # Filter charts based on preferred chart types
                filtered_charts = [
                    chart for chart in charts
                    if chart.get("type") in formatting["chart_types"]
                ]
                
                # If still too many, take the first N
                if len(filtered_charts) > formatting["max_charts"]:
                    filtered_charts = filtered_charts[:formatting["max_charts"]]
                
                formatted_insight["charts"] = filtered_charts
        
        # Add metadata about formatting
        formatted_insight["formatting_metadata"] = {
            "persona": formatting["focus"],
            "verbosity": formatting["verbosity"],
            "detail_level": formatting["detail_level"]
        }
        
        return formatted_insight 