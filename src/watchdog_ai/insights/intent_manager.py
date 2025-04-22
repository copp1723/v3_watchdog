"""
Intent management system.
"""

from typing import List, Optional
from .intents import Intent, TopMetricIntent, BottomMetricIntent, AverageMetricIntent

class IntentManager:
    """Manager for handling intents."""
    
    def __init__(self):
        """Initialize the intent manager."""
        self.intents: List[Intent] = [
            TopMetricIntent(),
            BottomMetricIntent(),
            AverageMetricIntent()
        ]
    
    def find_matching_intent(self, query: str) -> Optional[Intent]:
        """Find an intent that matches the query."""
        for intent in self.intents:
            if intent.matches(query):
                return intent
        return None