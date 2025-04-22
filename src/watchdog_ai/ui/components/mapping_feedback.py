"""
Column mapping feedback UI component.
"""

import streamlit as st
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Dict, Any
import logging

from ...utils.adaptive_schema import MappingSuggestion
from ...utils.data_lineage import DataLineage
from ...ui.utils.status_formatter import StatusType, format_status_text

logger = logging.getLogger(__name__)

@dataclass
class MappingFeedback:
    """Represents user feedback on a column mapping suggestion."""
    original_column: str
    suggested_column: str
    is_correct: bool
    correct_mapping: Optional[str]
    confidence: float
    user_id: str
    timestamp: str = datetime.now().isoformat()
    metadata: Dict[str, Any] = None

class MappingFeedbackUI:
    """UI component for collecting and displaying mapping feedback."""
    
    def __init__(self, lineage: DataLineage):
        """
        Initialize the mapping feedback UI.
        
        Args:
            lineage: DataLineage instance for tracking mapping history
        """
        self.lineage = lineage
        if 'mapping_feedback_submitted' not in st.session_state:
            st.session_state.mapping_feedback_submitted = set()
    
    def render_mapping_suggestions(self, suggestions: List[MappingSuggestion]) -> None:
        """
        Render mapping suggestions and collect feedback.
        
        Args:
            suggestions: List of mapping suggestions to display
        """
        if not suggestions:
            return
            
        warning_text = f"{format_status_text(StatusType.WARNING)} Schema validation found some issues:"
        st.markdown(warning_text, unsafe_allow_html=True)
        
        for i, suggestion in enumerate(suggestions, 1):
            with st.expander(f"Suggestion {i}: Map '{suggestion.source_column}' to '{suggestion.target_column}'"):
                st.write(suggestion.reason)
                if suggestion.alternatives:
                    selected = st.selectbox(
                        "Select mapping:",
                        options=[suggestion.target_column] + suggestion.alternatives,
                        key=f"mapping_suggestion_{i}"
                    )
                    if st.button("Apply", key=f"apply_mapping_{i}"):
                        self.lineage.track_column_mapping(
                            source_column=suggestion.source_column,
                            target_column=selected,
                            confidence=suggestion.confidence,
                            metadata=suggestion.metadata
                        )
                        success_text = f"{format_status_text(StatusType.SUCCESS)} Mapping applied!"
                        st.markdown(success_text, unsafe_allow_html=True)
                
    def render_feedback_history(self) -> None:
        """Render the history of mapping feedback."""
        history = self.lineage.get_mapping_history()
        if not history:
            return
            
        st.subheader("Mapping History")
        for entry in history:
            st.write(f"- {entry['timestamp']}: {entry['description']}")