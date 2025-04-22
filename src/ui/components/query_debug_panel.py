"""
Query Debug Panel for Watchdog AI.

This module provides a web UI component for debugging and analyzing query processing
including the schema mapping, query rewriting, and validation flow.
"""

import streamlit as st
import pandas as pd
import json
import yaml
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

from .exec_schema_profiles import (
    ExecSchemaProfile, 
    ExecRole, 
    QueryRewriter, 
    QueryPrecisionScorer, 
    BusinessRuleEngine,
    FeedbackLogger,
    create_gm_schema_profile,
    create_gsm_schema_profile
)

class QueryDebugPanel:
    """UI panel for debugging query processing."""
    
    def __init__(self, schema_profile: Optional[ExecSchemaProfile] = None, 
                rule_engine: Optional[BusinessRuleEngine] = None,
                feedback_logger: Optional[FeedbackLogger] = None):
        """Initialize with schema profile and optional components."""
        # Use default GM profile if none provided
        self.schema_profile = schema_profile or create_gm_schema_profile()
        
        # Create rule engine if none provided
        self.rule_engine = rule_engine or BusinessRuleEngine()
        
        # Initialize rewriter and scorer
        self.rewriter = QueryRewriter(self.schema_profile, self.rule_engine)
        self.scorer = QueryPrecisionScorer(self.schema_profile)
        
        # Initialize feedback logger
        self.feedback_logger = feedback_logger or FeedbackLogger()
        
        # Store debug info for current session
        self.debug_history = []
    
    def change_schema_profile(self, new_profile: ExecSchemaProfile) -> None:
        """Change the active schema profile."""
        self.schema_profile = new_profile
        # Reinitialize components with new profile
        self.rewriter = QueryRewriter(self.schema_profile, self.rule_engine)
        self.scorer = QueryPrecisionScorer(self.schema_profile)
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """
        Process a query and return debug information.
        """
        # 1. Rewrite the query
        rewritten_query, metadata = self.rewriter.rewrite_query(query)
        
        # 2. Score the rewritten query
        score_info = self.scorer.score_query(rewritten_query, metadata)
        
        # 3. Combine information for debug panel
        debug_info = {
            "original_query": query,
            "rewritten_query": rewritten_query,
            "schema_profile": self.schema_profile.name,
            "schema_role": self.schema_profile.role.value,
            "term_matches": metadata["column_matches"],
            "missing_terms": metadata["missing_terms"],
            "ambiguities": metadata.get("ambiguities", []),
            "suggested_additions": metadata.get("suggested_additions", []),
            "query_score": score_info["score"],
            "confidence_level": score_info["confidence_level"],
            "score_reasoning": score_info["reasoning"],
            "scored_adjustments": score_info.get("adjustments", {}),
            "timestamp": datetime.now().isoformat()
        }
        
        # Add to history
        self.debug_history.append(debug_info)
        
        return debug_info
    
    def render_streamlit_panel(self, debug_info: Dict[str, Any]) -> None:
        """Render a debug panel in Streamlit."""
        st.markdown("## Query Debug Panel")
        st.text(f"Generated: {datetime.fromisoformat(debug_info['timestamp']).strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Query information
        st.markdown("### Query Processing")
        cols = st.columns(2)
        with cols[0]:
            st.markdown("**Original Query:**")
            st.info(debug_info['original_query'])
        
        with cols[1]:
            st.markdown("**Rewritten Query:**")
            if debug_info['original_query'] != debug_info['rewritten_query']:
                st.success(debug_info['rewritten_query'])
            else:
                st.info("No rewriting needed")
        
        # Schema information
        st.markdown("### Schema Context")
        st.markdown(f"**Active Profile:** {debug_info['schema_profile']} ({debug_info['schema_role']})")
        
        # Query score
        score_val = debug_info['query_score']
        confidence = debug_info['confidence_level'].upper()
        score_color = "green" if score_val >= 0.7 else "orange" if score_val >= 0.4 else "red"
        
        st.markdown("### Query Reliability Assessment")
        st.markdown(f"**Score:** <span style='color:{score_color};font-weight:bold'>{score_val:.2f} ({confidence})</span>", unsafe_allow_html=True)
        st.markdown(f"**Reasoning:** {debug_info['score_reasoning']}")
        
        # Term matches
        st.markdown("### Term Analysis")
        
        # Generate tabs
        tabs = st.tabs(["Matched Terms", "Missing Terms", "Ambiguities", "Adjustments"])
        
        with tabs[0]:
            matches = debug_info["term_matches"]
            if matches:
                match_data = []
                for term, term_matches in matches.items():
                    if term_matches:
                        match_data.append({
                            "Term": term,
                            "Matched Column": term_matches[0]["column"],
                            "Confidence": f"{term_matches[0]['confidence']:.2f}"
                        })
                
                if match_data:
                    st.table(pd.DataFrame(match_data))
                else:
                    st.info("No term matches found")
            else:
                st.info("No term matches found")
        
        with tabs[1]:
            missing = debug_info["missing_terms"]
            if missing:
                st.markdown("These terms could not be matched to schema columns:")
                for term in missing:
                    st.markdown(f"- `{term}`")
            else:
                st.success("All terms successfully mapped to schema columns")
        
        with tabs[2]:
            ambiguities = debug_info.get("ambiguities", [])
            if ambiguities:
                st.markdown("These terms had multiple potential column matches:")
                for ambiguity in ambiguities:
                    st.markdown(f"**{ambiguity['term']}** could be:")
                    for candidate in ambiguity["candidates"]:
                        st.markdown(f"- {candidate['column']} ({candidate['confidence']:.2f})")
            else:
                st.success("No ambiguous terms found")
        
        with tabs[3]:
            adjustments = debug_info.get("scored_adjustments", {})
            if adjustments:
                adj_data = []
                for factor, value in adjustments.items():
                    adj_data.append({
                        "Factor": factor.replace("_", " ").title(),
                        "Adjustment": f"{value:+.2f}"
                    })
                
                st.table(pd.DataFrame(adj_data))
            else:
                st.info("No score adjustments available")
        
        # Suggested additions
        suggestions = debug_info.get("suggested_additions", [])
        if suggestions:
            st.markdown("### Suggested Additions")
            st.markdown("Consider adding these related columns to your query:")
            for suggestion in suggestions:
                st.markdown(f"- `{suggestion['column']}` ({suggestion['reason']})")
        
        # Feedback section
        st.markdown("### Query Feedback")
        cols = st.columns(3)
        
        with cols[0]:
            if st.button("👍 Good Result", key="thumbs_up"):
                self._log_feedback(debug_info, "thumbs_up")
                st.success("Thank you for the positive feedback!")
        
        with cols[1]:
            if st.button("👎 Poor Result", key="thumbs_down"):
                self._log_feedback(debug_info, "thumbs_down")
                st.warning("We'll work to improve this result!")
        
        with cols[2]:
            if st.button("✏️ Submit Correction", key="correction"):
                correction = st.text_area("Please provide your correction:", key="correction_text")
                if correction and st.button("Submit", key="submit_correction"):
                    self._log_feedback(debug_info, "correction", user_comment=correction)
                    st.success("Thank you for your correction!")
    
    def render_query_flow(self, debug_info: Dict[str, Any]) -> None:
        """Render the query flow diagram."""
        st.markdown("### Query Processing Flow")
        
        # Define the flow stages
        stages = [
            "Original Query",
            "Term Extraction",
            "Schema Matching",
            "Query Rewriting",
            "Rule Validation",
            "Confidence Scoring",
            "Final Query"
        ]
        
        # Create a flow diagram using columns
        cols = st.columns(len(stages))
        
        for i, stage in enumerate(stages):
            with cols[i]:
                st.markdown(f"**{stage}**")
                
                if stage == "Original Query":
                    st.info(debug_info['original_query'])
                    
                elif stage == "Term Extraction":
                    extracted_terms = list(debug_info["term_matches"].keys()) + debug_info["missing_terms"]
                    for term in extracted_terms:
                        st.markdown(f"- {term}")
                
                elif stage == "Schema Matching":
                    matches = debug_info["term_matches"]
                    for term, term_matches in matches.items():
                        if term_matches:
                            column = term_matches[0]["column"]
                            confidence = term_matches[0]["confidence"]
                            if confidence >= 0.7:
                                st.success(f"{term} → {column}")
                            elif confidence >= 0.4:
                                st.warning(f"{term} → {column}")
                            else:
                                st.error(f"{term} → {column}")
                    
                    for term in debug_info["missing_terms"]:
                        st.error(f"{term} → ?")
                
                elif stage == "Query Rewriting":
                    if debug_info['original_query'] != debug_info['rewritten_query']:
                        st.success("✓ Rewriting applied")
                    else:
                        st.info("No rewriting needed")
                
                elif stage == "Rule Validation":
                    # This would show any business rule validation results
                    # For now, just show a placeholder
                    st.info("Rules validated")
                
                elif stage == "Confidence Scoring":
                    score_val = debug_info['query_score']
                    confidence = debug_info['confidence_level']
                    if score_val >= 0.7:
                        st.success(f"{score_val:.2f} ({confidence})")
                    elif score_val >= 0.4:
                        st.warning(f"{score_val:.2f} ({confidence})")
                    else:
                        st.error(f"{score_val:.2f} ({confidence})")
                
                elif stage == "Final Query":
                    st.success(debug_info['rewritten_query'])
    
    def render_code_snippets(self, debug_info: Dict[str, Any]) -> None:
        """Render code snippets related to the query processing."""
        st.markdown("### Code Snippets")
        
        tabs = st.tabs(["Query Processing", "Schema Definition", "Business Rules"])
        
        with tabs[0]:
            python_code = f'''
# Process the query
rewriter = QueryRewriter(schema_profile, rule_engine)
rewritten_query, metadata = rewriter.rewrite_query("{debug_info['original_query']}")

# Score the query
scorer = QueryPrecisionScorer(schema_profile)
score_info = scorer.score_query(rewritten_query, metadata)

# Check confidence level
if score_info["confidence_level"] == "high":
    # Proceed with high confidence
    process_query(rewritten_query)
elif score_info["confidence_level"] == "medium":
    # Provide warning but proceed
    warn("Medium confidence in query interpretation")
    process_query(rewritten_query)
else:
    # Request clarification or use fallback
    handle_low_confidence(rewritten_query, metadata)
'''
            st.code(python_code, language="python")
        
        with tabs[1]:
            # Get the relevant columns for this query
            relevant_columns = set()
            for matches in debug_info["term_matches"].values():
                if matches:
                    relevant_columns.add(matches[0]["column"])
            
            yaml_code = "columns:\n"
            for col_name in relevant_columns:
                column = self.schema_profile.get_column_by_name(col_name)
                if column:
                    yaml_code += f"  {col_name}:\n"
                    yaml_code += f"    display_name: {column.display_name}\n"
                    yaml_code += f"    data_type: {column.data_type}\n"
                    if column.metric_type:
                        yaml_code += f"    metric_type: {column.metric_type}\n"
                    if column.visibility:
                        yaml_code += f"    visibility: {column.visibility}\n"
                    if column.aliases:
                        yaml_code += f"    aliases: {column.aliases}\n"
            
            st.code(yaml_code, language="yaml")
        
        with tabs[2]:
            # Show relevant business rules
            yaml_code = "business_rules:\n"
            
            # Get relevant rules for matched columns
            relevant_rules = []
            for matches in debug_info["term_matches"].values():
                if matches:
                    col_name = matches[0]["column"]
                    rules = self.rule_engine.get_rules_for_column(col_name)
                    relevant_rules.extend(rules)
            
            if relevant_rules:
                for rule in relevant_rules:
                    yaml_code += f"  - id: {rule.get('id', 'unnamed_rule')}\n"
                    yaml_code += f"    type: {rule.get('type', 'comparison')}\n"
                    yaml_code += f"    column: {rule.get('column', '')}\n"
                    
                    if 'operator' in rule:
                        yaml_code += f"    operator: '{rule['operator']}'\n"
                    
                    if 'threshold' in rule:
                        yaml_code += f"    threshold: {rule['threshold']}\n"
                    
                    if 'message' in rule:
                        yaml_code += f"    message: '{rule['message']}'\n"
            else:
                yaml_code += "  # No relevant business rules for this query"
            
            st.code(yaml_code, language="yaml")
    
    def render_fallback_reason(self, debug_info: Dict[str, Any]) -> None:
        """Render explanation for fallback scenarios."""
        score_val = debug_info['query_score']
        
        if score_val < 0.4:
            st.markdown("### ⚠️ Fallback Explanation")
            st.error("This query has low confidence and triggered a fallback mechanism.")
            
            reasons = []
            
            # Check for specific issues
            if debug_info["missing_terms"]:
                reasons.append(f"Unrecognized terms: {', '.join(debug_info['missing_terms'])}")
            
            if debug_info.get("ambiguities"):
                ambiguous_terms = [a["term"] for a in debug_info["ambiguities"]]
                reasons.append(f"Ambiguous terms: {', '.join(ambiguous_terms)}")
            
            if debug_info.get("scored_adjustments"):
                for factor, value in debug_info["scored_adjustments"].items():
                    if value < -0.1:
                        reasons.append(f"Negative adjustment from {factor}: {value:.2f}")
            
            st.markdown("**Specific Issues:**")
            for reason in reasons:
                st.markdown(f"- {reason}")
            
            st.markdown("**Suggested Improvements:**")
            st.markdown("- Try using more specific column names")
            st.markdown("- Check for typos in metric or dimension names")
            st.markdown("- Use simpler, more direct phrasing")
    
    def _log_feedback(self, debug_info: Dict[str, Any], feedback_type: str, user_comment: Optional[str] = None) -> None:
        """Log feedback from the user."""
        feedback_entry = {
            "timestamp": datetime.now().isoformat(),
            "query": debug_info["original_query"],
            "rewritten_query": debug_info["rewritten_query"],
            "schema_matches": debug_info["term_matches"],
            "success": feedback_type == "thumbs_up",
            "feedback_type": feedback_type,
            "user_comment": user_comment,
            "context_metadata": {
                "schema_profile": debug_info["schema_profile"],
                "schema_role": debug_info["schema_role"],
                "confidence_level": debug_info["confidence_level"],
                "query_score": debug_info["query_score"]
            }
        }
        
        if self.feedback_logger:
            self.feedback_logger.log_feedback(feedback_entry)

def render_debug_panel_ui():
    """Render the main debug panel UI in Streamlit."""
    st.title("Watchdog AI: Query Debug Panel")
    
    # Initialize state
    if 'debug_panel' not in st.session_state:
        # Create schema profiles
        gm_profile = create_gm_schema_profile()
        gsm_profile = create_gsm_schema_profile()
        
        # Create rule engine with default rules
        rule_engine = BusinessRuleEngine()
        rule_engine.rules = {
            "gross_not_negative": {
                "type": "comparison",
                "column": "total_gross_profit",
                "operator": ">=",
                "threshold": 0,
                "message": "Gross profit should not be negative"
            },
            "min_close_rate": {
                "type": "comparison",
                "column": "close_rate",
                "operator": ">=",
                "threshold": 8,
                "message": "Close rate should be at least 8%"
            }
        }
        
        # Create feedback logger
        feedback_logger = FeedbackLogger("query_feedback_log.json")
        
        # Initialize the debug panel
        debug_panel = QueryDebugPanel(gm_profile, rule_engine, feedback_logger)
        
        # Store all in session state
        st.session_state.debug_panel = debug_panel
        st.session_state.profiles = {
            "General Manager": gm_profile,
            "General Sales Manager": gsm_profile
        }
        st.session_state.current_profile = "General Manager"
        st.session_state.debug_history = []
    
    # Sidebar for configuration
    st.sidebar.markdown("## Configuration")
    
    # Schema profile selector
    selected_profile = st.sidebar.selectbox(
        "Executive Profile",
        options=list(st.session_state.profiles.keys()),
        index=list(st.session_state.profiles.keys()).index(st.session_state.current_profile)
    )
    
    # Update active profile if changed
    if selected_profile != st.session_state.current_profile:
        st.session_state.current_profile = selected_profile
        new_profile = st.session_state.profiles[selected_profile]
        st.session_state.debug_panel.change_schema_profile(new_profile)
    
    # View selector
    view_mode = st.sidebar.radio(
        "View Mode",
        options=["Basic", "Flow View", "Code Snippets", "Full Detail"]
    )
    
    # History viewer
    if st.session_state.debug_history:
        st.sidebar.markdown("## Query History")
        for i, debug_info in enumerate(st.session_state.debug_history):
            if st.sidebar.button(f"{i+1}. {debug_info['original_query'][:30]}...", key=f"history_{i}"):
                st.session_state.selected_history = i
    
    # Main input area
    st.markdown("## Query Input")
    
    query = st.text_input("Enter a business query:", 
                       placeholder="Example: What is our total gross profit by vehicle type?")
    
    if st.button("Process Query", key="process_button"):
        if query:
            # Process the query
            debug_info = st.session_state.debug_panel.process_query(query)
            
            # Add to history
            st.session_state.debug_history.append(debug_info)
            
            # Select this as current history item
            st.session_state.selected_history = len(st.session_state.debug_history) - 1
            
            # Force rerun to show results
            st.experimental_rerun()
    
    # Display results if available
    if hasattr(st.session_state, 'selected_history') and st.session_state.debug_history:
        debug_info = st.session_state.debug_history[st.session_state.selected_history]
        
        # Render the appropriate view
        if view_mode == "Basic" or view_mode == "Full Detail":
            st.session_state.debug_panel.render_streamlit_panel(debug_info)
        
        if view_mode == "Flow View" or view_mode == "Full Detail":
            st.session_state.debug_panel.render_query_flow(debug_info)
            
        if view_mode == "Code Snippets" or view_mode == "Full Detail":
            st.session_state.debug_panel.render_code_snippets(debug_info)
        
        # Always render fallback reason if confidence is low
        if debug_info['query_score'] < 0.4:
            st.session_state.debug_panel.render_fallback_reason(debug_info)

if __name__ == "__main__":
    render_debug_panel_ui()