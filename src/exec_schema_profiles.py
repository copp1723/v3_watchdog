"""
Executive Schema Profiles for Watchdog AI.

This module defines executive-specific schema profiles that allow for:
1. Per-persona schema loading (GM, GSM, etc.)
2. Dynamic mapping of user queries to appropriate schema elements
3. Schema-aware query rewrites for more accurate results
"""

import os
import json
import yaml
import logging
from enum import Enum
from typing import Dict, List, Set, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)

class ExecRole(str, Enum):
    """Executive roles with distinct schema profiles."""
    GENERAL_MANAGER = "general_manager"
    GENERAL_SALES_MANAGER = "general_sales_manager"
    FINANCE_MANAGER = "finance_manager"
    SERVICE_MANAGER = "service_manager"
    MARKETING_MANAGER = "marketing_manager"
    INVENTORY_MANAGER = "inventory_manager"

class ColumnVisibility(str, Enum):
    """Visibility level for schema columns."""
    PUBLIC = "public"     # Visible to all roles
    RESTRICTED = "restricted"  # Visible only to specific roles
    PRIVATE = "private"   # Visible only to highest authority roles

class MetricType(str, Enum):
    """Types of metrics available in the system."""
    FINANCIAL = "financial"
    SALES = "sales"
    INVENTORY = "inventory"
    MARKETING = "marketing"
    SERVICE = "service"
    CUSTOMER = "customer"
    OPERATIONAL = "operational"

class SchemaColumn(BaseModel):
    """Definition of a single column in the executive schema."""
    name: str
    display_name: str
    description: str
    data_type: str
    metric_type: Optional[MetricType] = None
    visibility: ColumnVisibility = ColumnVisibility.PUBLIC
    allowed_roles: List[ExecRole] = []
    aliases: List[str] = Field(default_factory=list)
    format: Optional[str] = None
    units: Optional[str] = None
    aggregations: List[str] = Field(default_factory=list)
    primary_groupings: List[str] = Field(default_factory=list)
    related_columns: List[str] = Field(default_factory=list)
    sample_queries: List[str] = Field(default_factory=list)
    business_rules: List[Dict[str, Any]] = Field(default_factory=list)
    
    @validator('allowed_roles', pre=True)
    def validate_allowed_roles(cls, v):
        """Validate that allowed_roles contains valid ExecRole values."""
        if v is None:
            return []
        
        return [role if isinstance(role, ExecRole) 
                else ExecRole(role) if isinstance(role, str) and role in [r.value for r in ExecRole]
                else ExecRole.GENERAL_MANAGER  # Default to GM if invalid
                for role in v]
    
    def is_visible_to(self, role: ExecRole) -> bool:
        """Check if this column is visible to the specified role."""
        if self.visibility == ColumnVisibility.PUBLIC:
            return True
        
        if self.visibility == ColumnVisibility.PRIVATE and role == ExecRole.GENERAL_MANAGER:
            return True
            
        return role in self.allowed_roles
    
    def matches_query_term(self, term: str) -> float:
        """
        Calculate how well this column matches a query term.
        Returns a confidence score between 0.0 and 1.0.
        """
        # 1. Exact match with column name or display name
        if term.lower() == self.name.lower() or term.lower() == self.display_name.lower():
            return 1.0
            
        # 2. Check aliases
        if any(term.lower() == alias.lower() for alias in self.aliases):
            return 0.9
            
        # 3. Partial matches
        if term.lower() in self.name.lower() or term.lower() in self.display_name.lower():
            return 0.7
            
        # 4. Partial matches in aliases
        if any(term.lower() in alias.lower() for alias in self.aliases):
            return 0.6
            
        # 5. Check description
        if term.lower() in self.description.lower():
            return 0.5
            
        # 6. Check sample queries
        if any(term.lower() in query.lower() for query in self.sample_queries):
            return 0.4
            
        return 0.0
        
class ExecSchemaProfile(BaseModel):
    """Schema profile for executive roles."""
    role: ExecRole
    name: str
    description: str
    columns: List[SchemaColumn] = Field(default_factory=list)
    default_metrics: List[str] = Field(default_factory=list)
    default_dimensions: List[str] = Field(default_factory=list)
    
    def get_visible_columns(self) -> List[SchemaColumn]:
        """Get all columns visible to this role."""
        return [col for col in self.columns if col.is_visible_to(self.role)]
    
    def get_column_by_name(self, name: str) -> Optional[SchemaColumn]:
        """Get a column by its name."""
        for col in self.columns:
            if col.name.lower() == name.lower():
                return col
        return None
    
    def find_matching_columns(self, query_terms: List[str], threshold: float = 0.4) -> Dict[str, List[Tuple[SchemaColumn, float]]]:
        """
        Find columns that match the query terms.
        Returns a dictionary mapping query terms to lists of (column, confidence) tuples.
        """
        results = {}
        
        for term in query_terms:
            term_matches = []
            
            for col in self.get_visible_columns():
                confidence = col.matches_query_term(term)
                if confidence >= threshold:
                    term_matches.append((col, confidence))
            
            if term_matches:
                # Sort by confidence score
                term_matches.sort(key=lambda x: x[1], reverse=True)
                results[term] = term_matches
                
        return results
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "role": self.role.value,
            "name": self.name,
            "description": self.description,
            "columns": [col.dict() for col in self.columns],
            "default_metrics": self.default_metrics,
            "default_dimensions": self.default_dimensions
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExecSchemaProfile':
        """Create from dictionary."""
        # Convert columns to SchemaColumn objects
        if "columns" in data:
            data["columns"] = [
                col if isinstance(col, SchemaColumn) else SchemaColumn(**col)
                for col in data["columns"]
            ]
        
        # Ensure role is an ExecRole
        if "role" in data and isinstance(data["role"], str):
            data["role"] = ExecRole(data["role"])
            
        return cls(**data)

class BusinessRuleEngine:
    """Engine for evaluating business rules against data."""
    
    def __init__(self, rules_file: Optional[str] = None):
        """Initialize the business rule engine with an optional rules file."""
        self.rules = {}
        
        if rules_file and os.path.exists(rules_file):
            self.load_rules(rules_file)
    
    def load_rules(self, file_path: str) -> bool:
        """Load rules from a YAML or JSON file."""
        try:
            with open(file_path, 'r') as f:
                ext = os.path.splitext(file_path)[1].lower()
                
                if ext == '.yaml' or ext == '.yml':
                    self.rules = yaml.safe_load(f)
                else:  # Assume JSON
                    self.rules = json.load(f)
                    
            logger.info(f"Loaded {len(self.rules)} business rules from {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load business rules from {file_path}: {str(e)}")
            return False
    
    def add_rule(self, rule_id: str, rule_def: Dict[str, Any]) -> None:
        """Add or update a business rule."""
        self.rules[rule_id] = rule_def
        
    def get_rules_for_column(self, column_name: str) -> List[Dict[str, Any]]:
        """Get all rules applicable to a specific column."""
        return [rule for rule_id, rule in self.rules.items() 
                if rule.get('column') == column_name]
    
    def evaluate_rule(self, rule_id: str, data: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Evaluate a single business rule against data.
        Returns (is_valid, message) tuple.
        """
        if rule_id not in self.rules:
            return False, f"Rule {rule_id} not found"
            
        rule = self.rules[rule_id]
        rule_type = rule.get('type', 'comparison')
        
        try:
            if rule_type == 'comparison':
                column = rule.get('column')
                operator = rule.get('operator', '==')
                threshold = rule.get('threshold')
                
                if column not in data:
                    return False, f"Column {column} not found in data"
                
                value = data[column]
                
                result = self._evaluate_comparison(value, operator, threshold)
                message = rule.get('message', f"Rule {rule_id} {'passed' if result else 'failed'}")
                
                return result, message
                
            elif rule_type == 'range':
                column = rule.get('column')
                min_val = rule.get('min_value')
                max_val = rule.get('max_value')
                
                if column not in data:
                    return False, f"Column {column} not found in data"
                
                value = data[column]
                
                in_range = (min_val is None or value >= min_val) and (max_val is None or value <= max_val)
                message = rule.get('message', f"Value {'in' if in_range else 'outside'} valid range")
                
                return in_range, message
                
            elif rule_type == 'regexp':
                column = rule.get('column')
                pattern = rule.get('pattern')
                
                if column not in data:
                    return False, f"Column {column} not found in data"
                
                import re
                value = str(data[column])
                matches = bool(re.match(pattern, value))
                
                message = rule.get('message', f"Value {'matches' if matches else 'does not match'} pattern")
                return matches, message
                
            else:
                return False, f"Unsupported rule type: {rule_type}"
                
        except Exception as e:
            return False, f"Error evaluating rule {rule_id}: {str(e)}"
    
    def _evaluate_comparison(self, value: Any, operator: str, threshold: Any) -> bool:
        """Evaluate a comparison operation."""
        if operator == '==':
            return value == threshold
        elif operator == '!=':
            return value != threshold
        elif operator == '>':
            return value > threshold
        elif operator == '>=':
            return value >= threshold
        elif operator == '<':
            return value < threshold
        elif operator == '<=':
            return value <= threshold
        elif operator == 'in':
            return value in threshold
        elif operator == 'not in':
            return value not in threshold
        else:
            raise ValueError(f"Unsupported operator: {operator}")

class QueryRewriter:
    """Rewrites user queries to be schema-compatible."""
    
    def __init__(self, schema_profile: ExecSchemaProfile, rule_engine: Optional[BusinessRuleEngine] = None):
        """Initialize with a schema profile and optional rule engine."""
        self.schema_profile = schema_profile
        self.rule_engine = rule_engine
        
    def rewrite_query(self, query: str) -> Tuple[str, Dict[str, Any]]:
        """
        Rewrite a user query to be schema-compatible.
        Returns the rewritten query and metadata about the rewrite.
        """
        # 1. Extract query terms
        query_terms = self._extract_query_terms(query)
        
        # 2. Match terms to schema columns
        column_matches = self.schema_profile.find_matching_columns(query_terms)
        
        # 3. Perform rewrites
        rewritten_query = query
        metadata = {
            "original_query": query,
            "extracted_terms": query_terms,
            "column_matches": {},
            "rewrites": [],
            "ambiguities": [],
            "missing_terms": [term for term in query_terms if term not in column_matches]
        }
        
        # Record column matches in metadata
        for term, matches in column_matches.items():
            metadata["column_matches"][term] = [
                {"column": col.name, "confidence": conf} 
                for col, conf in matches
            ]
        
        # Handle each term
        for term, matches in column_matches.items():
            if not matches:
                continue
                
            best_match_col, confidence = matches[0]
            
            # Check for ambiguity (multiple high-confidence matches)
            ambiguous = len([m for m, c in matches if c > 0.7]) > 1
            if ambiguous:
                metadata["ambiguities"].append({
                    "term": term,
                    "candidates": [{"column": col.name, "confidence": conf} for col, conf in matches if conf > 0.7]
                })
            
            # Perform the rewrite if confidence is high enough
            if confidence > 0.7:
                if best_match_col.display_name != term:
                    old_term = term
                    new_term = best_match_col.display_name
                    
                    # Replace the term with the display name
                    rewritten_query = self._replace_term(rewritten_query, old_term, new_term)
                    
                    metadata["rewrites"].append({
                        "original": old_term,
                        "rewritten": new_term,
                        "column": best_match_col.name,
                        "confidence": confidence
                    })
        
        # 4. Apply business rules to ensure query makes sense
        if self.rule_engine:
            # Check for related columns that should be included
            all_matched_columns = {col.name for term_matches in column_matches.values() 
                                 for col, _ in term_matches if col.name}
            
            for col_name in all_matched_columns:
                col = self.schema_profile.get_column_by_name(col_name)
                if col and col.related_columns:
                    # Check if related columns should be added
                    for related_col_name in col.related_columns:
                        related_col = self.schema_profile.get_column_by_name(related_col_name)
                        if related_col and not any(rewrite.get("column") == related_col.name for rewrite in metadata.get("rewrites", [])):
                            # Consider adding this related column to query if applicable
                            metadata["suggested_additions"] = metadata.get("suggested_additions", [])
                            metadata["suggested_additions"].append({
                                "column": related_col.name,
                                "reason": f"Related to {col.name}"
                            })
        
        return rewritten_query, metadata
    
    def _extract_query_terms(self, query: str) -> List[str]:
        """Extract meaningful terms from a query."""
        # Tokenize and remove stopwords
        import re
        from collections import Counter
        
        # Simple preprocessing
        query = query.lower()
        
        # Split into words and keep only alphanumeric words
        words = re.findall(r'\b[a-z0-9]+\b', query)
        
        # Remove common stopwords
        stopwords = {'and', 'or', 'the', 'in', 'on', 'at', 'by', 'for', 'with', 'about', 
                    'from', 'to', 'of', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 
                    'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'but',
                    'me', 'show', 'tell', 'give', 'find', 'what', 'where', 'when', 
                    'which', 'who', 'whom', 'why', 'how'}
        
        filtered_words = [word for word in words if word not in stopwords and len(word) > 2]
        
        # Add potential multi-word terms based on column names and aliases
        all_columns = self.schema_profile.get_visible_columns()
        multi_word_terms = []
        
        for col in all_columns:
            # Check column name
            if ' ' in col.name:
                lower_name = col.name.lower()
                if lower_name in query:
                    multi_word_terms.append(col.name)
            
            # Check display name
            if ' ' in col.display_name:
                lower_display = col.display_name.lower()
                if lower_display in query:
                    multi_word_terms.append(col.display_name)
            
            # Check aliases
            for alias in col.aliases:
                if ' ' in alias and alias.lower() in query:
                    multi_word_terms.append(alias)
        
        # Combine single words and multi-word terms
        all_terms = filtered_words + multi_word_terms
        
        # Count term frequency and keep most common
        term_counter = Counter(all_terms)
        significant_terms = [term for term, count in term_counter.most_common(10)]
        
        return significant_terms
    
    def _replace_term(self, text: str, old_term: str, new_term: str) -> str:
        """Replace a term in text, handling case variants."""
        import re
        
        # Different case variants to handle
        patterns = [
            re.escape(old_term),                   # Exact match
            re.escape(old_term.lower()),           # Lowercase
            re.escape(old_term.upper()),           # Uppercase
            re.escape(old_term.capitalize())       # Capitalized
        ]
        
        # Build regex to match any of these patterns
        combined_pattern = '|'.join(f'({p})' for p in patterns)
        
        # Replace all occurrences
        replaced = re.sub(combined_pattern, new_term, text)
        return replaced

class QueryPrecisionScorer:
    """Scores queries for reliability prediction."""
    
    def __init__(self, schema_profile: ExecSchemaProfile, history_file: Optional[str] = None):
        """Initialize with schema profile and optional history file."""
        self.schema_profile = schema_profile
        self.query_history = {}
        
        if history_file and os.path.exists(history_file):
            self.load_history(history_file)
    
    def load_history(self, file_path: str) -> None:
        """Load query history from a file."""
        try:
            with open(file_path, 'r') as f:
                self.query_history = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load query history: {str(e)}")
    
    def save_history(self, file_path: str) -> None:
        """Save query history to a file."""
        try:
            with open(file_path, 'w') as f:
                json.dump(self.query_history, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save query history: {str(e)}")
    
    def score_query(self, query: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Score a query for reliability prediction.
        Returns a dict with score and reasoning.
        """
        # Base score starts at 0.5 (neutral)
        base_score = 0.5
        adjustments = []
        
        # 1. Token overlap with schema
        overlap_score = self._calculate_schema_overlap(metadata)
        adjustments.append(("schema_overlap", overlap_score))
        
        # 2. Historical success rate (if available)
        if query in self.query_history:
            history_score = self._calculate_history_score(query)
            adjustments.append(("historical_performance", history_score))
        
        # 3. Ambiguity penalty
        ambiguity_count = len(metadata.get("ambiguities", []))
        ambiguity_score = -0.1 * min(ambiguity_count, 3)  # Cap at -0.3
        adjustments.append(("ambiguity", ambiguity_score))
        
        # 4. Missing terms penalty
        missing_terms = len(metadata.get("missing_terms", []))
        missing_score = -0.05 * min(missing_terms, 4)  # Cap at -0.2
        adjustments.append(("missing_terms", missing_score))
        
        # Calculate final score (clamped between 0.0 and 1.0)
        final_score = base_score
        for _, adjustment in adjustments:
            final_score += adjustment
        
        final_score = max(0.0, min(1.0, final_score))
        
        # Convert to confidence level
        confidence_level = "high" if final_score >= 0.7 else "medium" if final_score >= 0.4 else "low"
        
        return {
            "score": final_score,
            "confidence_level": confidence_level,
            "adjustments": dict(adjustments),
            "reasoning": self._generate_reasoning(adjustments)
        }
    
    def _calculate_schema_overlap(self, metadata: Dict[str, Any]) -> float:
        """Calculate score adjustment based on schema overlap."""
        # Count how many terms were successfully matched to columns
        matched_terms = len(metadata.get("column_matches", {}))
        total_terms = matched_terms + len(metadata.get("missing_terms", []))
        
        if total_terms == 0:
            return 0.0
            
        # Calculate percentage of terms that matched schema columns
        match_percentage = matched_terms / total_terms
        
        # Calculate confidence scores of matches
        confidence_sum = sum(m[0]["confidence"] 
                           for m in metadata.get("column_matches", {}).values() 
                           if m)
        avg_confidence = confidence_sum / matched_terms if matched_terms > 0 else 0
        
        # Combine match percentage and confidence
        return (match_percentage * 0.3) + (avg_confidence * 0.2)
    
    def _calculate_history_score(self, query: str) -> float:
        """Calculate score adjustment based on query history."""
        history = self.query_history.get(query, {})
        success_count = history.get("success_count", 0)
        fail_count = history.get("fail_count", 0)
        fallback_count = history.get("fallback_count", 0)
        
        total_count = success_count + fail_count + fallback_count
        
        if total_count == 0:
            return 0.0
            
        # Success rate adjustment: +0.3 for perfect, scaled down for less success
        success_rate = success_count / total_count
        success_adjustment = success_rate * 0.3
        
        # Failure rate adjustment: -0.3 for all failures, scaled down for fewer
        failure_rate = fail_count / total_count
        failure_adjustment = -failure_rate * 0.3
        
        # Fallback rate adjustment: -0.15 for all fallbacks, scaled down for fewer
        fallback_rate = fallback_count / total_count
        fallback_adjustment = -fallback_rate * 0.15
        
        return success_adjustment + failure_adjustment + fallback_adjustment
    
    def _generate_reasoning(self, adjustments: List[Tuple[str, float]]) -> str:
        """Generate an explanation of the score."""
        reasons = []
        
        for factor, value in adjustments:
            if abs(value) < 0.01:  # Ignore negligible factors
                continue
                
            if factor == "schema_overlap":
                if value > 0.2:
                    reasons.append("Strong schema column matches")
                elif value > 0:
                    reasons.append("Some schema column matches")
                else:
                    reasons.append("Poor schema column matching")
                    
            elif factor == "historical_performance":
                if value > 0.1:
                    reasons.append("Strong historical performance")
                elif value > 0:
                    reasons.append("Moderate historical success")
                elif value < -0.1:
                    reasons.append("Poor historical performance")
                    
            elif factor == "ambiguity":
                if value < -0.1:
                    reasons.append("Contains ambiguous terms")
                    
            elif factor == "missing_terms":
                if value < -0.05:
                    reasons.append("Contains unrecognized terms")
        
        return ", ".join(reasons)
    
    def record_query_result(self, query: str, success: bool, used_fallback: bool) -> None:
        """Record the success/failure of a query."""
        if query not in self.query_history:
            self.query_history[query] = {
                "success_count": 0,
                "fail_count": 0,
                "fallback_count": 0,
                "first_seen": datetime.now().isoformat()
            }
            
        history = self.query_history[query]
        
        if success:
            history["success_count"] += 1
        else:
            history["fail_count"] += 1
            
        if used_fallback:
            history["fallback_count"] += 1
            
        history["last_seen"] = datetime.now().isoformat()

class FeedbackLogEntry(BaseModel):
    """Entry in the feedback log."""
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    user_id: Optional[str] = None
    query: str
    rewritten_query: Optional[str] = None
    schema_matches: Dict[str, Any] = Field(default_factory=dict)
    success: bool = True
    feedback_type: str = "thumbs_up"  # thumbs_up, thumbs_down, correction
    user_comment: Optional[str] = None
    correction_details: Optional[Dict[str, Any]] = None
    context_metadata: Dict[str, Any] = Field(default_factory=dict)

class FeedbackLogger:
    """Logs and manages user feedback for query processing."""
    
    def __init__(self, log_file: Optional[str] = None):
        """Initialize with an optional log file path."""
        self.log_file = log_file
        self.entries = []
        
        if log_file and os.path.exists(log_file):
            self.load_log()
    
    def load_log(self) -> None:
        """Load feedback entries from the log file."""
        try:
            with open(self.log_file, 'r') as f:
                data = json.load(f)
                self.entries = [FeedbackLogEntry(**entry) for entry in data]
        except Exception as e:
            logger.error(f"Failed to load feedback log: {str(e)}")
    
    def save_log(self) -> None:
        """Save feedback entries to the log file."""
        if not self.log_file:
            logger.warning("No log file specified, feedback will not be persisted")
            return
            
        try:
            with open(self.log_file, 'w') as f:
                json.dump([entry.dict() for entry in self.entries], f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save feedback log: {str(e)}")
    
    def log_feedback(self, entry: Union[FeedbackLogEntry, Dict[str, Any]]) -> None:
        """Add a feedback entry to the log."""
        if not isinstance(entry, FeedbackLogEntry):
            entry = FeedbackLogEntry(**entry)
            
        self.entries.append(entry)
        
        # Auto-save if a log file is specified
        if self.log_file:
            self.save_log()
    
    def get_feedback_for_query(self, query: str) -> List[FeedbackLogEntry]:
        """Get all feedback entries for a specific query."""
        return [entry for entry in self.entries if entry.query == query]
    
    def get_feedback_stats(self) -> Dict[str, Any]:
        """Get statistics about the feedback."""
        total_entries = len(self.entries)
        
        if total_entries == 0:
            return {
                "total_entries": 0,
                "success_rate": 0,
                "feedback_types": {}
            }
            
        success_count = sum(1 for entry in self.entries if entry.success)
        
        feedback_types = {}
        for entry in self.entries:
            feedback_types[entry.feedback_type] = feedback_types.get(entry.feedback_type, 0) + 1
            
        return {
            "total_entries": total_entries,
            "success_rate": success_count / total_entries,
            "feedback_types": feedback_types
        }

class QueryDebugPanel:
    """UI panel for debugging query processing."""
    
    def __init__(self, schema_profile: ExecSchemaProfile, rewriter: QueryRewriter, scorer: QueryPrecisionScorer):
        """Initialize with required components."""
        self.schema_profile = schema_profile
        self.rewriter = rewriter
        self.scorer = scorer
        self.debug_logs = []
    
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
            "timestamp": datetime.now().isoformat()
        }
        
        # Add to history
        self.debug_logs.append(debug_info)
        
        return debug_info
    
    def render_html(self, debug_info: Dict[str, Any]) -> str:
        """Render debug information as HTML."""
        html = "<div class='query-debug-panel'>"
        
        # Header
        html += f"<h3>Query Debug Panel</h3>"
        html += f"<div class='timestamp'>Generated: {debug_info['timestamp']}</div>"
        
        # Query information
        html += "<div class='query-section'>"
        html += f"<div class='original-query'><strong>Original Query:</strong> {debug_info['original_query']}</div>"
        html += f"<div class='rewritten-query'><strong>Rewritten Query:</strong> {debug_info['rewritten_query']}</div>"
        html += "</div>"
        
        # Schema information
        html += "<div class='schema-section'>"
        html += f"<div><strong>Schema Profile:</strong> {debug_info['schema_profile']} ({debug_info['schema_role']})</div>"
        html += "</div>"
        
        # Term matches
        html += "<div class='matches-section'>"
        html += "<h4>Term Matches</h4>"
        html += "<table border='1'><tr><th>Term</th><th>Matched Column</th><th>Confidence</th></tr>"
        
        for term, matches in debug_info["term_matches"].items():
            if matches:
                html += f"<tr><td>{term}</td><td>{matches[0]['column']}</td><td>{matches[0]['confidence']:.2f}</td></tr>"
        
        html += "</table>"
        html += "</div>"
        
        # Missing terms
        if debug_info["missing_terms"]:
            html += "<div class='missing-section'>"
            html += "<h4>Unrecognized Terms</h4>"
            html += "<ul>"
            for term in debug_info["missing_terms"]:
                html += f"<li>{term}</li>"
            html += "</ul>"
            html += "</div>"
        
        # Ambiguities
        if debug_info["ambiguities"]:
            html += "<div class='ambiguities-section'>"
            html += "<h4>Ambiguous Terms</h4>"
            html += "<ul>"
            for ambiguity in debug_info["ambiguities"]:
                html += f"<li>{ambiguity['term']} - Possible matches: "
                html += ", ".join(f"{c['column']} ({c['confidence']:.2f})" for c in ambiguity["candidates"])
                html += "</li>"
            html += "</ul>"
            html += "</div>"
        
        # Score information
        html += "<div class='score-section'>"
        html += f"<h4>Query Reliability Score: {debug_info['query_score']:.2f} ({debug_info['confidence_level'].upper()})</h4>"
        html += f"<div><strong>Reasoning:</strong> {debug_info['score_reasoning']}</div>"
        html += "</div>"
        
        html += "</div>"  # End panel
        
        return html

def create_gm_schema_profile() -> ExecSchemaProfile:
    """Create a schema profile for the General Manager role."""
    columns = [
        SchemaColumn(
            name="total_gross_profit",
            display_name="Total Gross Profit",
            description="Total gross profit across all departments",
            data_type="decimal",
            metric_type=MetricType.FINANCIAL,
            visibility=ColumnVisibility.PUBLIC,
            aliases=["total gross", "gross profit", "total gp", "profit"],
            format="currency",
            units="$",
            aggregations=["sum", "avg", "min", "max"],
            primary_groupings=["date", "salesperson", "department"],
            business_rules=[
                {"type": "comparison", "operator": ">=", "threshold": 0, 
                 "message": "Gross profit should not be negative"}
            ],
            sample_queries=[
                "What is our total gross profit?",
                "Show me gross profit by department",
                "What's the average gross profit per sale?"
            ]
        ),
        SchemaColumn(
            name="frontend_gross",
            display_name="Front End Gross",
            description="Gross profit from the vehicle sale before F&I products",
            data_type="decimal",
            metric_type=MetricType.FINANCIAL,
            visibility=ColumnVisibility.PUBLIC,
            aliases=["front end", "front gross", "vehicle gross"],
            format="currency",
            units="$",
            aggregations=["sum", "avg", "min", "max"],
            primary_groupings=["date", "salesperson", "vehicle_type"],
            related_columns=["total_gross_profit", "backend_gross"],
            sample_queries=[
                "What is our front end gross profit?",
                "Front gross profit by salesperson",
                "Average front end gross for new vehicles"
            ]
        ),
        SchemaColumn(
            name="backend_gross",
            display_name="Back End Gross",
            description="Gross profit from F&I products and services",
            data_type="decimal",
            metric_type=MetricType.FINANCIAL,
            visibility=ColumnVisibility.PUBLIC,
            aliases=["back end", "back gross", "finance gross", "f&i gross"],
            format="currency",
            units="$",
            aggregations=["sum", "avg", "min", "max"],
            primary_groupings=["date", "finance_manager", "product_type"],
            related_columns=["total_gross_profit", "frontend_gross"],
            sample_queries=[
                "What is our back end gross profit?",
                "Back gross by finance manager",
                "Average F&I gross per deal"
            ]
        ),
        SchemaColumn(
            name="gross_profit_per_unit",
            display_name="Gross Profit Per Unit",
            description="Average gross profit per vehicle sold",
            data_type="decimal",
            metric_type=MetricType.FINANCIAL,
            visibility=ColumnVisibility.PUBLIC,
            aliases=["PVR", "PRU", "per vehicle retailed", "grosser"],
            format="currency",
            units="$",
            aggregations=["avg"],
            primary_groupings=["date", "vehicle_type", "model"],
            related_columns=["total_gross_profit", "units_sold"],
            business_rules=[
                {"type": "comparison", "operator": ">=", "threshold": 500, 
                 "message": "Gross profit per unit should meet minimum target"}
            ],
            sample_queries=[
                "What is our average PVR?",
                "Gross per unit by model",
                "Show me PVR trend over time"
            ]
        ),
        SchemaColumn(
            name="units_sold",
            display_name="Units Sold",
            description="Number of vehicles sold",
            data_type="integer",
            metric_type=MetricType.SALES,
            visibility=ColumnVisibility.PUBLIC,
            aliases=["sales count", "vehicles sold", "unit sales", "sales"],
            aggregations=["sum", "count"],
            primary_groupings=["date", "vehicle_type", "salesperson"],
            related_columns=["gross_profit_per_unit", "total_gross_profit"],
            sample_queries=[
                "How many units did we sell?",
                "Units sold by salesperson",
                "Monthly unit sales trend"
            ]
        ),
        SchemaColumn(
            name="revenue",
            display_name="Revenue",
            description="Total revenue from all sources",
            data_type="decimal",
            metric_type=MetricType.FINANCIAL,
            visibility=ColumnVisibility.PUBLIC,
            aliases=["total revenue", "sales revenue", "income"],
            format="currency",
            units="$",
            aggregations=["sum", "avg"],
            primary_groupings=["date", "department", "revenue_source"],
            sample_queries=[
                "What is our total revenue?",
                "Revenue by department",
                "Monthly revenue trend"
            ]
        ),
        SchemaColumn(
            name="salesperson",
            display_name="Salesperson",
            description="Sales representative who handled the transaction",
            data_type="string",
            visibility=ColumnVisibility.PUBLIC,
            aliases=["sales rep", "representative", "sales consultant"],
            primary_groupings=["team", "experience_level"],
            related_columns=["units_sold", "frontend_gross"],
            sample_queries=[
                "Performance by salesperson",
                "Top salespeople",
                "Who sold the most units?"
            ]
        ),
        SchemaColumn(
            name="vehicle_type",
            display_name="Vehicle Type",
            description="Type of vehicle (New, Used, CPO)",
            data_type="string",
            metric_type=MetricType.INVENTORY,
            visibility=ColumnVisibility.PUBLIC,
            aliases=["inventory type", "new/used", "vehicle category"],
            primary_groupings=["make", "model"],
            related_columns=["units_sold", "frontend_gross"],
            sample_queries=[
                "Performance by vehicle type",
                "New vs used sales",
                "CPO sales performance"
            ]
        ),
        SchemaColumn(
            name="days_in_inventory",
            display_name="Days in Inventory",
            description="Average number of days vehicles are in inventory before sale",
            data_type="integer",
            metric_type=MetricType.INVENTORY,
            visibility=ColumnVisibility.PUBLIC,
            aliases=["aging", "inventory age", "days supply"],
            aggregations=["avg", "min", "max"],
            primary_groupings=["vehicle_type", "make", "model"],
            sample_queries=[
                "Average days in inventory",
                "Aging by model",
                "Vehicles with highest days in inventory"
            ]
        ),
        SchemaColumn(
            name="lead_source",
            display_name="Lead Source",
            description="Origin of the customer lead",
            data_type="string",
            metric_type=MetricType.MARKETING,
            visibility=ColumnVisibility.PUBLIC,
            aliases=["source", "traffic source", "lead origin"],
            primary_groupings=["campaign", "channel"],
            related_columns=["conversion_rate", "marketing_spend"],
            sample_queries=[
                "Performance by lead source",
                "Which lead sources have highest conversion?",
                "Lead distribution by source"
            ]
        ),
        # Special GM-only financial columns
        SchemaColumn(
            name="net_profit",
            display_name="Net Profit",
            description="Profit after all expenses",
            data_type="decimal",
            metric_type=MetricType.FINANCIAL,
            visibility=ColumnVisibility.PRIVATE,
            allowed_roles=[ExecRole.GENERAL_MANAGER],
            aliases=["bottom line", "net income", "after expenses", "actual profit"],
            format="currency",
            units="$",
            aggregations=["sum", "avg"],
            primary_groupings=["date", "department"],
            related_columns=["total_gross_profit", "expenses"],
            sample_queries=[
                "What is our net profit?",
                "Net profit by department",
                "Monthly net profit trend"
            ]
        ),
        SchemaColumn(
            name="expenses",
            display_name="Expenses",
            description="Total expenses across all departments",
            data_type="decimal",
            metric_type=MetricType.FINANCIAL,
            visibility=ColumnVisibility.PRIVATE,
            allowed_roles=[ExecRole.GENERAL_MANAGER],
            aliases=["costs", "overhead", "operating expenses", "opex"],
            format="currency",
            units="$",
            aggregations=["sum", "avg"],
            primary_groupings=["date", "expense_category", "department"],
            related_columns=["net_profit", "total_gross_profit"],
            sample_queries=[
                "What are our total expenses?",
                "Expenses by category",
                "Monthly expense trend"
            ]
        ),
        SchemaColumn(
            name="salary_data",
            display_name="Salary Information",
            description="Employee salary and commission data",
            data_type="decimal",
            metric_type=MetricType.FINANCIAL,
            visibility=ColumnVisibility.PRIVATE,
            allowed_roles=[ExecRole.GENERAL_MANAGER],
            aliases=["pay", "compensation", "commission", "earnings"],
            format="currency",
            units="$",
            aggregations=["sum", "avg"],
            primary_groupings=["employee", "department", "position"],
            sample_queries=[
                "Total salary expenses",
                "Average commission by salesperson",
                "Compensation analysis"
            ]
        )
    ]
    
    return ExecSchemaProfile(
        role=ExecRole.GENERAL_MANAGER,
        name="General Manager Schema Profile",
        description="Complete data access for dealership GM with all metrics and confidential financial data",
        columns=columns,
        default_metrics=["total_gross_profit", "units_sold", "revenue", "net_profit"],
        default_dimensions=["vehicle_type", "department", "date"]
    )

def create_gsm_schema_profile() -> ExecSchemaProfile:
    """Create a schema profile for the General Sales Manager role."""
    columns = [
        SchemaColumn(
            name="total_gross_profit",
            display_name="Total Gross Profit",
            description="Total gross profit across all departments",
            data_type="decimal",
            metric_type=MetricType.FINANCIAL,
            visibility=ColumnVisibility.PUBLIC,
            aliases=["total gross", "gross profit", "total gp", "profit"],
            format="currency",
            units="$",
            aggregations=["sum", "avg", "min", "max"],
            primary_groupings=["date", "salesperson", "department"],
            business_rules=[
                {"type": "comparison", "operator": ">=", "threshold": 0, 
                 "message": "Gross profit should not be negative"}
            ],
            sample_queries=[
                "What is our total gross profit?",
                "Show me gross profit by department",
                "What's the average gross profit per sale?"
            ]
        ),
        SchemaColumn(
            name="frontend_gross",
            display_name="Front End Gross",
            description="Gross profit from the vehicle sale before F&I products",
            data_type="decimal",
            metric_type=MetricType.FINANCIAL,
            visibility=ColumnVisibility.PUBLIC,
            aliases=["front end", "front gross", "vehicle gross"],
            format="currency",
            units="$",
            aggregations=["sum", "avg", "min", "max"],
            primary_groupings=["date", "salesperson", "vehicle_type"],
            related_columns=["total_gross_profit", "backend_gross"],
            sample_queries=[
                "What is our front end gross profit?",
                "Front gross profit by salesperson",
                "Average front end gross for new vehicles"
            ]
        ),
        SchemaColumn(
            name="backend_gross",
            display_name="Back End Gross",
            description="Gross profit from F&I products and services",
            data_type="decimal",
            metric_type=MetricType.FINANCIAL,
            visibility=ColumnVisibility.PUBLIC,
            aliases=["back end", "back gross", "finance gross", "f&i gross"],
            format="currency",
            units="$",
            aggregations=["sum", "avg", "min", "max"],
            primary_groupings=["date", "finance_manager", "product_type"],
            related_columns=["total_gross_profit", "frontend_gross"],
            sample_queries=[
                "What is our back end gross profit?",
                "Back gross by finance manager",
                "Average F&I gross per deal"
            ]
        ),
        SchemaColumn(
            name="gross_profit_per_unit",
            display_name="Gross Profit Per Unit",
            description="Average gross profit per vehicle sold",
            data_type="decimal",
            metric_type=MetricType.FINANCIAL,
            visibility=ColumnVisibility.PUBLIC,
            aliases=["PVR", "PRU", "per vehicle retailed", "grosser"],
            format="currency",
            units="$",
            aggregations=["avg"],
            primary_groupings=["date", "vehicle_type", "model"],
            related_columns=["total_gross_profit", "units_sold"],
            business_rules=[
                {"type": "comparison", "operator": ">=", "threshold": 500, 
                 "message": "Gross profit per unit should meet minimum target"}
            ],
            sample_queries=[
                "What is our average PVR?",
                "Gross per unit by model",
                "Show me PVR trend over time"
            ]
        ),
        SchemaColumn(
            name="units_sold",
            display_name="Units Sold",
            description="Number of vehicles sold",
            data_type="integer",
            metric_type=MetricType.SALES,
            visibility=ColumnVisibility.PUBLIC,
            aliases=["sales count", "vehicles sold", "unit sales", "sales"],
            aggregations=["sum", "count"],
            primary_groupings=["date", "vehicle_type", "salesperson"],
            related_columns=["gross_profit_per_unit", "total_gross_profit"],
            sample_queries=[
                "How many units did we sell?",
                "Units sold by salesperson",
                "Monthly unit sales trend"
            ]
        ),
        SchemaColumn(
            name="revenue",
            display_name="Revenue",
            description="Total revenue from all sources",
            data_type="decimal",
            metric_type=MetricType.FINANCIAL,
            visibility=ColumnVisibility.PUBLIC,
            aliases=["total revenue", "sales revenue", "income"],
            format="currency",
            units="$",
            aggregations=["sum", "avg"],
            primary_groupings=["date", "department", "revenue_source"],
            sample_queries=[
                "What is our total revenue?",
                "Revenue by department",
                "Monthly revenue trend"
            ]
        ),
        SchemaColumn(
            name="salesperson",
            display_name="Salesperson",
            description="Sales representative who handled the transaction",
            data_type="string",
            visibility=ColumnVisibility.PUBLIC,
            aliases=["sales rep", "representative", "sales consultant"],
            primary_groupings=["team", "experience_level"],
            related_columns=["units_sold", "frontend_gross"],
            sample_queries=[
                "Performance by salesperson",
                "Top salespeople",
                "Who sold the most units?"
            ]
        ),
        SchemaColumn(
            name="vehicle_type",
            display_name="Vehicle Type",
            description="Type of vehicle (New, Used, CPO)",
            data_type="string",
            metric_type=MetricType.INVENTORY,
            visibility=ColumnVisibility.PUBLIC,
            aliases=["inventory type", "new/used", "vehicle category"],
            primary_groupings=["make", "model"],
            related_columns=["units_sold", "frontend_gross"],
            sample_queries=[
                "Performance by vehicle type",
                "New vs used sales",
                "CPO sales performance"
            ]
        ),
        SchemaColumn(
            name="days_in_inventory",
            display_name="Days in Inventory",
            description="Average number of days vehicles are in inventory before sale",
            data_type="integer",
            metric_type=MetricType.INVENTORY,
            visibility=ColumnVisibility.PUBLIC,
            aliases=["aging", "inventory age", "days supply"],
            aggregations=["avg", "min", "max"],
            primary_groupings=["vehicle_type", "make", "model"],
            sample_queries=[
                "Average days in inventory",
                "Aging by model",
                "Vehicles with highest days in inventory"
            ]
        ),
        SchemaColumn(
            name="lead_source",
            display_name="Lead Source",
            description="Origin of the customer lead",
            data_type="string",
            metric_type=MetricType.MARKETING,
            visibility=ColumnVisibility.PUBLIC,
            aliases=["source", "traffic source", "lead origin"],
            primary_groupings=["campaign", "channel"],
            related_columns=["conversion_rate", "marketing_spend"],
            sample_queries=[
                "Performance by lead source",
                "Which lead sources have highest conversion?",
                "Lead distribution by source"
            ]
        ),
        # Sales-specific columns that GM doesn't focus on
        SchemaColumn(
            name="close_rate",
            display_name="Close Rate",
            description="Percentage of leads that convert to sales",
            data_type="decimal",
            metric_type=MetricType.SALES,
            visibility=ColumnVisibility.PUBLIC,
            aliases=["conversion rate", "closing percentage", "lead conversion"],
            format="percentage",
            units="%",
            aggregations=["avg"],
            primary_groupings=["lead_source", "salesperson", "vehicle_type"],
            business_rules=[
                {"type": "comparison", "operator": ">=", "threshold": 8, 
                 "message": "Close rate should meet minimum target percentage"}
            ],
            sample_queries=[
                "What is our overall close rate?",
                "Close rate by salesperson",
                "Which lead source has highest close rate?"
            ]
        ),
        SchemaColumn(
            name="time_to_close",
            display_name="Time to Close",
            description="Average days from lead creation to sale",
            data_type="integer",
            metric_type=MetricType.SALES,
            visibility=ColumnVisibility.PUBLIC,
            aliases=["days to close", "sales cycle", "lead time"],
            units="days",
            aggregations=["avg", "min", "max"],
            primary_groupings=["lead_source", "salesperson", "vehicle_type"],
            sample_queries=[
                "What is our average time to close?",
                "Time to close by lead source",
                "Which salespeople close fastest?"
            ]
        ),
        SchemaColumn(
            name="be_penetration",
            display_name="Back End Penetration",
            description="Percentage of deals that include F&I products",
            data_type="decimal",
            metric_type=MetricType.SALES,
            visibility=ColumnVisibility.PUBLIC,
            aliases=["F&I penetration", "product penetration", "finance penetration"],
            format="percentage",
            units="%",
            aggregations=["avg"],
            primary_groupings=["finance_manager", "product_type", "vehicle_type"],
            sample_queries=[
                "What is our F&I penetration rate?",
                "Product penetration by vehicle type",
                "Back end penetration by finance manager"
            ]
        )
    ]
    
    return ExecSchemaProfile(
        role=ExecRole.GENERAL_SALES_MANAGER,
        name="General Sales Manager Schema Profile",
        description="Sales-focused metrics for dealership GSM with emphasis on close rates and unit metrics",
        columns=columns,
        default_metrics=["units_sold", "frontend_gross", "close_rate", "time_to_close"],
        default_dimensions=["vehicle_type", "salesperson", "lead_source"]
    )

def create_business_rule_registry() -> Dict[str, Dict[str, Any]]:
    """Create a default business rule registry."""
    return {
        "gross_not_negative": {
            "type": "comparison",
            "column": "total_gross_profit",
            "operator": ">=",
            "threshold": 0,
            "message": "Gross profit should not be negative",
            "severity": "high",
            "applies_to": ["total_gross_profit", "frontend_gross", "backend_gross"]
        },
        "min_gross_per_unit": {
            "type": "comparison",
            "column": "gross_profit_per_unit",
            "operator": ">=",
            "threshold": 500,
            "message": "Gross profit per unit should be at least $500",
            "severity": "medium"
        },
        "min_close_rate": {
            "type": "comparison",
            "column": "close_rate",
            "operator": ">=",
            "threshold": 8,
            "message": "Close rate should be at least 8%",
            "severity": "medium"
        },
        "max_days_in_inventory": {
            "type": "comparison",
            "column": "days_in_inventory",
            "operator": "<=",
            "threshold": 90,
            "message": "Days in inventory should not exceed 90 days",
            "severity": "medium"
        },
        "min_be_penetration": {
            "type": "comparison",
            "column": "be_penetration",
            "operator": ">=",
            "threshold": 70,
            "message": "Back end penetration should be at least 70%",
            "severity": "low"
        },
        "revenue_matching": {
            "type": "custom",
            "condition": "revenue >= (units_sold * avg_selling_price)",
            "message": "Revenue should match or exceed calculated value from units sold",
            "severity": "high"
        },
        # Price validation rule
        "price_below_cost": {
            "type": "comparison",
            "column": "price",
            "operator": ">",
            "threshold": {"column": "cost"},
            "message": "Selling price should be higher than cost",
            "severity": "high"
        },
        # Date range validation
        "sale_date_range": {
            "type": "range",
            "column": "sale_date",
            "min_value": "2020-01-01",
            "max_value": "TODAY",
            "message": "Sale date must be between 2020 and today",
            "severity": "high"
        },
        # String validation
        "valid_vin": {
            "type": "regexp",
            "column": "vin",
            "pattern": "^[A-HJ-NPR-Z0-9]{17}$",
            "message": "VIN must be 17 characters and contain only valid characters",
            "severity": "medium"
        }
    }

def save_business_rule_registry(rules: Dict[str, Dict[str, Any]], file_path: str) -> None:
    """Save the business rule registry to a YAML file."""
    try:
        with open(file_path, 'w') as f:
            yaml.dump(rules, f, default_flow_style=False)
        logger.info(f"Saved business rule registry to {file_path}")
    except Exception as e:
        logger.error(f"Failed to save business rule registry: {str(e)}")

def print_sample_implementation():
    """Print a sample implementation of the schema-aware query processing system."""
    # Create schema profiles
    gm_profile = create_gm_schema_profile()
    gsm_profile = create_gsm_schema_profile()
    
    # Create and save business rule registry
    rules = create_business_rule_registry()
    rule_registry_file = "BusinessRuleRegistry.yaml"
    save_business_rule_registry(rules, rule_registry_file)
    
    # Create rule engine
    rule_engine = BusinessRuleEngine()
    rule_engine.rules = rules
    
    # Create query rewriter
    rewriter = QueryRewriter(gm_profile, rule_engine)
    
    # Create query precision scorer
    scorer = QueryPrecisionScorer(gm_profile)
    
    # Process sample queries
    sample_queries = [
        "What is our total gross profit?",
        "Show me top salespeople by units",
        "Which vehicle type has highest front end gross?",
        "What's our PVR for new vehicles?",
        "Average days in inventory by model",
        "What's our net profit for last month?"
    ]
    
    print("\n===== SAMPLE IMPLEMENTATION =====\n")
    
    for query in sample_queries:
        print(f"\n>>> Processing query: '{query}'")
        rewritten, metadata = rewriter.rewrite_query(query)
        score_info = scorer.score_query(rewritten, metadata)
        
        print(f"Rewritten as: '{rewritten}'")
        print(f"Confidence: {score_info['confidence_level']} ({score_info['score']:.2f})")
        
        # Show column matches
        print("\nColumn matches:")
        for term, matches in metadata["column_matches"].items():
            if matches:
                col, conf = matches[0]["column"], matches[0]["confidence"]
                print(f"  • '{term}' → '{col}' ({conf:.2f})")
        
        # Show missing/ambiguous terms
        if metadata["missing_terms"]:
            print("\nUnrecognized terms:", ", ".join(metadata["missing_terms"]))
        
        if metadata.get("ambiguities"):
            print("\nAmbiguous terms:")
            for amb in metadata["ambiguities"]:
                print(f"  • '{amb['term']}' could be: " + 
                      ", ".join(f"{c['column']} ({c['confidence']:.2f})" for c in amb["candidates"]))
        
        print("\n-----------------------------------")

if __name__ == "__main__":
    print_sample_implementation()