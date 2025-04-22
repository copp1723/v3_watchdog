"""
Semantic Layer for Watchdog AI.

This module provides the semantic layer that ensures queries are semantically correct,
schema-safe, and business rule compliant. It combines executive schema profiles,
business rules, query rewriting, and precision scoring.
"""

import os
import json
import yaml
import logging
import pickle
import hashlib
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass, field
from datetime import datetime
import pandas as pd
import numpy as np
from pathlib import Path

# Import existing components
from .exec_schema_profiles import (
    ExecSchemaProfile,
    ExecRole,
    SchemaColumn,
    ColumnVisibility,
    MetricType,
    BusinessRuleEngine,
    QueryRewriter,
    QueryPrecisionScorer,
    FeedbackLogger,
    create_gm_schema_profile,
    create_gsm_schema_profile
)

# Import advanced rule engine
from .rule_engine import (
    BusinessRuleEngine as AdvancedRuleEngine,
    RuleValidationResult
)

logger = logging.getLogger(__name__)

# Default paths
DEFAULT_PROFILES_DIR = "profiles"
DEFAULT_RULES_FILE = "BusinessRuleRegistry.yaml"
DEFAULT_FEEDBACK_FILE = "query_feedback.json"
DEFAULT_USER_PROFILES_DIR = "user_profiles"
DEFAULT_MODELS_DIR = "models"

@dataclass
class SchemaAdjustment:
    """Represents a user-specific schema adjustment."""
    column_name: str
    adjustment_type: str  # "alias", "visibility", "relevance", etc.
    value: Any
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    source: str = "user_feedback"  # or "system_learning"
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "column_name": self.column_name,
            "adjustment_type": self.adjustment_type,
            "value": self.value,
            "timestamp": self.timestamp,
            "source": self.source,
            "confidence": self.confidence
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SchemaAdjustment':
        """Create from dictionary representation."""
        return cls(
            column_name=data["column_name"],
            adjustment_type=data["adjustment_type"],
            value=data["value"],
            timestamp=data.get("timestamp", datetime.now().isoformat()),
            source=data.get("source", "user_feedback"),
            confidence=data.get("confidence", 1.0)
        )

@dataclass
class UserSchemaProfile:
    """Represents a user-specific schema profile with adjustments."""
    user_id: str
    base_profile_id: str  # ID of the base profile (e.g., "general_manager")
    role: ExecRole
    adjustments: List[SchemaAdjustment] = field(default_factory=list)
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "user_id": self.user_id,
            "base_profile_id": self.base_profile_id,
            "role": self.role.value,
            "adjustments": [adj.to_dict() for adj in self.adjustments],
            "last_updated": self.last_updated
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UserSchemaProfile':
        """Create from dictionary representation."""
        return cls(
            user_id=data["user_id"],
            base_profile_id=data["base_profile_id"],
            role=ExecRole(data["role"]) if isinstance(data["role"], str) else data["role"],
            adjustments=[
                SchemaAdjustment.from_dict(adj) if isinstance(adj, dict) else adj
                for adj in data.get("adjustments", [])
            ],
            last_updated=data.get("last_updated", datetime.now().isoformat())
        )

class QueryStatistics:
    """Tracks and stores statistics about queries and rewrites."""
    
    def __init__(self, stats_file: Optional[str] = None):
        """Initialize query statistics tracker."""
        self.stats_file = stats_file
        self.stats = {
            "total_queries": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "queries_by_confidence": {
                "high": 0,
                "medium": 0,
                "low": 0
            },
            "rewrite_stats": {
                "total_rewrites": 0,
                "successful_rewrites": 0,
                "failed_rewrites": 0
            },
            "term_matches": {},
            "rewrite_patterns": {},
            "ambiguity_patterns": {},
            "common_issues": {}
        }
        
        if stats_file and os.path.exists(stats_file):
            self._load_stats()
    
    def _load_stats(self) -> None:
        """Load statistics from file."""
        try:
            with open(self.stats_file, 'r') as f:
                self.stats = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load query statistics: {e}")
    
    def _save_stats(self) -> None:
        """Save statistics to file."""
        if not self.stats_file:
            return
            
        try:
            os.makedirs(os.path.dirname(self.stats_file), exist_ok=True)
            with open(self.stats_file, 'w') as f:
                json.dump(self.stats, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save query statistics: {e}")
    
    def record_query(self, query: str, rewritten_query: str, metadata: Dict[str, Any], 
                    confidence: str, success: bool) -> None:
        """Record statistics for a query."""
        self.stats["total_queries"] += 1
        
        if success:
            self.stats["successful_queries"] += 1
        else:
            self.stats["failed_queries"] += 1
        
        # Record confidence level
        self.stats["queries_by_confidence"][confidence] = \
            self.stats["queries_by_confidence"].get(confidence, 0) + 1
        
        # Record rewrite stats
        if query != rewritten_query:
            self.stats["rewrite_stats"]["total_rewrites"] += 1
            if success:
                self.stats["rewrite_stats"]["successful_rewrites"] += 1
            else:
                self.stats["rewrite_stats"]["failed_rewrites"] += 1
        
        # Record term matches
        for term, matches in metadata.get("column_matches", {}).items():
            if not matches:
                continue
                
            if term not in self.stats["term_matches"]:
                self.stats["term_matches"][term] = {"count": 0, "columns": {}}
            
            self.stats["term_matches"][term]["count"] += 1
            
            for match in matches:
                column = match["column"]
                confidence = match["confidence"]
                
                if column not in self.stats["term_matches"][term]["columns"]:
                    self.stats["term_matches"][term]["columns"][column] = {"count": 0, "confidence_sum": 0}
                
                self.stats["term_matches"][term]["columns"][column]["count"] += 1
                self.stats["term_matches"][term]["columns"][column]["confidence_sum"] += confidence
        
        # Record rewrite patterns if any
        for rewrite in metadata.get("rewrites", []):
            original = rewrite.get("original", "")
            rewritten = rewrite.get("rewritten", "")
            
            if not original or not rewritten:
                continue
                
            pattern = f"{original} → {rewritten}"
            self.stats["rewrite_patterns"][pattern] = self.stats["rewrite_patterns"].get(pattern, 0) + 1
        
        # Record ambiguity patterns
        for ambiguity in metadata.get("ambiguities", []):
            term = ambiguity.get("term", "")
            
            if not term:
                continue
                
            if term not in self.stats["ambiguity_patterns"]:
                self.stats["ambiguity_patterns"][term] = {"count": 0, "candidates": {}}
            
            self.stats["ambiguity_patterns"][term]["count"] += 1
            
            for candidate in ambiguity.get("candidates", []):
                column = candidate.get("column", "")
                confidence = candidate.get("confidence", 0)
                
                if not column:
                    continue
                    
                if column not in self.stats["ambiguity_patterns"][term]["candidates"]:
                    self.stats["ambiguity_patterns"][term]["candidates"][column] = {"count": 0, "confidence_sum": 0}
                
                self.stats["ambiguity_patterns"][term]["candidates"][column]["count"] += 1
                self.stats["ambiguity_patterns"][term]["candidates"][column]["confidence_sum"] += confidence
        
        # Record common issues
        for issue in metadata.get("missing_terms", []):
            self.stats["common_issues"][issue] = self.stats["common_issues"].get(issue, 0) + 1
        
        # Save after each update
        self._save_stats()
    
    def get_term_suggestions(self, term: str, min_confidence: float = 0.7) -> List[Tuple[str, float]]:
        """Get column suggestions for a term based on historical matches."""
        if term not in self.stats["term_matches"]:
            return []
            
        columns = self.stats["term_matches"][term]["columns"]
        suggestions = []
        
        for column, data in columns.items():
            count = data["count"]
            avg_confidence = data["confidence_sum"] / count
            
            if avg_confidence >= min_confidence:
                suggestions.append((column, avg_confidence))
        
        # Sort by confidence
        suggestions.sort(key=lambda x: x[1], reverse=True)
        
        return suggestions
    
    def get_ambiguity_resolution(self, term: str) -> Optional[str]:
        """Get the most likely resolution for an ambiguous term."""
        if term not in self.stats["ambiguity_patterns"]:
            return None
            
        candidates = self.stats["ambiguity_patterns"][term]["candidates"]
        if not candidates:
            return None
            
        # Find the candidate with the highest count
        best_candidate = max(candidates.items(), key=lambda x: x[1]["count"])
        return best_candidate[0]
    
    def get_common_issues(self, limit: int = 10) -> List[Tuple[str, int]]:
        """Get the most common issues."""
        issues = sorted(self.stats["common_issues"].items(), key=lambda x: x[1], reverse=True)
        return issues[:limit]
    
    def get_rewrite_confidence(self) -> float:
        """Calculate the confidence in the rewrite system based on historical success."""
        total = self.stats["rewrite_stats"]["total_rewrites"]
        if total == 0:
            return 0.5  # Default neutral confidence
            
        successful = self.stats["rewrite_stats"]["successful_rewrites"]
        return successful / total

class PrecisionScoringEngine:
    """Engine for scoring query precision using trained models."""
    
    def __init__(self, model_path: Optional[str] = None):
        """Initialize the precision scoring engine."""
        self.model = None
        self.vectorizer = None
        self.features = {
            "term_match_ratio": 0.3,
            "avg_confidence": 0.2,
            "has_ambiguities": -0.1,
            "missing_terms_ratio": -0.2,
            "rewrite_ratio": 0.1,
            "historical_success": 0.2
        }
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def load_model(self, model_path: str) -> bool:
        """Load a trained model from the specified path."""
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
                
            self.model = model_data.get("model")
            self.vectorizer = model_data.get("vectorizer")
            self.features = model_data.get("features", self.features)
            
            logger.info(f"Loaded precision scoring model from {model_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load precision scoring model: {e}")
            return False
    
    def save_model(self, model_path: str) -> bool:
        """Save the current model to the specified path."""
        try:
            model_data = {
                "model": self.model,
                "vectorizer": self.vectorizer,
                "features": self.features
            }
            
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            with open(model_path, 'wb') as f:
                pickle.dump(model_data, f)
                
            logger.info(f"Saved precision scoring model to {model_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save precision scoring model: {e}")
            return False
    
    def extract_features(self, query: str, metadata: Dict[str, Any], stats: QueryStatistics) -> Dict[str, float]:
        """Extract features for precision prediction."""
        features = {}
        
        # 1. Term match ratio
        total_terms = len(metadata.get("column_matches", {})) + len(metadata.get("missing_terms", []))
        matched_terms = len(metadata.get("column_matches", {}))
        
        if total_terms > 0:
            features["term_match_ratio"] = matched_terms / total_terms
        else:
            features["term_match_ratio"] = 0.0
        
        # 2. Average confidence
        confidences = []
        for matches in metadata.get("column_matches", {}).values():
            if matches:
                confidences.append(matches[0]["confidence"])
        
        features["avg_confidence"] = sum(confidences) / len(confidences) if confidences else 0.0
        
        # 3. Has ambiguities
        features["has_ambiguities"] = 1.0 if metadata.get("ambiguities") else 0.0
        
        # 4. Missing terms ratio
        if total_terms > 0:
            features["missing_terms_ratio"] = len(metadata.get("missing_terms", [])) / total_terms
        else:
            features["missing_terms_ratio"] = 0.0
        
        # 5. Rewrite ratio (how much was rewritten)
        original_query = query
        rewritten_query = metadata.get("rewritten_query", query)
        
        if not original_query:
            features["rewrite_ratio"] = 0.0
        else:
            # Simple diff score based on length differences
            features["rewrite_ratio"] = 1.0 - (len(set(original_query) - set(rewritten_query)) / len(original_query))
        
        # 6. Historical success rate
        features["historical_success"] = stats.get_rewrite_confidence()
        
        return features
    
    def predict_precision(self, query: str, metadata: Dict[str, Any], 
                        stats: QueryStatistics) -> Tuple[float, str, Dict[str, Any]]:
        """
        Predict the precision score for a query.
        Returns (score, confidence_level, details).
        """
        # Extract features
        features = self.extract_features(query, metadata, stats)
        
        # Use ML model if available
        if self.model and self.vectorizer:
            # Prepare feature vector
            feature_vector = self.vectorizer.transform([features])
            
            # Make prediction
            score = self.model.predict(feature_vector)[0]
            
        else:
            # Use heuristic model based on feature weights
            score = 0.5  # Base score
            
            for feature, value in features.items():
                if feature in self.features:
                    score += value * self.features[feature]
            
            # Ensure score is between 0 and 1
            score = max(0.0, min(1.0, score))
        
        # Determine confidence level
        if score >= 0.7:
            confidence_level = "high"
        elif score >= 0.4:
            confidence_level = "medium"
        else:
            confidence_level = "low"
        
        # Prepare details
        details = {
            "score": score,
            "confidence_level": confidence_level,
            "features": features,
            "feature_weights": self.features.copy()  # Send a copy to avoid modification
        }
        
        return score, confidence_level, details

class DynamicSchemaProfileManager:
    """Manages dynamic loading and updating of executive schema profiles."""
    
    def __init__(self, profiles_dir: str = DEFAULT_PROFILES_DIR, 
                user_profiles_dir: str = DEFAULT_USER_PROFILES_DIR):
        """Initialize the schema profile manager."""
        self.profiles_dir = profiles_dir
        self.user_profiles_dir = user_profiles_dir
        
        # Ensure directories exist
        os.makedirs(profiles_dir, exist_ok=True)
        os.makedirs(user_profiles_dir, exist_ok=True)
        
        # Load base profiles
        self.base_profiles = {}
        self._load_base_profiles()
        
        # User profiles cache
        self.user_profiles = {}
    
    def _load_base_profiles(self) -> None:
        """Load base profiles from profiles directory."""
        # Add built-in profiles
        self.base_profiles = {
            "general_manager": create_gm_schema_profile(),
            "general_sales_manager": create_gsm_schema_profile()
        }
        
        # Load profiles from files
        for filename in os.listdir(self.profiles_dir):
            if filename.endswith(".json"):
                try:
                    file_path = os.path.join(self.profiles_dir, filename)
                    with open(file_path, 'r') as f:
                        profile_data = json.load(f)
                    
                    profile = ExecSchemaProfile.from_dict(profile_data)
                    self.base_profiles[profile.role.value] = profile
                    
                    logger.info(f"Loaded base profile: {profile.role.value}")
                except Exception as e:
                    logger.error(f"Failed to load profile {filename}: {e}")
    
    def _load_user_profile(self, user_id: str) -> Optional[UserSchemaProfile]:
        """Load a user profile from file."""
        file_path = os.path.join(self.user_profiles_dir, f"{user_id}.json")
        
        if not os.path.exists(file_path):
            return None
            
        try:
            with open(file_path, 'r') as f:
                profile_data = json.load(f)
                
            user_profile = UserSchemaProfile.from_dict(profile_data)
            return user_profile
        except Exception as e:
            logger.error(f"Failed to load user profile for {user_id}: {e}")
            return None
    
    def _save_user_profile(self, user_profile: UserSchemaProfile) -> bool:
        """Save a user profile to file."""
        file_path = os.path.join(self.user_profiles_dir, f"{user_profile.user_id}.json")
        
        try:
            with open(file_path, 'w') as f:
                json.dump(user_profile.to_dict(), f, indent=2)
                
            logger.info(f"Saved user profile for {user_profile.user_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to save user profile for {user_profile.user_id}: {e}")
            return False
    
    def get_profile(self, user_id: str, role: Optional[ExecRole] = None) -> ExecSchemaProfile:
        """
        Get a schema profile for a user, apply any user-specific adjustments.
        If no user profile exists, uses the base profile for the specified role.
        """
        # Check cache first
        if user_id in self.user_profiles:
            user_profile = self.user_profiles[user_id]
        else:
            # Load user profile
            user_profile = self._load_user_profile(user_id)
            
            if user_profile:
                # Cache for future use
                self.user_profiles[user_id] = user_profile
        
        # If no user profile but role is specified, use base profile
        if not user_profile and role:
            if role.value in self.base_profiles:
                return self.base_profiles[role.value]
            else:
                # Default to GM if role not found
                logger.warning(f"Role {role.value} not found, using GM profile")
                return self.base_profiles["general_manager"]
        
        # If no user profile and no role specified, use GM profile
        if not user_profile:
            return self.base_profiles["general_manager"]
        
        # Get base profile
        if user_profile.base_profile_id in self.base_profiles:
            base_profile = self.base_profiles[user_profile.base_profile_id]
        else:
            # Default to the role in user profile
            role_value = user_profile.role.value
            if role_value in self.base_profiles:
                base_profile = self.base_profiles[role_value]
            else:
                # Fall back to GM profile
                base_profile = self.base_profiles["general_manager"]
        
        # Apply user adjustments to create a customized profile
        return self._apply_adjustments(base_profile, user_profile.adjustments)
    
    def _apply_adjustments(self, base_profile: ExecSchemaProfile, 
                          adjustments: List[SchemaAdjustment]) -> ExecSchemaProfile:
        """Apply user adjustments to a base profile."""
        # Create a copy of the base profile
        profile_data = base_profile.dict()
        
        # Apply adjustments
        for adj in adjustments:
            # Find the column to adjust
            column_found = False
            
            for i, col_data in enumerate(profile_data["columns"]):
                if col_data["name"] == adj.column_name:
                    column_found = True
                    
                    # Apply the adjustment based on type
                    if adj.adjustment_type == "alias":
                        # Add a new alias if it doesn't exist
                        if adj.value not in col_data["aliases"]:
                            col_data["aliases"].append(adj.value)
                    
                    elif adj.adjustment_type == "visibility":
                        # Update visibility
                        col_data["visibility"] = adj.value
                    
                    elif adj.adjustment_type == "relevance":
                        # Adjust column priority or sorting (implementation-specific)
                        pass
                    
                    # Update the column
                    profile_data["columns"][i] = col_data
                    break
            
            # If column not found and it's an alias adjustment, consider adding it to
            # an existing column by searching aliases
            if not column_found and adj.adjustment_type == "alias":
                for i, col_data in enumerate(profile_data["columns"]):
                    # Check if any existing alias partially matches
                    for alias in col_data["aliases"]:
                        if (adj.value.lower() in alias.lower() or 
                            alias.lower() in adj.value.lower()):
                            # Add the alias
                            if adj.value not in col_data["aliases"]:
                                col_data["aliases"].append(adj.value)
                                profile_data["columns"][i] = col_data
                                column_found = True
                                break
                    
                    if column_found:
                        break
        
        # Create a new profile from the adjusted data
        return ExecSchemaProfile.from_dict(profile_data)
    
    def update_profile(self, user_id: str, feedback: Dict[str, Any]) -> bool:
        """
        Update a user's schema profile based on feedback.
        Creates a new profile if none exists.
        """
        # Get user profile or create new one
        user_profile = self._load_user_profile(user_id)
        
        if not user_profile:
            # Determine role from feedback or default to GM
            role_value = feedback.get("role", "general_manager")
            role = ExecRole(role_value) if isinstance(role_value, str) else role_value
            
            user_profile = UserSchemaProfile(
                user_id=user_id,
                base_profile_id=role_value if isinstance(role_value, str) else role_value.value,
                role=role
            )
        
        # Process different types of feedback
        adjustments = []
        
        # 1. New aliases
        if "term_mappings" in feedback:
            for term, column in feedback["term_mappings"].items():
                adjustments.append(SchemaAdjustment(
                    column_name=column,
                    adjustment_type="alias",
                    value=term,
                    source=feedback.get("source", "user_feedback"),
                    confidence=feedback.get("confidence", 1.0)
                ))
        
        # 2. Visibility changes
        if "visibility_changes" in feedback:
            for column, visibility in feedback["visibility_changes"].items():
                adjustments.append(SchemaAdjustment(
                    column_name=column,
                    adjustment_type="visibility",
                    value=visibility,
                    source=feedback.get("source", "user_feedback"),
                    confidence=feedback.get("confidence", 1.0)
                ))
        
        # 3. Other adjustments
        if "adjustments" in feedback:
            for adj_data in feedback["adjustments"]:
                if isinstance(adj_data, dict):
                    adjustments.append(SchemaAdjustment.from_dict(adj_data))
                elif isinstance(adj_data, SchemaAdjustment):
                    adjustments.append(adj_data)
        
        # Add adjustments to user profile
        user_profile.adjustments.extend(adjustments)
        user_profile.last_updated = datetime.now().isoformat()
        
        # Cache the updated profile
        self.user_profiles[user_id] = user_profile
        
        # Save to file
        return self._save_user_profile(user_profile)

class EnhancedQueryRewriter:
    """Enhanced query rewriter backed by NLP models for better rephrasing."""
    
    def __init__(self, schema_profile: ExecSchemaProfile, 
                rule_engine: Optional[BusinessRuleEngine] = None,
                stats: Optional[QueryStatistics] = None,
                nlp_model_path: Optional[str] = None):
        """Initialize the enhanced query rewriter."""
        self.schema_profile = schema_profile
        self.rule_engine = rule_engine
        self.stats = stats
        
        # Create base rewriter
        self.base_rewriter = QueryRewriter(schema_profile, rule_engine)
        
        # NLP model for rewrites
        self.nlp_model = None
        if nlp_model_path and os.path.exists(nlp_model_path):
            self._load_nlp_model(nlp_model_path)
    
    def _load_nlp_model(self, model_path: str) -> bool:
        """Load NLP model for query rewriting."""
        try:
            # This is a placeholder for loading an actual NLP model
            # Could be spaCy, Hugging Face transformers, or custom model
            logger.info(f"Loading NLP model from {model_path}")
            
            # Placeholder for model loading code
            # self.nlp_model = spacy.load(model_path)
            # or
            # self.nlp_model = transformers.AutoModelForSequenceClassification.from_pretrained(model_path)
            
            return True
        except Exception as e:
            logger.error(f"Failed to load NLP model: {e}")
            return False
    
    def rewrite_query(self, query: str) -> Tuple[str, Dict[str, Any]]:
        """
        Rewrite a query to be schema-compatible with NLP enhancements.
        Returns the rewritten query and metadata about the rewrite.
        """
        # Start with base rewriter
        rewritten_query, metadata = self.base_rewriter.rewrite_query(query)
        
        # Apply NLP-based rewrites for ambiguities
        if self.nlp_model and metadata.get("ambiguities"):
            rewritten_query, nlp_metadata = self._apply_nlp_rewrites(rewritten_query, metadata)
            
            # Update metadata with NLP results
            metadata["nlp_rewrites"] = nlp_metadata
        
        # Use historical statistics for low-confidence matches
        if self.stats:
            rewritten_query, stats_metadata = self._apply_statistical_rewrites(rewritten_query, metadata)
            
            # Update metadata
            metadata["statistical_rewrites"] = stats_metadata
        
        return rewritten_query, metadata
    
    def _apply_nlp_rewrites(self, query: str, metadata: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Apply NLP-based rewrites to resolve ambiguities."""
        # This is a placeholder for actual NLP-based rewriting
        nlp_metadata = {"rewrites": []}
        
        # Process ambiguities
        for ambiguity in metadata.get("ambiguities", []):
            term = ambiguity.get("term")
            candidates = ambiguity.get("candidates", [])
            
            if not term or not candidates:
                continue
            
            # In a real implementation, we would use the NLP model to select the best candidate
            # based on the full query context
            
            # For now, just use the highest confidence candidate
            best_candidate = max(candidates, key=lambda c: c.get("confidence", 0))
            
            # Only rewrite if confidence is reasonable
            if best_candidate.get("confidence", 0) >= 0.6:
                old_term = term
                new_term = best_candidate.get("column")
                
                # Replace in the query
                query = self._replace_term(query, old_term, new_term)
                
                # Record the rewrite
                nlp_metadata["rewrites"].append({
                    "type": "ambiguity_resolution",
                    "original": old_term,
                    "rewritten": new_term,
                    "confidence": best_candidate.get("confidence", 0)
                })
        
        return query, nlp_metadata
    
    def _apply_statistical_rewrites(self, query: str, metadata: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Apply statistical rewrites based on historical data."""
        stats_metadata = {"rewrites": []}
        
        # Process missing terms
        for term in metadata.get("missing_terms", []):
            # Get suggestions from statistics
            suggestions = self.stats.get_term_suggestions(term) if self.stats else []
            
            if suggestions:
                # Use the best suggestion
                new_term = suggestions[0][0]
                confidence = suggestions[0][1]
                
                # Only rewrite if confidence is high
                if confidence >= 0.7:
                    # Replace in the query
                    query = self._replace_term(query, term, new_term)
                    
                    # Record the rewrite
                    stats_metadata["rewrites"].append({
                        "type": "statistical_mapping",
                        "original": term,
                        "rewritten": new_term,
                        "confidence": confidence
                    })
        
        # Process low-confidence matches
        for term, matches in metadata.get("column_matches", {}).items():
            if not matches:
                continue
                
            # Check if the match has low confidence
            if matches[0]["confidence"] < 0.7:
                # Get suggestions from statistics
                suggestions = self.stats.get_term_suggestions(term) if self.stats else []
                
                if suggestions and suggestions[0][1] > matches[0]["confidence"]:
                    # Use the statistical suggestion instead
                    new_term = suggestions[0][0]
                    confidence = suggestions[0][1]
                    
                    # Replace in the query
                    query = self._replace_term(query, term, new_term)
                    
                    # Record the rewrite
                    stats_metadata["rewrites"].append({
                        "type": "confidence_improvement",
                        "original": term,
                        "original_match": matches[0]["column"],
                        "original_confidence": matches[0]["confidence"],
                        "rewritten": new_term,
                        "confidence": confidence
                    })
        
        return query, stats_metadata
    
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

class SemanticQueryProcessor:
    """
    Central processor for semantically correct, schema-safe, and rule-compliant queries.
    Combines all components of the semantic layer.
    """
    
    def __init__(self, 
                profiles_dir: str = DEFAULT_PROFILES_DIR,
                user_profiles_dir: str = DEFAULT_USER_PROFILES_DIR,
                rules_file: str = DEFAULT_RULES_FILE,
                feedback_file: str = DEFAULT_FEEDBACK_FILE,
                stats_file: str = "query_stats.json",
                models_dir: str = DEFAULT_MODELS_DIR):
        """Initialize the semantic query processor."""
        self.profiles_dir = profiles_dir
        self.user_profiles_dir = user_profiles_dir
        self.rules_file = rules_file
        self.feedback_file = feedback_file
        self.stats_file = stats_file
        self.models_dir = models_dir
        
        # Initialize components
        self.profile_manager = DynamicSchemaProfileManager(profiles_dir, user_profiles_dir)
        self.rule_engine = AdvancedRuleEngine(rules_file) if os.path.exists(rules_file) else AdvancedRuleEngine()
        self.feedback_logger = FeedbackLogger(feedback_file)
        self.stats = QueryStatistics(stats_file)
        
        # Initialize precision scoring engine
        precision_model_path = os.path.join(models_dir, "precision_model.pkl")
        self.precision_engine = PrecisionScoringEngine(
            precision_model_path if os.path.exists(precision_model_path) else None
        )
        
        # Cached components per user
        self.user_rewriters = {}
        self.user_scorers = {}
    
    def _get_user_components(self, user_id: str, role: Optional[ExecRole] = None) -> Tuple[
            ExecSchemaProfile,
            EnhancedQueryRewriter,
            QueryPrecisionScorer
        ]:
        """Get or create user-specific components."""
        # Get user profile
        schema_profile = self.profile_manager.get_profile(user_id, role)
        
        # Check cache for rewriter
        rewriter_key = self._make_cache_key(user_id, schema_profile)
        if rewriter_key in self.user_rewriters:
            rewriter = self.user_rewriters[rewriter_key]
        else:
            # Create rewriter
            rewriter = EnhancedQueryRewriter(schema_profile, self.rule_engine, self.stats)
            self.user_rewriters[rewriter_key] = rewriter
        
        # Check cache for scorer
        if user_id in self.user_scorers:
            scorer = self.user_scorers[user_id]
        else:
            # Create scorer
            scorer = QueryPrecisionScorer(schema_profile)
            self.user_scorers[user_id] = scorer
        
        return schema_profile, rewriter, scorer
    
    def _make_cache_key(self, user_id: str, profile: ExecSchemaProfile) -> str:
        """Create a cache key for a user and profile."""
        # Use hash of profile to detect changes
        profile_hash = hashlib.md5(profile.json().encode()).hexdigest()
        return f"{user_id}:{profile_hash}"
    
    def process_query(self, query: str, user_id: str, 
                     role: Optional[ExecRole] = None) -> Dict[str, Any]:
        """
        Process a query to ensure it is semantically correct, schema-safe, and rule-compliant.
        Returns a detailed result with the processed query and metadata.
        """
        # Get user components
        schema_profile, rewriter, scorer = self._get_user_components(user_id, role)
        
        # 1. Rewrite query
        rewritten_query, rewrite_metadata = rewriter.rewrite_query(query)
        
        # 2. Score query precision
        precision_score, confidence_level, precision_details = self.precision_engine.predict_precision(
            query, rewrite_metadata, self.stats
        )
        
        # 3. Apply business rules
        rule_results = {}
        if self.rule_engine:
            # Extract data for rule validation
            data_to_validate = self._extract_data_for_validation(rewrite_metadata)
            
            # Get applicable rules for the user's role
            rule_set = self.rule_engine.get_rules_for_role(
                schema_profile.role.value if isinstance(schema_profile.role, ExecRole) 
                else schema_profile.role
            )
            
            # Evaluate rules
            for rule_id in rule_set:
                result = self.rule_engine.evaluate_rule(rule_id, data_to_validate)
                rule_results[rule_id] = result.to_dict()
        
        # Determine if query is valid for execution
        is_valid = precision_score >= 0.4  # Medium or high confidence
        
        # Check business rule violations
        if is_valid:
            critical_violations = [
                rule_id for rule_id, result in rule_results.items()
                if not result.get("is_valid", True) and result.get("severity", "medium") == "high"
            ]
            
            if critical_violations:
                is_valid = False
        
        # Prepare result
        result = {
            "original_query": query,
            "rewritten_query": rewritten_query,
            "is_valid": is_valid,
            "precision_score": precision_score,
            "confidence_level": confidence_level,
            "schema_profile": {
                "role": schema_profile.role.value if isinstance(schema_profile.role, ExecRole) 
                       else schema_profile.role,
                "name": schema_profile.name
            },
            "rewrite_metadata": rewrite_metadata,
            "precision_details": precision_details,
            "rule_results": rule_results,
            "timestamp": datetime.now().isoformat()
        }
        
        # Record query statistics
        self.stats.record_query(
            query=query,
            rewritten_query=rewritten_query,
            metadata=rewrite_metadata,
            confidence=confidence_level,
            success=is_valid
        )
        
        return result
    
    def _extract_data_for_validation(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Extract data from query metadata for rule validation."""
        data = {}
        
        # Extract column matches
        for term, matches in metadata.get("column_matches", {}).items():
            if not matches:
                continue
                
            column = matches[0]["column"]
            # For now, use a placeholder value
            data[column] = term
        
        # In a real implementation, this would extract actual values from the query
        # or from the context
        
        return data
    
    def process_feedback(self, feedback_data: Dict[str, Any]) -> bool:
        """Process user feedback and update profiles accordingly."""
        user_id = feedback_data.get("user_id")
        if not user_id:
            logger.error("Feedback missing user_id")
            return False
        
        # Log feedback
        self.feedback_logger.log_feedback(feedback_data)
        
        # Extract profile update information
        profile_updates = {}
        
        # Extract term mappings
        if "query" in feedback_data and "schema_matches" in feedback_data:
            term_mappings = {}
            
            for term, matches in feedback_data["schema_matches"].items():
                if matches:
                    term_mappings[term] = matches[0]["column"]
            
            profile_updates["term_mappings"] = term_mappings
        
        # Handle explicit corrections
        if feedback_data.get("feedback_type") == "correction" and feedback_data.get("correction_details"):
            corrections = feedback_data["correction_details"]
            
            if "term_corrections" in corrections:
                term_mappings = profile_updates.get("term_mappings", {})
                term_mappings.update(corrections["term_corrections"])
                profile_updates["term_mappings"] = term_mappings
            
            if "visibility_corrections" in corrections:
                profile_updates["visibility_changes"] = corrections["visibility_corrections"]
        
        # Update profile if there are changes
        if profile_updates:
            return self.profile_manager.update_profile(user_id, profile_updates)
        
        return True
    
    def get_query_debug_info(self, query_result: Dict[str, Any]) -> Dict[str, Any]:
        """Get detailed debug information for a query result."""
        debug_info = {
            "query": {
                "original": query_result["original_query"],
                "rewritten": query_result["rewritten_query"],
                "confidence": query_result["confidence_level"],
                "score": query_result["precision_score"]
            },
            "schema": {
                "role": query_result["schema_profile"]["role"],
                "schema_name": query_result["schema_profile"]["name"]
            },
            "processing": {
                "term_matches": query_result["rewrite_metadata"].get("column_matches", {}),
                "missing_terms": query_result["rewrite_metadata"].get("missing_terms", []),
                "ambiguities": query_result["rewrite_metadata"].get("ambiguities", []),
                "rewrites": query_result["rewrite_metadata"].get("rewrites", []),
                "nlp_rewrites": query_result["rewrite_metadata"].get("nlp_rewrites", {}).get("rewrites", []),
                "statistical_rewrites": query_result["rewrite_metadata"].get("statistical_rewrites", {}).get("rewrites", [])
            },
            "rules": {
                "passed": [rule_id for rule_id, result in query_result["rule_results"].items() if result.get("is_valid", True)],
                "failed": [
                    {
                        "rule_id": rule_id,
                        "message": result.get("message", ""),
                        "severity": result.get("severity", "medium")
                    }
                    for rule_id, result in query_result["rule_results"].items() 
                    if not result.get("is_valid", True)
                ]
            }
        }
        
        return debug_info