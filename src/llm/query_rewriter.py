"""
Query Rewriter for Watchdog AI.

This module provides a sophisticated query rewriting system that
transforms user queries to be schema-compatible, addressing ambiguities
and low-confidence token matches.
"""

import os
import re
import json
import yaml
import logging
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from collections import Counter
from datetime import datetime

# Import schema and rule components
from src.utils.adaptive_schema import (
    SchemaProfile, SchemaColumn, extract_query_terms, ExecRole
)
from src.rule_engine import BusinessRuleEngine, RuleValidationResult

logger = logging.getLogger(__name__)

class QueryRewriteStats:
    """Tracks statistics about query rewrites."""
    
    def __init__(self, stats_file: Optional[str] = None):
        """Initialize rewrite statistics tracker."""
        self.stats_file = stats_file
        self.stats = {
            "total_queries": 0,
            "queries_rewritten": 0,
            "term_mappings": {},
            "ambiguity_resolutions": {},
            "successful_rewrites": 0,
            "failed_rewrites": 0
        }
        
        if stats_file and os.path.exists(stats_file):
            self._load_stats()
    
    def _load_stats(self) -> None:
        """Load statistics from file."""
        try:
            with open(self.stats_file, 'r') as f:
                self.stats = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load rewrite statistics: {e}")
    
    def _save_stats(self) -> None:
        """Save statistics to file."""
        if not self.stats_file:
            return
        
        try:
            os.makedirs(os.path.dirname(self.stats_file), exist_ok=True)
            with open(self.stats_file, 'w') as f:
                json.dump(self.stats, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save rewrite statistics: {e}")
    
    def record_rewrite(self, original_query: str, rewritten_query: str, 
                     metadata: Dict[str, Any], success: bool = True) -> None:
        """Record a query rewrite operation."""
        self.stats["total_queries"] += 1
        
        if original_query != rewritten_query:
            self.stats["queries_rewritten"] += 1
            
            if success:
                self.stats["successful_rewrites"] += 1
            else:
                self.stats["failed_rewrites"] += 1
        
        # Record term mappings
        for rewrite in metadata.get("rewrites", []):
            original = rewrite.get("original", "")
            rewritten = rewrite.get("rewritten", "")
            
            if not original or not rewritten:
                continue
                
            if original not in self.stats["term_mappings"]:
                self.stats["term_mappings"][original] = {}
            
            self.stats["term_mappings"][original][rewritten] = \
                self.stats["term_mappings"][original].get(rewritten, 0) + 1
        
        # Record ambiguity resolutions
        for ambiguity in metadata.get("ambiguities", []):
            term = ambiguity.get("term", "")
            resolved_to = None
            
            # Check if this ambiguity was resolved in rewrites
            for rewrite in metadata.get("rewrites", []):
                if rewrite.get("original", "") == term:
                    resolved_to = rewrite.get("rewritten", "")
                    break
            
            if not term or not resolved_to:
                continue
                
            if term not in self.stats["ambiguity_resolutions"]:
                self.stats["ambiguity_resolutions"][term] = {}
            
            self.stats["ambiguity_resolutions"][term][resolved_to] = \
                self.stats["ambiguity_resolutions"][term].get(resolved_to, 0) + 1
        
        # Save after each update
        self._save_stats()
    
    def get_best_rewrite(self, term: str) -> Optional[str]:
        """Get the statistically best rewrite for a term."""
        if term not in self.stats["term_mappings"]:
            return None
            
        term_stats = self.stats["term_mappings"][term]
        if not term_stats:
            return None
            
        # Find the most common rewrite
        best_rewrite = max(term_stats.items(), key=lambda x: x[1])
        return best_rewrite[0]
    
    def get_ambiguity_resolution(self, term: str) -> Optional[str]:
        """Get the most likely resolution for an ambiguous term."""
        if term not in self.stats["ambiguity_resolutions"]:
            return None
            
        resolutions = self.stats["ambiguity_resolutions"][term]
        if not resolutions:
            return None
            
        # Find the most common resolution
        best_resolution = max(resolutions.items(), key=lambda x: x[1])
        return best_resolution[0]

class QueryRewriter:
    """Rewrites user queries to be schema-compatible."""
    
    def __init__(self, schema_profile: SchemaProfile, 
                rule_engine: Optional[BusinessRuleEngine] = None,
                stats: Optional[QueryRewriteStats] = None,
                nlp_model_path: Optional[str] = None):
        """Initialize the query rewriter."""
        self.schema_profile = schema_profile
        self.rule_engine = rule_engine
        self.stats = stats
        self.nlp_model = None
        
        if nlp_model_path and os.path.exists(nlp_model_path):
            self._load_nlp_model(nlp_model_path)
    
    def _load_nlp_model(self, model_path: str) -> bool:
        """Load NLP model for query rewriting."""
        try:
            # This is a placeholder - in a real implementation, we would load
            # a spaCy model, Hugging Face transformer, or other NLP model
            logger.info(f"Loading NLP model from {model_path}")
            # self.nlp_model = spacy.load(model_path)
            return True
        except Exception as e:
            logger.error(f"Failed to load NLP model: {e}")
            return False
    
    def rewrite_query(self, query: str) -> Tuple[str, Dict[str, Any]]:
        """
        Rewrite a user query to be schema-compatible.
        Returns the rewritten query and metadata about the rewrite.
        """
        # 1. Extract meaningful terms from the query
        query_terms = extract_query_terms(query)
        
        # 2. Match terms to schema columns
        column_matches = self.schema_profile.find_matching_columns(query_terms)
        
        # 3. Initialize metadata
        metadata = {
            "original_query": query,
            "extracted_terms": query_terms,
            "column_matches": {},
            "rewrites": [],
            "ambiguities": [],
            "missing_terms": [term for term in query_terms if term not in column_matches]
        }
        
        # Convert column matches to serializable format for metadata
        for term, matches in column_matches.items():
            if not matches:
                continue
                
            metadata["column_matches"][term] = [
                {"column": col.name, "confidence": conf} 
                for col, conf in matches
            ]
        
        # 4. Perform rewrites for high-confidence matches
        rewritten_query = query
        for term, matches in column_matches.items():
            if not matches:
                continue
                
            best_match_col, confidence = matches[0]
            
            # Check for ambiguity (multiple high-confidence matches)
            ambiguous = len([m for m, c in matches if c > 0.7]) > 1
            if ambiguous:
                metadata["ambiguities"].append({
                    "term": term,
                    "candidates": [{"column": col.name, "confidence": conf} 
                                 for col, conf in matches if conf > 0.7]
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
        
        # 5. Apply business rules to ensure query makes sense
        if self.rule_engine:
            # Check for related columns that should be included
            all_matched_columns = set()
            for term_matches in column_matches.values():
                for col, _ in term_matches:
                    if col:
                        all_matched_columns.add(col.name)
            
            # Look for related columns that might be needed
            suggested_additions = []
            for col_name in all_matched_columns:
                col = self.schema_profile.get_column_by_name(col_name)
                if col and col.related_columns:
                    for related_col_name in col.related_columns:
                        related_col = self.schema_profile.get_column_by_name(related_col_name)
                        if not related_col:
                            continue
                            
                        # Check if related column is already mentioned
                        mentioned = False
                        for rewrites in metadata.get("rewrites", []):
                            if rewrites.get("column") == related_col.name:
                                mentioned = True
                                break
                                
                        if not mentioned:
                            suggested_additions.append({
                                "column": related_col.name,
                                "display_name": related_col.display_name,
                                "reason": f"Related to {col.name}"
                            })
            
            if suggested_additions:
                metadata["suggested_additions"] = suggested_additions
        
        # 6. Apply historical statistics for ambiguities and missing terms
        if self.stats:
            # Resolve ambiguities using historical data
            for ambiguity in metadata.get("ambiguities", []):
                term = ambiguity.get("term")
                if not term:
                    continue
                    
                resolution = self.stats.get_ambiguity_resolution(term)
                if resolution:
                    # Check if this ambiguity was already resolved in rewrites
                    already_resolved = False
                    for rewrite in metadata.get("rewrites", []):
                        if rewrite.get("original") == term:
                            already_resolved = True
                            break
                    
                    if not already_resolved:
                        # Get the display name for the resolved column
                        resolved_col = self.schema_profile.get_column_by_name(resolution)
                        if resolved_col:
                            new_term = resolved_col.display_name
                            
                            # Replace the term
                            rewritten_query = self._replace_term(rewritten_query, term, new_term)
                            
                            metadata["rewrites"].append({
                                "original": term,
                                "rewritten": new_term,
                                "column": resolution,
                                "confidence": 0.8,  # High confidence from historical data
                                "source": "historical"
                            })
            
            # Handle missing terms using historical data
            for term in metadata.get("missing_terms", []):
                rewrite = self.stats.get_best_rewrite(term)
                if rewrite:
                    # Get the column for this rewrite
                    for col in self.schema_profile.columns:
                        if col.name == rewrite or col.display_name == rewrite:
                            # Replace the term
                            rewritten_query = self._replace_term(rewritten_query, term, col.display_name)
                            
                            metadata["rewrites"].append({
                                "original": term,
                                "rewritten": col.display_name,
                                "column": col.name,
                                "confidence": 0.75,  # Good confidence from historical data
                                "source": "historical"
                            })
                            
                            # Remove from missing terms
                            metadata["missing_terms"].remove(term)
                            break
        
        # 7. Apply NLP-based rewrites if available
        if self.nlp_model:
            nlp_rewrites, nlp_metadata = self._apply_nlp_rewrites(rewritten_query, metadata)
            if nlp_rewrites != rewritten_query:
                rewritten_query = nlp_rewrites
                metadata["nlp_rewrites"] = nlp_metadata
        
        # 8. Record statistics
        if self.stats:
            self.stats.record_rewrite(query, rewritten_query, metadata)
        
        # Add final rewritten query to metadata
        metadata["rewritten_query"] = rewritten_query
        
        return rewritten_query, metadata
    
    def _apply_nlp_rewrites(self, query: str, metadata: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Apply NLP-based rewrites to improve query quality."""
        # This is a placeholder for NLP-based rewrites
        # In a real implementation, we would use the NLP model to:
        # 1. Identify entities and intent
        # 2. Resolve ambiguities based on semantic context
        # 3. Rephrase complex or unclear parts of the query
        
        # Just return original query for now
        return query, {"model": "placeholder"}
    
    def _replace_term(self, text: str, old_term: str, new_term: str) -> str:
        """Replace a term in text, handling case variants."""
        if not old_term or old_term == new_term:
            return text
            
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
    
    def load_model(self, model_path: str) -> bool:
        """Load a language model for query rewriting."""
        return self._load_nlp_model(model_path)

class PrecisionScoringEngine:
    """Engine that scores query precision based on various factors."""
    
    def __init__(self, schema_profile: SchemaProfile, model_path: Optional[str] = None):
        """Initialize the precision scoring engine."""
        self.schema_profile = schema_profile
        self.model = None
        
        # Feature weights
        self.feature_weights = {
            "schema_overlap": 0.3,       # How well query terms match schema columns
            "ambiguity": -0.2,           # Penalty for ambiguous terms
            "missing_terms": -0.1,       # Penalty for terms not in schema
            "rewrite_confidence": 0.2,   # Confidence in rewrites
            "related_columns": 0.1,      # Bonus for including related columns
            "historical_success": 0.2    # Historical success of similar queries
        }
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def load_model(self, model_path: str) -> bool:
        """Load a trained model for precision scoring."""
        try:
            with open(model_path, 'rb') as f:
                import pickle
                model_data = pickle.load(f)
                
                # Check if model_data has expected components
                if isinstance(model_data, dict) and "weights" in model_data:
                    self.feature_weights = model_data["weights"]
                    self.model = model_data.get("model")
                    logger.info(f"Loaded precision scoring model from {model_path}")
                    return True
                else:
                    logger.error(f"Invalid model format in {model_path}")
                    return False
                    
        except Exception as e:
            logger.error(f"Failed to load precision scoring model: {e}")
            return False
    
    def save_model(self, model_path: str) -> bool:
        """Save the current model to disk."""
        try:
            import pickle
            
            # Create parent directory if needed
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            
            # Save model data
            model_data = {
                "weights": self.feature_weights,
                "model": self.model,
                "timestamp": datetime.now().isoformat()
            }
            
            with open(model_path, 'wb') as f:
                pickle.dump(model_data, f)
                
            logger.info(f"Saved precision scoring model to {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save precision scoring model: {e}")
            return False
    
    def predict_precision(self, query: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict the precision score for a query.
        Returns detailed information about the score and factors.
        """
        # 1. Extract features
        features = self._extract_features(query, metadata)
        
        # 2. Apply statistical scoring model
        if self.model:
            # For now, just pretend we're using a model
            # In a real implementation, we would prepare features for the model
            # and then call self.model.predict()
            score = 0.5  # Default score
        else:
            # Use heuristic model based on feature weights
            score = self._calculate_score(features)
        
        # 3. Determine confidence level
        if score >= 0.7:
            confidence = "high"
        elif score >= 0.4:
            confidence = "medium"
        else:
            confidence = "low"
        
        # 4. Generate reasoning text
        reasoning = self._generate_reasoning(features, score)
        
        # 5. Prepare result
        result = {
            "score": score,
            "confidence": confidence,
            "reasoning": reasoning,
            "features": features,
            "feature_weights": self.feature_weights.copy(),
            "threshold": {
                "high": 0.7,
                "medium": 0.4,
                "low": 0.0
            }
        }
        
        return result
    
    def _extract_features(self, query: str, metadata: Dict[str, Any]) -> Dict[str, float]:
        """Extract features from query metadata for precision scoring."""
        features = {}
        
        # 1. Schema overlap (what % of terms match schema columns)
        total_terms = len(metadata.get("column_matches", {})) + len(metadata.get("missing_terms", []))
        matched_terms = len(metadata.get("column_matches", {}))
        
        if total_terms > 0:
            features["schema_overlap"] = matched_terms / total_terms
        else:
            features["schema_overlap"] = 0.0
        
        # 2. Ambiguity (penalty for ambiguous terms)
        ambiguity_count = len(metadata.get("ambiguities", []))
        if total_terms > 0:
            features["ambiguity"] = ambiguity_count / total_terms
        else:
            features["ambiguity"] = 0.0
        
        # 3. Missing terms (penalty for terms not in schema)
        if total_terms > 0:
            features["missing_terms"] = len(metadata.get("missing_terms", [])) / total_terms
        else:
            features["missing_terms"] = 0.0
        
        # 4. Rewrite confidence
        rewrite_conf_sum = 0.0
        rewrite_count = 0
        
        for rewrite in metadata.get("rewrites", []):
            confidence = rewrite.get("confidence", 0.0)
            rewrite_conf_sum += confidence
            rewrite_count += 1
        
        if rewrite_count > 0:
            features["rewrite_confidence"] = rewrite_conf_sum / rewrite_count
        else:
            features["rewrite_confidence"] = 1.0  # No rewrites needed, high confidence
        
        # 5. Related columns (bonus for including related columns)
        # Calculate how many suggested related columns are included
        suggested = len(metadata.get("suggested_additions", []))
        if suggested > 0:
            # Penalize for missing related columns
            features["related_columns"] = 0.0
        else:
            # No missing related columns
            features["related_columns"] = 1.0
        
        # 6. Historical success (placeholder)
        # In a real implementation, this would look at how similar queries
        # have performed in the past
        features["historical_success"] = 0.5  # Neutral
        
        return features
    
    def _calculate_score(self, features: Dict[str, float]) -> float:
        """Calculate precision score using feature weights."""
        score = 0.5  # Start with neutral score
        
        for feature, value in features.items():
            if feature in self.feature_weights:
                weight = self.feature_weights[feature]
                score += value * weight
        
        # Ensure score is between 0 and 1
        score = max(0.0, min(1.0, score))
        
        return score
    
    def _generate_reasoning(self, features: Dict[str, float], score: float) -> str:
        """Generate human-readable reasoning for the score."""
        reasons = []
        
        # Term matching
        schema_overlap = features.get("schema_overlap", 0.0)
        if schema_overlap > 0.8:
            reasons.append("Strong schema column matches")
        elif schema_overlap > 0.5:
            reasons.append("Good schema column matches")
        elif schema_overlap > 0.0:
            reasons.append("Some schema column matches")
        else:
            reasons.append("No schema column matches")
        
        # Ambiguity
        ambiguity = features.get("ambiguity", 0.0)
        if ambiguity > 0.3:
            reasons.append("High query ambiguity")
        elif ambiguity > 0.0:
            reasons.append("Some query ambiguity")
        
        # Missing terms
        missing = features.get("missing_terms", 0.0)
        if missing > 0.5:
            reasons.append("Many unrecognized terms")
        elif missing > 0.2:
            reasons.append("Some unrecognized terms")
        
        # Rewrite confidence
        rewrite_conf = features.get("rewrite_confidence", 0.0)
        if rewrite_conf > 0.8:
            reasons.append("High confidence rewrites")
        elif rewrite_conf > 0.5:
            reasons.append("Moderate confidence rewrites")
        elif rewrite_conf > 0.0:
            reasons.append("Low confidence rewrites")
        
        # Related columns
        related = features.get("related_columns", 0.0)
        if related < 0.5:
            reasons.append("Missing related columns")
        
        return ", ".join(reasons)

def test_query_rewriter():
    """Test the query rewriter with sample data."""
    # Create simple schema profile for testing
    from src.utils.adaptive_schema import SchemaProfile, SchemaColumn
    
    columns = [
        SchemaColumn(
            name="total_gross_profit",
            display_name="Total Gross Profit",
            description="Total gross profit across all departments",
            data_type="decimal",
            aliases=["total gross", "gross profit", "total gp", "profit"]
        ),
        SchemaColumn(
            name="frontend_gross",
            display_name="Front End Gross",
            description="Gross profit from the vehicle sale before F&I products",
            data_type="decimal",
            aliases=["front end", "front gross", "vehicle gross"],
            related_columns=["total_gross_profit", "backend_gross"]
        ),
        SchemaColumn(
            name="backend_gross",
            display_name="Back End Gross",
            description="Gross profit from F&I products and services",
            data_type="decimal",
            aliases=["back end", "back gross", "finance gross", "f&i gross"],
            related_columns=["total_gross_profit", "frontend_gross"]
        ),
        SchemaColumn(
            name="units_sold",
            display_name="Units Sold",
            description="Number of vehicles sold",
            data_type="integer",
            aliases=["sales count", "vehicles sold", "unit sales", "sales"]
        )
    ]
    
    profile = SchemaProfile(
        id="test_profile",
        name="Test Profile",
        description="Test schema profile",
        role=ExecRole.GENERAL_MANAGER,
        columns=columns
    )
    
    # Create a query rewriter
    rewriter = QueryRewriter(profile)
    
    # Create precision scorer
    scorer = PrecisionScoringEngine(profile)
    
    # Test queries
    test_queries = [
        "What is our total gross profit?",
        "Show me front gross by salesperson",
        "How many units did we sell last month?",
        "What's the average profit per sale?",
        "Compare front and back end gross"
    ]
    
    print("\n=== QUERY REWRITER TEST ===")
    for query in test_queries:
        print(f"\nOriginal: {query}")
        
        # Rewrite query
        rewritten, metadata = rewriter.rewrite_query(query)
        print(f"Rewritten: {rewritten}")
        
        # Score the query
        precision = scorer.predict_precision(rewritten, metadata)
        print(f"Precision: {precision['score']:.2f} ({precision['confidence']})")
        print(f"Reasoning: {precision['reasoning']}")
        
        # Show column matches
        for term, matches in metadata.get("column_matches", {}).items():
            if matches:
                col = matches[0]["column"]
                conf = matches[0]["confidence"]
                print(f"  • {term} → {col} ({conf:.2f})")
                
        # Show rewrites
        for rewrite in metadata.get("rewrites", []):
            print(f"  • Rewrote '{rewrite['original']}' to '{rewrite['rewritten']}'")
        
        print("-" * 50)

if __name__ == "__main__":
    # Initialize logging
    logging.basicConfig(level=logging.INFO)
    
    # Run the test
    test_query_rewriter()