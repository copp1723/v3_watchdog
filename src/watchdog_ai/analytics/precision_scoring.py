"""
Precision Scoring Engine for Watchdog AI.

This module provides an engine for scoring query precision, which predicts
how likely a query is to produce accurate results. It can use statistical
features or ML models to make predictions.
"""

import os
import pickle
import logging
import json
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime
import numpy as np
from pathlib import Path

# Import schema components
from src.utils.adaptive_schema import SchemaProfile, SchemaColumn, ExecRole

logger = logging.getLogger(__name__)

class QueryFeatureExtractor:
    """Extracts features from query metadata for precision prediction."""
    
    def __init__(self, schema_profile: SchemaProfile):
        """Initialize with a schema profile."""
        self.schema_profile = schema_profile
    
    def extract_features(self, query: str, metadata: Dict[str, Any]) -> Dict[str, float]:
        """Extract query features for precision prediction."""
        features = {}
        
        # 1. Schema overlap (what % of terms match schema columns)
        total_terms = len(metadata.get("column_matches", {})) + len(metadata.get("missing_terms", []))
        matched_terms = len(metadata.get("column_matches", {}))
        
        if total_terms > 0:
            features["schema_overlap"] = matched_terms / total_terms
        else:
            features["schema_overlap"] = 0.0
        
        # 2. Average match confidence
        conf_sum = 0.0
        conf_count = 0
        
        for matches in metadata.get("column_matches", {}).values():
            if matches:
                conf_sum += matches[0]["confidence"]
                conf_count += 1
        
        if conf_count > 0:
            features["avg_confidence"] = conf_sum / conf_count
        else:
            features["avg_confidence"] = 0.0
        
        # 3. Ambiguity (penalty for ambiguous terms)
        ambiguity_count = len(metadata.get("ambiguities", []))
        features["ambiguity_count"] = ambiguity_count
        
        if total_terms > 0:
            features["ambiguity_ratio"] = ambiguity_count / total_terms
        else:
            features["ambiguity_ratio"] = 0.0
        
        # 4. Missing terms (penalty for terms not in schema)
        missing_count = len(metadata.get("missing_terms", []))
        features["missing_count"] = missing_count
        
        if total_terms > 0:
            features["missing_ratio"] = missing_count / total_terms
        else:
            features["missing_ratio"] = 0.0
        
        # 5. Rewrite metrics
        rewrite_count = len(metadata.get("rewrites", []))
        features["rewrite_count"] = rewrite_count
        
        if rewrite_count > 0:
            # Calculate average rewrite confidence
            rewrite_conf_sum = sum(
                rewrite.get("confidence", 0.0) 
                for rewrite in metadata.get("rewrites", [])
            )
            features["rewrite_confidence"] = rewrite_conf_sum / rewrite_count
        else:
            features["rewrite_confidence"] = 1.0  # No rewrites needed, high confidence
        
        # 6. Query complexity metrics
        query_length = len(query.split())
        features["query_length"] = min(1.0, query_length / 20)  # Normalize to 0-1
        
        # 7. Related columns metrics
        suggested = len(metadata.get("suggested_additions", []))
        features["suggested_additions"] = suggested
        
        if suggested > 0:
            # Penalize for missing related columns
            features["related_columns_coverage"] = 0.0
        else:
            # No missing related columns
            features["related_columns_coverage"] = 1.0
        
        return features

class FeatureVectorizer:
    """Converts feature dictionaries to numeric vectors for ML models."""
    
    def __init__(self, feature_names: Optional[List[str]] = None):
        """Initialize with optional list of feature names."""
        self.feature_names = feature_names or [
            "schema_overlap", "avg_confidence", "ambiguity_ratio", 
            "missing_ratio", "rewrite_confidence", "related_columns_coverage"
        ]
    
    def transform(self, features: Union[Dict[str, float], List[Dict[str, float]]]) -> np.ndarray:
        """Transform features to a vector (or matrix for multiple samples)."""
        if isinstance(features, dict):
            # Single sample
            return self._transform_single(features)
            
        # Multiple samples
        return np.array([self._transform_single(f) for f in features])
    
    def _transform_single(self, features: Dict[str, float]) -> np.ndarray:
        """Transform a single sample to a vector."""
        return np.array([features.get(name, 0.0) for name in self.feature_names])
    
    def get_feature_names(self) -> List[str]:
        """Get the feature names used by this vectorizer."""
        return self.feature_names.copy()
    
    def save(self, file_path: str) -> bool:
        """Save the vectorizer to a file."""
        try:
            data = {
                "feature_names": self.feature_names,
                "timestamp": datetime.now().isoformat()
            }
            
            with open(file_path, 'wb') as f:
                pickle.dump(data, f)
                
            return True
        except Exception as e:
            logger.error(f"Failed to save vectorizer: {e}")
            return False
    
    @classmethod
    def load(cls, file_path: str) -> Optional['FeatureVectorizer']:
        """Load a vectorizer from a file."""
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
                
            if not isinstance(data, dict) or "feature_names" not in data:
                logger.error(f"Invalid vectorizer format in {file_path}")
                return None
                
            return cls(data["feature_names"])
        except Exception as e:
            logger.error(f"Failed to load vectorizer: {e}")
            return None

class PrecisionScoringRule:
    """Rule for scoring query precision based on feature thresholds."""
    
    def __init__(self, feature: str, operator: str, threshold: float, 
                adjustment: float, description: str):
        """Initialize a precision scoring rule."""
        self.feature = feature
        self.operator = operator
        self.threshold = threshold
        self.adjustment = adjustment
        self.description = description
    
    def apply(self, features: Dict[str, float]) -> Tuple[float, bool, str]:
        """
        Apply the rule to features.
        Returns (adjustment, triggered, description).
        """
        value = features.get(self.feature, 0.0)
        triggered = False
        
        if self.operator == ">" and value > self.threshold:
            triggered = True
        elif self.operator == ">=" and value >= self.threshold:
            triggered = True
        elif self.operator == "<" and value < self.threshold:
            triggered = True
        elif self.operator == "<=" and value <= self.threshold:
            triggered = True
        elif self.operator == "==" and value == self.threshold:
            triggered = True
        
        if triggered:
            return self.adjustment, True, self.description
        else:
            return 0.0, False, ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "feature": self.feature,
            "operator": self.operator,
            "threshold": self.threshold,
            "adjustment": self.adjustment,
            "description": self.description
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PrecisionScoringRule':
        """Create from dictionary representation."""
        return cls(
            feature=data["feature"],
            operator=data["operator"],
            threshold=data["threshold"],
            adjustment=data["adjustment"],
            description=data["description"]
        )

class HeuristicPrecisionModel:
    """Heuristic model for scoring query precision using rule-based approach."""
    
    def __init__(self, base_score: float = 0.5):
        """Initialize with a base score."""
        self.base_score = base_score
        self.rules = []
        
        # Initialize with default rules
        self._add_default_rules()
    
    def _add_default_rules(self) -> None:
        """Add default precision scoring rules."""
        self.rules = [
            # Schema overlap rules
            PrecisionScoringRule(
                feature="schema_overlap",
                operator=">=",
                threshold=0.8,
                adjustment=0.2,
                description="Strong schema column matches"
            ),
            PrecisionScoringRule(
                feature="schema_overlap",
                operator="<",
                threshold=0.5,
                adjustment=-0.1,
                description="Poor schema column matches"
            ),
            
            # Confidence rules
            PrecisionScoringRule(
                feature="avg_confidence",
                operator=">=",
                threshold=0.8,
                adjustment=0.15,
                description="High confidence matches"
            ),
            PrecisionScoringRule(
                feature="avg_confidence",
                operator="<",
                threshold=0.6,
                adjustment=-0.1,
                description="Low confidence matches"
            ),
            
            # Ambiguity rules
            PrecisionScoringRule(
                feature="ambiguity_count",
                operator=">",
                threshold=0,
                adjustment=-0.15,
                description="Contains ambiguous terms"
            ),
            
            # Missing terms rules
            PrecisionScoringRule(
                feature="missing_ratio",
                operator=">",
                threshold=0.3,
                adjustment=-0.2,
                description="Many unrecognized terms"
            ),
            
            # Rewrite rules
            PrecisionScoringRule(
                feature="rewrite_confidence",
                operator="<",
                threshold=0.7,
                adjustment=-0.1,
                description="Low confidence rewrites"
            ),
            
            # Related columns rules
            PrecisionScoringRule(
                feature="related_columns_coverage",
                operator="<",
                threshold=1.0,
                adjustment=-0.1,
                description="Missing related columns"
            )
        ]
    
    def predict(self, features: Dict[str, float]) -> Tuple[float, List[str]]:
        """
        Predict precision score from features.
        Returns (score, reasons).
        """
        score = self.base_score
        reasons = []
        
        for rule in self.rules:
            adjustment, triggered, description = rule.apply(features)
            
            if triggered:
                score += adjustment
                reasons.append(description)
        
        # Ensure score is between 0 and 1
        score = max(0.0, min(1.0, score))
        
        return score, reasons
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "base_score": self.base_score,
            "rules": [rule.to_dict() for rule in self.rules]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'HeuristicPrecisionModel':
        """Create from dictionary representation."""
        model = cls(base_score=data.get("base_score", 0.5))
        
        # Clear default rules and load from data
        model.rules = []
        for rule_data in data.get("rules", []):
            model.rules.append(PrecisionScoringRule.from_dict(rule_data))
            
        return model
    
    def save(self, file_path: str) -> bool:
        """Save the model to a file."""
        try:
            data = self.to_dict()
            data["timestamp"] = datetime.now().isoformat()
            data["model_type"] = "heuristic"
            
            with open(file_path, 'wb') as f:
                pickle.dump(data, f)
                
            return True
        except Exception as e:
            logger.error(f"Failed to save heuristic model: {e}")
            return False
    
    @classmethod
    def load(cls, file_path: str) -> Optional['HeuristicPrecisionModel']:
        """Load a model from a file."""
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
                
            if not isinstance(data, dict) or data.get("model_type") != "heuristic":
                logger.error(f"Invalid model format in {file_path}")
                return None
                
            return cls.from_dict(data)
        except Exception as e:
            logger.error(f"Failed to load heuristic model: {e}")
            return None

class MLPrecisionModel:
    """Machine learning model for scoring query precision."""
    
    def __init__(self, vectorizer: Optional[FeatureVectorizer] = None, 
                model=None):
        """Initialize with optional vectorizer and model."""
        self.vectorizer = vectorizer or FeatureVectorizer()
        self.model = model
        
        # Placeholder for an actual ML model
        # In a real implementation, this would be a scikit-learn model
        # like RandomForestClassifier or LogisticRegression
    
    def predict(self, features: Dict[str, float]) -> float:
        """Predict precision score from features."""
        if self.model is None:
            # Fall back to heuristic approach if no model
            heuristic = HeuristicPrecisionModel()
            score, _ = heuristic.predict(features)
            return score
        
        # Transform features to vector
        X = self.vectorizer.transform(features)
        
        # Make prediction
        # This is a placeholder - in a real implementation, we would
        # use the ML model's predict method
        y_pred = 0.5  # Default score
        
        return y_pred
    
    def train(self, X_train: List[Dict[str, float]], y_train: List[float]) -> bool:
        """Train the model on labeled data."""
        # Placeholder for actual model training
        # In a real implementation, we would:
        # 1. Transform features using self.vectorizer
        # 2. Initialize and train an ML model
        # 3. Evaluate performance
        
        logger.info(f"Training ML model on {len(X_train)} samples")
        
        # Initialize a placeholder model (just stores the average score)
        self.model = {"avg_score": sum(y_train) / len(y_train) if y_train else 0.5}
        
        return True
    
    def save(self, file_path: str, vectorizer_path: Optional[str] = None) -> bool:
        """Save the model and optionally the vectorizer."""
        try:
            # Save vectorizer if path provided
            if vectorizer_path and self.vectorizer:
                self.vectorizer.save(vectorizer_path)
            
            # Save model
            data = {
                "model": self.model,
                "timestamp": datetime.now().isoformat(),
                "model_type": "ml"
            }
            
            with open(file_path, 'wb') as f:
                pickle.dump(data, f)
                
            return True
        except Exception as e:
            logger.error(f"Failed to save ML model: {e}")
            return False
    
    @classmethod
    def load(cls, file_path: str, vectorizer_path: Optional[str] = None) -> Optional['MLPrecisionModel']:
        """Load a model and optionally the vectorizer."""
        try:
            # Load vectorizer if available
            vectorizer = None
            if vectorizer_path and os.path.exists(vectorizer_path):
                vectorizer = FeatureVectorizer.load(vectorizer_path)
            
            # Load model
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
                
            if not isinstance(data, dict) or data.get("model_type") != "ml":
                logger.error(f"Invalid model format in {file_path}")
                return None
                
            return cls(vectorizer=vectorizer, model=data.get("model"))
        except Exception as e:
            logger.error(f"Failed to load ML model: {e}")
            return None

class PrecisionScoringEngine:
    """Engine for predicting query precision using statistical features or ML models."""
    
    def __init__(self, schema_profile: SchemaProfile, model_path: Optional[str] = None):
        """Initialize the precision scoring engine."""
        self.schema_profile = schema_profile
        self.feature_extractor = QueryFeatureExtractor(schema_profile)
        self.model = None
        
        # Try to load model from path
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            # Use default heuristic model
            self.model = HeuristicPrecisionModel()
    
    def load_model(self, model_path: str) -> bool:
        """Load a model from the specified path."""
        try:
            with open(model_path, 'rb') as f:
                data = pickle.load(f)
                
            if not isinstance(data, dict):
                logger.error(f"Invalid model format in {model_path}")
                return False
                
            model_type = data.get("model_type", "heuristic")
            
            if model_type == "heuristic":
                self.model = HeuristicPrecisionModel.from_dict(data)
            elif model_type == "ml":
                # Determine vectorizer path
                vectorizer_path = None
                base_dir = os.path.dirname(model_path)
                base_name = os.path.basename(model_path).split('.')[0]
                possible_vectorizer = os.path.join(base_dir, f"{base_name}_vectorizer.pkl")
                
                if os.path.exists(possible_vectorizer):
                    vectorizer_path = possible_vectorizer
                    
                self.model = MLPrecisionModel.load(model_path, vectorizer_path)
            else:
                logger.error(f"Unknown model type: {model_type}")
                return False
                
            logger.info(f"Loaded precision scoring model from {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load precision scoring model: {e}")
            return False
    
    def save_model(self, model_path: str) -> bool:
        """Save the current model to the specified path."""
        if self.model is None:
            logger.error("No model to save")
            return False
            
        try:
            # Make sure directory exists
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            
            # Save appropriate model type
            if isinstance(self.model, HeuristicPrecisionModel):
                return self.model.save(model_path)
            elif isinstance(self.model, MLPrecisionModel):
                # Also save vectorizer
                base_dir = os.path.dirname(model_path)
                base_name = os.path.basename(model_path).split('.')[0]
                vectorizer_path = os.path.join(base_dir, f"{base_name}_vectorizer.pkl")
                
                return self.model.save(model_path, vectorizer_path)
            else:
                logger.error(f"Unknown model type: {type(self.model)}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to save precision scoring model: {e}")
            return False
    
    def predict_precision(self, query: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict the precision of a query based on its metadata.
        Returns detailed information about the prediction.
        """
        # Extract features
        features = self.feature_extractor.extract_features(query, metadata)
        
        # Make prediction
        if isinstance(self.model, HeuristicPrecisionModel):
            score, reasons = self.model.predict(features)
        else:
            # ML model
            score = self.model.predict(features)
            
            # Generate reasons based on key feature thresholds
            reasons = self._generate_reasons_from_features(features)
        
        # Determine confidence level
        if score >= 0.7:
            confidence = "high"
        elif score >= 0.4:
            confidence = "medium"
        else:
            confidence = "low"
        
        # Prepare result
        result = {
            "score": score,
            "confidence": confidence,
            "reasoning": ", ".join(reasons) if reasons else "No specific factors",
            "features": features,
            "threshold": {
                "high": 0.7,
                "medium": 0.4,
                "low": 0.0
            }
        }
        
        return result
    
    def _generate_reasons_from_features(self, features: Dict[str, float]) -> List[str]:
        """Generate human-readable reasons from features."""
        reasons = []
        
        # Schema overlap
        schema_overlap = features.get("schema_overlap", 0.0)
        if schema_overlap >= 0.8:
            reasons.append("Strong schema column matches")
        elif schema_overlap >= 0.5:
            reasons.append("Good schema column matches")
        elif schema_overlap < 0.3:
            reasons.append("Poor schema column matches")
        
        # Match confidence
        avg_confidence = features.get("avg_confidence", 0.0)
        if avg_confidence >= 0.8:
            reasons.append("High confidence matches")
        elif avg_confidence < 0.6:
            reasons.append("Low confidence matches")
        
        # Ambiguity
        ambiguity_count = features.get("ambiguity_count", 0)
        if ambiguity_count > 0:
            reasons.append("Contains ambiguous terms")
        
        # Missing terms
        missing_ratio = features.get("missing_ratio", 0.0)
        if missing_ratio >= 0.3:
            reasons.append("Many unrecognized terms")
        elif missing_ratio > 0.0:
            reasons.append("Some unrecognized terms")
        
        # Related columns
        related_coverage = features.get("related_columns_coverage", 1.0)
        if related_coverage < 1.0:
            reasons.append("Missing related columns")
        
        return reasons
    
    def train_on_feedback(self, feedback_data: List[Dict[str, Any]]) -> bool:
        """
        Train a new model using feedback data.
        Each feedback item should contain query, metadata, and actual_precision.
        """
        if not feedback_data:
            logger.error("No feedback data provided for training")
            return False
            
        try:
            # Extract features and labels
            X_train = []
            y_train = []
            
            for item in feedback_data:
                query = item.get("query", "")
                metadata = item.get("metadata", {})
                actual_precision = item.get("actual_precision", 0.5)
                
                if not query or not metadata:
                    continue
                    
                features = self.feature_extractor.extract_features(query, metadata)
                X_train.append(features)
                y_train.append(actual_precision)
            
            if not X_train:
                logger.error("No valid training examples found")
                return False
                
            # Create and train ML model
            vectorizer = FeatureVectorizer()
            ml_model = MLPrecisionModel(vectorizer)
            success = ml_model.train(X_train, y_train)
            
            if success:
                self.model = ml_model
                return True
            else:
                logger.error("Failed to train ML model")
                return False
                
        except Exception as e:
            logger.error(f"Error training precision model: {e}")
            return False

def test_precision_scoring():
    """Test the precision scoring engine with sample data."""
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
    
    # Create a precision scoring engine
    engine = PrecisionScoringEngine(profile)
    
    # Test queries with metadata
    test_cases = [
        {
            "query": "What is our total gross profit?",
            "metadata": {
                "column_matches": {
                    "gross": [{"column": "total_gross_profit", "confidence": 0.9}],
                    "profit": [{"column": "total_gross_profit", "confidence": 0.9}]
                },
                "missing_terms": [],
                "ambiguities": []
            }
        },
        {
            "query": "Show me front gross by salesperson",
            "metadata": {
                "column_matches": {
                    "front": [{"column": "frontend_gross", "confidence": 0.8}],
                    "gross": [{"column": "frontend_gross", "confidence": 0.8}]
                },
                "missing_terms": ["salesperson"],
                "ambiguities": []
            }
        },
        {
            "query": "Compare front and back end gross by month",
            "metadata": {
                "column_matches": {
                    "front": [{"column": "frontend_gross", "confidence": 0.8}],
                    "back": [{"column": "backend_gross", "confidence": 0.8}],
                    "gross": [
                        {"column": "frontend_gross", "confidence": 0.7},
                        {"column": "backend_gross", "confidence": 0.7},
                        {"column": "total_gross_profit", "confidence": 0.6}
                    ]
                },
                "missing_terms": ["month"],
                "ambiguities": [
                    {
                        "term": "gross",
                        "candidates": [
                            {"column": "frontend_gross", "confidence": 0.7},
                            {"column": "backend_gross", "confidence": 0.7},
                            {"column": "total_gross_profit", "confidence": 0.6}
                        ]
                    }
                ]
            }
        },
        {
            "query": "How many bananas did we sell last Tuesday?",
            "metadata": {
                "column_matches": {
                    "sell": [{"column": "units_sold", "confidence": 0.7}]
                },
                "missing_terms": ["bananas", "tuesday"],
                "ambiguities": []
            }
        }
    ]
    
    print("\n=== PRECISION SCORING TEST ===")
    for i, test_case in enumerate(test_cases):
        query = test_case["query"]
        metadata = test_case["metadata"]
        
        print(f"\n{i+1}. Query: '{query}'")
        
        # Score the query
        result = engine.predict_precision(query, metadata)
        
        print(f"   Score: {result['score']:.2f} ({result['confidence']})")
        print(f"   Reasoning: {result['reasoning']}")
        
        # Show key features
        print("   Key Features:")
        for feature, value in result["features"].items():
            if feature in ["schema_overlap", "avg_confidence", "missing_ratio", "ambiguity_count"]:
                print(f"     • {feature}: {value:.2f}")
        
        print("-" * 50)
    
    # Test saving and loading
    temp_path = "temp_precision_model.pkl"
    print(f"\nSaving model to {temp_path}...")
    engine.save_model(temp_path)
    
    print("Loading model...")
    new_engine = PrecisionScoringEngine(profile, temp_path)
    
    # Compare results
    query = test_cases[0]["query"]
    metadata = test_cases[0]["metadata"]
    
    orig_result = engine.predict_precision(query, metadata)
    new_result = new_engine.predict_precision(query, metadata)
    
    print(f"\nOriginal engine score: {orig_result['score']:.2f}")
    print(f"Loaded engine score: {new_result['score']:.2f}")
    
    # Clean up
    if os.path.exists(temp_path):
        os.remove(temp_path)
        print(f"Removed temporary file {temp_path}")

if __name__ == "__main__":
    # Initialize logging
    logging.basicConfig(level=logging.INFO)
    
    # Run the test
    test_precision_scoring()