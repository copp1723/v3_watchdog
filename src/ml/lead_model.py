"""
Lead Outcome Prediction Model for V3 Watchdog AI.

Provides predictive modeling for forecasting sales outcomes from leads
and associated probability scoring for decision support.
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union, Tuple
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import logging

logger = logging.getLogger(__name__)

# Model constants
DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "data",
    "models",
    "lead_outcome_model.pkl"
)

DEFAULT_FEATURES = [
    "rep",           # Sales rep assigned
    "vehicle",       # Vehicle/model of interest
    "source",        # Lead source
    "hour_created",  # Hour of day lead was created
    "day_created",   # Day of week lead was created
    "contact_delay", # Hours between lead creation and first contact
]

# Outcome timeframes
OUTCOME_TIMEFRAMES = {
    "short": 14,  # 14-day outcome
    "medium": 30, # 30-day outcome
}

class LeadOutcomePredictor:
    """Predictive model for lead outcomes."""
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 features: Optional[List[str]] = None,
                 model_type: str = "random_forest"):
        """
        Initialize the lead outcome predictor.
        
        Args:
            model_path: Optional path to the saved model file
            features: Optional list of features to use in the model
            model_type: Model type to use ("random_forest", "gradient_boosting", or "logistic")
        """
        self.model_path = model_path or DEFAULT_MODEL_PATH
        self.features = features or DEFAULT_FEATURES.copy()
        self.model_type = model_type
        self.model = None
        self.feature_columns = []
        self.preprocessor = None
        self.threshold = 0.5  # Default threshold for binary classification
        self.metadata = {
            "created_at": None,
            "last_trained": None,
            "model_type": model_type,
            "features": self.features,
            "metrics": {},
            "threshold": self.threshold
        }
        
        # Try to load existing model
        if os.path.exists(self.model_path):
            self._load_model()
        else:
            # Create a new model
            self._initialize_model()
    
    def _initialize_model(self) -> None:
        """Initialize a new model with default parameters."""
        if self.model_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=10,
                random_state=42
            )
        elif self.model_type == "gradient_boosting":
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
        else:  # logistic regression as default fallback
            self.model = LogisticRegression(
                C=1.0,
                max_iter=1000,
                random_state=42
            )
        
        # Set created timestamp
        self.metadata["created_at"] = datetime.now().isoformat()
        self.metadata["model_type"] = self.model_type
    
    def _load_model(self) -> None:
        """Load model from disk."""
        try:
            with open(self.model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data.get("model")
            self.preprocessor = model_data.get("preprocessor")
            self.metadata = model_data.get("metadata", {})
            self.feature_columns = model_data.get("feature_columns", [])
            self.threshold = self.metadata.get("threshold", 0.5)
            
            logger.info(f"Loaded model from {self.model_path}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self._initialize_model()
    
    def _save_model(self) -> None:
        """Save model to disk."""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            
            # Update metadata
            self.metadata["last_saved"] = datetime.now().isoformat()
            
            # Create model data dict
            model_data = {
                "model": self.model,
                "preprocessor": self.preprocessor,
                "metadata": self.metadata,
                "feature_columns": self.feature_columns
            }
            
            # Save to file
            with open(self.model_path, 'wb') as f:
                pickle.dump(model_data, f)
                
            logger.info(f"Saved model to {self.model_path}")
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
    
    def _prepare_data(self, df: pd.DataFrame, target_col: str = "sold") -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        Prepare data for training or prediction.
        
        Args:
            df: DataFrame with lead data
            target_col: Column name for the target variable (for training)
            
        Returns:
            Tuple of (processed features DataFrame, target Series if available)
        """
        # Create a working copy
        work_df = df.copy()
        
        # Extract target if it exists
        y = None
        if target_col in work_df.columns:
            y = work_df[target_col]
            work_df = work_df.drop(columns=[target_col])
        
        # Extract temporal features
        if 'created_date' in work_df.columns:
            if pd.api.types.is_datetime64_any_dtype(work_df['created_date']):
                created_dt = work_df['created_date']
            else:
                created_dt = pd.to_datetime(work_df['created_date'], errors='coerce')
            
            # Extract hour and day of week
            work_df['hour_created'] = created_dt.dt.hour
            work_df['day_created'] = created_dt.dt.dayofweek
        
        # Calculate contact delay if contact date exists
        if 'created_date' in work_df.columns and 'contacted_date' in work_df.columns:
            if not pd.api.types.is_datetime64_any_dtype(work_df['contacted_date']):
                work_df['contacted_date'] = pd.to_datetime(work_df['contacted_date'], errors='coerce')
            
            # Calculate delay in hours
            work_df['contact_delay'] = (work_df['contacted_date'] - work_df['created_date']).dt.total_seconds() / 3600
            
            # Handle negative values and missing data
            work_df.loc[work_df['contact_delay'] < 0, 'contact_delay'] = np.nan
            work_df['contact_delay'].fillna(work_df['contact_delay'].median(), inplace=True)
        else:
            # Add placeholder if contact delay can't be calculated
            work_df['contact_delay'] = 0
        
        # Standardize column names
        rename_map = {
            'SalesRep': 'rep',
            'LeadSource': 'source',
            'Model': 'vehicle',
            'VehicleModel': 'vehicle'
        }
        work_df.rename(columns={k: v for k, v in rename_map.items() if k in work_df.columns}, inplace=True)
        
        # Ensure all required feature columns exist
        for feature in self.features:
            if feature not in work_df.columns:
                work_df[feature] = None
        
        # Select only the needed feature columns
        X = work_df[self.features].copy()
        
        # Fill missing values
        for col in X.columns:
            if X[col].dtype.kind in 'ifc':  # numeric
                X[col].fillna(X[col].median() if not X[col].isna().all() else 0, inplace=True)
            else:  # categorical
                X[col].fillna('Unknown', inplace=True)
        
        return X, y
    
    def _create_preprocessor(self, X: pd.DataFrame) -> ColumnTransformer:
        """
        Create a preprocessor for the input features.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            ColumnTransformer for preprocessing
        """
        # Identify numeric and categorical columns
        numeric_features = []
        categorical_features = []
        
        for col in X.columns:
            if X[col].dtype.kind in 'ifc':
                numeric_features.append(col)
            else:
                categorical_features.append(col)
        
        # Create preprocessor
        numeric_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(handle_unknown='ignore')
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ],
            remainder='passthrough'
        )
        
        return preprocessor
    
    def train(self, df: pd.DataFrame, target_col: str = "sold", 
              test_size: float = 0.2, tune_hyperparameters: bool = False) -> Dict[str, Any]:
        """
        Train the lead outcome prediction model.
        
        Args:
            df: DataFrame with lead data
            target_col: Column with target outcome (1 for sold, 0 for not sold)
            test_size: Fraction of data to use for testing
            tune_hyperparameters: Whether to tune hyperparameters using grid search
            
        Returns:
            Dictionary with training results and metrics
        """
        # Prepare data
        X, y = self._prepare_data(df, target_col)
        
        if y is None or len(y) == 0:
            return {"error": "No target data available for training"}
        
        if len(X) < 10:
            return {"error": "Insufficient data for training (minimum 10 samples required)"}
        
        # Store feature columns
        self.feature_columns = X.columns.tolist()
        
        # Create preprocessor
        self.preprocessor = self._create_preprocessor(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        if tune_hyperparameters:
            # Define parameter grid based on model type
            if self.model_type == "random_forest":
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5, 10]
                }
                base_model = RandomForestClassifier(random_state=42)
            elif self.model_type == "gradient_boosting":
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.1, 0.2]
                }
                base_model = GradientBoostingClassifier(random_state=42)
            else:  # logistic regression
                param_grid = {
                    'C': [0.1, 1.0, 10.0],
                    'penalty': ['l2'],
                    'solver': ['liblinear', 'saga']
                }
                base_model = LogisticRegression(random_state=42, max_iter=1000)
            
            # Create pipeline
            pipeline = Pipeline([
                ('preprocessor', self.preprocessor),
                ('classifier', base_model)
            ])
            
            # Grid search
            grid_search = GridSearchCV(
                pipeline, 
                param_grid=param_grid,
                cv=5,
                scoring='roc_auc',
                n_jobs=-1
            )
            
            # Train with grid search
            grid_search.fit(X_train, y_train)
            
            # Get best model
            self.model = grid_search.best_estimator_.named_steps['classifier']
            
            # Update metadata with best parameters
            self.metadata["best_params"] = grid_search.best_params_
            
        else:
            # Create pipeline
            pipeline = Pipeline([
                ('preprocessor', self.preprocessor),
                ('classifier', self.model)
            ])
            
            # Train model
            pipeline.fit(X_train, y_train)
        
        # Make predictions on test set
        pipeline = Pipeline([
            ('preprocessor', self.preprocessor),
            ('classifier', self.model)
        ])
        
        y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba >= self.threshold).astype(int)
        
        # Calculate metrics
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1_score": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_pred_proba)
        }
        
        # Calculate feature importance if available
        if hasattr(self.model, 'feature_importances_'):
            # Get feature names from preprocessor
            feature_names = []
            if hasattr(self.preprocessor, 'get_feature_names_out'):
                feature_names = self.preprocessor.get_feature_names_out()
            
            # Map importances to feature names
            importances = self.model.feature_importances_
            feature_importance = {}
            
            # If feature names match the number of importances
            if len(feature_names) == len(importances):
                for name, importance in zip(feature_names, importances):
                    feature_importance[str(name)] = float(importance)
            else:
                # Fall back to indices
                for i, importance in enumerate(importances):
                    feature_importance[f"feature_{i}"] = float(importance)
            
            metrics["feature_importance"] = feature_importance
        
        # Update metadata
        self.metadata["last_trained"] = datetime.now().isoformat()
        self.metadata["metrics"] = metrics
        self.metadata["samples_trained"] = len(X)
        
        # Save model
        self._save_model()
        
        return {
            "status": "success",
            "metrics": metrics,
            "samples_trained": len(X),
            "samples_tested": len(X_test)
        }
    
    def predict(self, df: pd.DataFrame, timeframe: str = "medium") -> pd.DataFrame:
        """
        Predict lead outcomes with probability scores.
        
        Args:
            df: DataFrame with lead data
            timeframe: Prediction timeframe ("short" for 14-day, "medium" for 30-day)
            
        Returns:
            DataFrame with original data plus predictions and probabilities
        """
        if self.model is None or self.preprocessor is None:
            raise ValueError("Model is not trained. Please train the model first.")
        
        # Prepare data
        X, _ = self._prepare_data(df)
        
        # Create pipeline
        pipeline = Pipeline([
            ('preprocessor', self.preprocessor),
            ('classifier', self.model)
        ])
        
        # Make predictions
        try:
            probabilities = pipeline.predict_proba(X)[:, 1]
            predictions = (probabilities >= self.threshold).astype(int)
            
            # Create result dataframe
            result_df = df.copy()
            result_df[f'sale_probability_{timeframe}'] = probabilities
            result_df[f'predicted_outcome_{timeframe}'] = predictions
            
            return result_df
            
        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            # Return original dataframe if prediction fails
            return df
    
    def evaluate(self, df: pd.DataFrame, target_col: str = "sold") -> Dict[str, Any]:
        """
        Evaluate model performance on new data.
        
        Args:
            df: DataFrame with lead data
            target_col: Column with actual outcomes
            
        Returns:
            Dictionary with evaluation metrics
        """
        if self.model is None or self.preprocessor is None:
            return {"error": "Model is not trained. Please train the model first."}
        
        if target_col not in df.columns:
            return {"error": f"Target column '{target_col}' not found in data"}
        
        # Prepare data
        X, y = self._prepare_data(df, target_col)
        
        if y is None or len(y) == 0:
            return {"error": "No target data available for evaluation"}
        
        # Create pipeline
        pipeline = Pipeline([
            ('preprocessor', self.preprocessor),
            ('classifier', self.model)
        ])
        
        # Make predictions
        try:
            y_pred_proba = pipeline.predict_proba(X)[:, 1]
            y_pred = (y_pred_proba >= self.threshold).astype(int)
            
            # Calculate metrics
            metrics = {
                "accuracy": accuracy_score(y, y_pred),
                "precision": precision_score(y, y_pred, zero_division=0),
                "recall": recall_score(y, y_pred, zero_division=0),
                "f1_score": f1_score(y, y_pred, zero_division=0),
                "roc_auc": roc_auc_score(y, y_pred_proba)
            }
            
            return {
                "status": "success",
                "metrics": metrics,
                "samples_evaluated": len(X)
            }
            
        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            return {"error": f"Error evaluating model: {str(e)}"}
    
    def optimize_threshold(self, df: pd.DataFrame, target_col: str = "sold", 
                          metric: str = "f1_score") -> float:
        """
        Find the optimal probability threshold for binary classification.
        
        Args:
            df: DataFrame with lead data
            target_col: Column with actual outcomes
            metric: Metric to optimize ("f1_score", "precision", "recall", or "accuracy")
            
        Returns:
            Optimal threshold value
        """
        if self.model is None or self.preprocessor is None:
            raise ValueError("Model is not trained. Please train the model first.")
        
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in data")
        
        # Prepare data
        X, y = self._prepare_data(df, target_col)
        
        if y is None or len(y) == 0:
            raise ValueError("No target data available for threshold optimization")
        
        # Create pipeline
        pipeline = Pipeline([
            ('preprocessor', self.preprocessor),
            ('classifier', self.model)
        ])
        
        # Make predictions
        y_pred_proba = pipeline.predict_proba(X)[:, 1]
        
        # Test different thresholds
        thresholds = np.arange(0.1, 1.0, 0.05)
        best_threshold = 0.5
        best_score = 0.0
        
        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)
            
            if metric == "precision":
                score = precision_score(y, y_pred, zero_division=0)
            elif metric == "recall":
                score = recall_score(y, y_pred, zero_division=0)
            elif metric == "accuracy":
                score = accuracy_score(y, y_pred)
            else:  # default to f1_score
                score = f1_score(y, y_pred, zero_division=0)
            
            if score > best_score:
                best_score = score
                best_threshold = threshold
        
        # Update model threshold
        self.threshold = best_threshold
        self.metadata["threshold"] = best_threshold
        self._save_model()
        
        return best_threshold
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model.
        
        Returns:
            Dictionary with model information
        """
        return {
            "model_type": self.model_type,
            "features": self.features,
            "threshold": self.threshold,
            "metrics": self.metadata.get("metrics", {}),
            "last_trained": self.metadata.get("last_trained"),
            "created_at": self.metadata.get("created_at"),
            "samples_trained": self.metadata.get("samples_trained"),
            "is_trained": self.model is not None and self.preprocessor is not None
        }


# Helper function to create training data from lead flow data
def prepare_training_data(df: pd.DataFrame, outcome_days: int = 30, 
                         sold_col: str = "sold_date") -> pd.DataFrame:
    """
    Prepare training data from lead flow data.
    
    Args:
        df: DataFrame with lead data
        outcome_days: Number of days to consider for outcome (14 or 30)
        sold_col: Column indicating sold date
        
    Returns:
        DataFrame ready for training
    """
    # Create a working copy
    work_df = df.copy()
    
    # Create target column based on outcome days
    if sold_col in work_df.columns and 'created_date' in work_df.columns:
        # Convert columns to datetime if they aren't already
        if not pd.api.types.is_datetime64_any_dtype(work_df[sold_col]):
            work_df[sold_col] = pd.to_datetime(work_df[sold_col], errors='coerce')
            
        if not pd.api.types.is_datetime64_any_dtype(work_df['created_date']):
            work_df['created_date'] = pd.to_datetime(work_df['created_date'], errors='coerce')
        
        # Calculate days to sell
        work_df['days_to_sell'] = (work_df[sold_col] - work_df['created_date']).dt.total_seconds() / (24 * 3600)
        
        # Create outcome target (1 if sold within outcome_days, 0 otherwise)
        work_df['sold'] = ((work_df['days_to_sell'] <= outcome_days) & 
                           (work_df['days_to_sell'] >= 0)).astype(int)
        
        # Fill missing values with 0 (not sold)
        work_df['sold'].fillna(0, inplace=True)
    else:
        # If required columns don't exist, create a dummy sold column
        work_df['sold'] = 0
    
    return work_df


def load_test_data() -> pd.DataFrame:
    """
    Load sample test data for lead outcome prediction.
    
    Returns:
        DataFrame with sample lead data
    """
    # Create a date range for test data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)
    
    # Sample lead sources
    sources = ["Website", "Autotrader", "CarGurus", "Facebook", "Referral", "Walk-in"]
    
    # Sample sales reps
    reps = ["John Smith", "Jane Doe", "Mike Johnson", "Sarah Williams", "Robert Brown"]
    
    # Sample vehicle models
    models = ["Sedan X", "SUV Pro", "Truck Max", "Compact Y", "Luxury Z"]
    
    # Generate sample data
    np.random.seed(42)  # For reproducibility
    
    num_leads = 200
    data = []
    
    for i in range(num_leads):
        # Generate lead ID
        lead_id = f"L{10000 + i}"
        
        # Generate created date
        days_ago = np.random.randint(0, 90)
        created = end_date - timedelta(days=days_ago)
        
        # Randomly select source, rep, and model
        source = np.random.choice(sources)
        rep = np.random.choice(reps)
        model = np.random.choice(models)
        
        # Determine contact delay (hours)
        contact_delay = np.random.exponential(scale=12)  # Average 12-hour delay
        
        # Determine if contacted and when
        contacted = created + timedelta(hours=contact_delay) if np.random.random() < 0.9 else None
        
        # Determine if sold and when (influenced by features)
        sold_probability = 0.0
        
        # Factors that influence sale probability
        # 1. Rep factor
        rep_factors = {
            "John Smith": 0.2,
            "Jane Doe": 0.25,
            "Mike Johnson": 0.15,
            "Sarah Williams": 0.3,
            "Robert Brown": 0.1
        }
        sold_probability += rep_factors.get(rep, 0.2)
        
        # 2. Source factor
        source_factors = {
            "Website": 0.15,
            "Autotrader": 0.25,
            "CarGurus": 0.2,
            "Facebook": 0.1,
            "Referral": 0.3,
            "Walk-in": 0.2
        }
        sold_probability += source_factors.get(source, 0.15)
        
        # 3. Model factor
        model_factors = {
            "Sedan X": 0.2,
            "SUV Pro": 0.25,
            "Truck Max": 0.15,
            "Compact Y": 0.1,
            "Luxury Z": 0.3
        }
        sold_probability += model_factors.get(model, 0.2)
        
        # 4. Contact delay factor (higher delay = lower probability)
        delay_factor = max(0, 0.2 - (contact_delay / 100))
        sold_probability += delay_factor
        
        # 5. Time of day factor
        hour_created = created.hour
        # Better conversion during business hours
        if 9 <= hour_created <= 17:
            sold_probability += 0.1
        
        # Normalize probability
        sold_probability = sold_probability / 4  # Divide by number of factors + normalization
        
        # Determine if sold based on probability
        is_sold = np.random.random() < sold_probability
        
        # Determine sold date if sold
        if is_sold:
            # Time to close distribution (most sales close within 14-30 days)
            days_to_close = np.random.lognormal(mean=3.0, sigma=0.7)
            sold_date = created + timedelta(days=float(days_to_close))
            
            # Cap at current date
            if sold_date > datetime.now():
                sold_date = None
                is_sold = False
        else:
            sold_date = None
        
        # Add to data
        data.append({
            "LeadID": lead_id,
            "LeadSource": source,
            "SalesRep": rep,
            "Model": model,
            "created_date": created,
            "contacted_date": contacted,
            "sold_date": sold_date,
            "contact_delay": contact_delay,
            "hour_created": created.hour,
            "day_created": created.weekday()
        })
    
    return pd.DataFrame(data)


def run_test():
    """Run a test of the lead outcome predictor with sample data."""
    # Load sample data
    df = load_test_data()
    
    # Prepare data for training
    train_df = prepare_training_data(df, outcome_days=30)
    
    # Create and train model
    predictor = LeadOutcomePredictor(model_type="random_forest")
    results = predictor.train(train_df, target_col="sold")
    
    # Make predictions
    pred_df = predictor.predict(df)
    
    # Print results
    print("Lead Outcome Prediction Test")
    print("===========================")
    print(f"\nTrained model with {results.get('samples_trained', 0)} samples")
    print("\nModel Metrics:")
    metrics = results.get('metrics', {})
    for metric, value in metrics.items():
        if metric != 'feature_importance':
            print(f"- {metric}: {value:.4f}")
    
    # Print feature importance if available
    if 'feature_importance' in metrics:
        print("\nFeature Importance:")
        importances = sorted(metrics['feature_importance'].items(), key=lambda x: x[1], reverse=True)
        for feature, importance in importances[:5]:  # Top 5 features
            print(f"- {feature}: {importance:.4f}")
    
    # Print sample predictions
    print("\nSample Predictions:")
    sample_pred = pred_df.sample(min(5, len(pred_df)))
    for _, row in sample_pred.iterrows():
        print(f"Lead {row['LeadID']} ({row['LeadSource']}, Rep: {row['SalesRep']})")
        print(f"  Sale Probability (30-day): {row.get('sale_probability_medium', 0):.2f}")
        print(f"  Predicted Outcome: {'Likely Sale' if row.get('predicted_outcome_medium', 0) == 1 else 'Unlikely Sale'}")
        print(f"  Actual Outcome: {'Sold' if not pd.isna(row['sold_date']) else 'Not Sold'}")
        print()
    
    return results


if __name__ == "__main__":
    run_test()