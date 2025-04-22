"""
Unit tests for the Lead Outcome Predictor module.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import tempfile
import shutil

from src.ml.lead_model import LeadOutcomePredictor, prepare_training_data, load_test_data


class TestLeadOutcomePredictor(unittest.TestCase):
    """Test cases for LeadOutcomePredictor."""
    
    def setUp(self):
        """Set up test environment."""
        # Create temporary directory for test models
        self.test_dir = tempfile.mkdtemp()
        self.model_path = os.path.join(self.test_dir, "test_model.pkl")
        
        # Create test data
        self.test_data = load_test_data()
        self.training_data = prepare_training_data(self.test_data, outcome_days=30)
    
    def tearDown(self):
        """Clean up after tests."""
        # Remove temporary directory
        shutil.rmtree(self.test_dir)
    
    def test_model_initialization(self):
        """Test model initialization."""
        predictor = LeadOutcomePredictor(model_path=self.model_path)
        
        # Check default values
        self.assertEqual(predictor.model_path, self.model_path)
        self.assertIsNotNone(predictor.model)
        self.assertEqual(predictor.model_type, "random_forest")
        self.assertEqual(predictor.threshold, 0.5)
    
    def test_model_training(self):
        """Test model training."""
        predictor = LeadOutcomePredictor(model_path=self.model_path)
        
        # Train model
        result = predictor.train(self.training_data, target_col="sold")
        
        # Check result
        self.assertEqual(result["status"], "success")
        self.assertTrue("metrics" in result)
        self.assertTrue("samples_trained" in result)
        self.assertTrue("samples_tested" in result)
        
        # Check metrics
        metrics = result["metrics"]
        self.assertTrue(0 <= metrics["accuracy"] <= 1)
        self.assertTrue(0 <= metrics["precision"] <= 1)
        self.assertTrue(0 <= metrics["recall"] <= 1)
        self.assertTrue(0 <= metrics["f1_score"] <= 1)
        self.assertTrue(0 <= metrics["roc_auc"] <= 1)
    
    def test_model_prediction(self):
        """Test model prediction."""
        predictor = LeadOutcomePredictor(model_path=self.model_path)
        
        # Train model
        predictor.train(self.training_data, target_col="sold")
        
        # Make predictions
        pred_df = predictor.predict(self.test_data)
        
        # Check result columns
        self.assertTrue("sale_probability_medium" in pred_df.columns)
        self.assertTrue("predicted_outcome_medium" in pred_df.columns)
        
        # Check prediction values
        self.assertTrue((0 <= pred_df["sale_probability_medium"]).all())
        self.assertTrue((pred_df["sale_probability_medium"] <= 1).all())
        self.assertTrue(pred_df["predicted_outcome_medium"].isin([0, 1]).all())
    
    def test_threshold_optimization(self):
        """Test threshold optimization."""
        predictor = LeadOutcomePredictor(model_path=self.model_path)
        
        # Train model
        predictor.train(self.training_data, target_col="sold")
        
        # Original threshold
        original_threshold = predictor.threshold
        
        # Optimize threshold
        new_threshold = predictor.optimize_threshold(self.training_data, target_col="sold", metric="f1_score")
        
        # Check that threshold changed
        self.assertNotEqual(original_threshold, new_threshold)
        self.assertEqual(predictor.threshold, new_threshold)
        self.assertTrue(0 <= new_threshold <= 1)
    
    def test_model_evaluation(self):
        """Test model evaluation."""
        predictor = LeadOutcomePredictor(model_path=self.model_path)
        
        # Train model
        predictor.train(self.training_data, target_col="sold")
        
        # Evaluate on same data
        result = predictor.evaluate(self.training_data, target_col="sold")
        
        # Check result
        self.assertEqual(result["status"], "success")
        self.assertTrue("metrics" in result)
        self.assertTrue("samples_evaluated" in result)
        
        # Check metrics
        metrics = result["metrics"]
        self.assertTrue(0 <= metrics["accuracy"] <= 1)
        self.assertTrue(0 <= metrics["precision"] <= 1)
        self.assertTrue(0 <= metrics["recall"] <= 1)
        self.assertTrue(0 <= metrics["f1_score"] <= 1)
        self.assertTrue(0 <= metrics["roc_auc"] <= 1)
    
    def test_different_model_types(self):
        """Test different model types."""
        model_types = ["random_forest", "gradient_boosting", "logistic"]
        
        for model_type in model_types:
            # Create predictor with specific model type
            predictor = LeadOutcomePredictor(
                model_path=os.path.join(self.test_dir, f"{model_type}_model.pkl"),
                model_type=model_type
            )
            
            # Train model
            result = predictor.train(self.training_data, target_col="sold")
            
            # Check result
            self.assertEqual(result["status"], "success")
            self.assertTrue("metrics" in result)
    
    def test_prepare_training_data(self):
        """Test prepare_training_data function."""
        # Prepare data for 14-day outcome
        data_14 = prepare_training_data(self.test_data, outcome_days=14)
        
        # Prepare data for 30-day outcome
        data_30 = prepare_training_data(self.test_data, outcome_days=30)
        
        # Check that target column exists
        self.assertTrue("sold" in data_14.columns)
        self.assertTrue("sold" in data_30.columns)
        
        # Check that target column is binary
        self.assertTrue(data_14["sold"].isin([0, 1]).all())
        self.assertTrue(data_30["sold"].isin([0, 1]).all())
        
        # Check that 30-day outcome has more positive examples
        self.assertTrue(data_14["sold"].sum() <= data_30["sold"].sum())


if __name__ == "__main__":
    unittest.main()