"""
Integration test for the lead flow optimizer and prediction system.
"""

import os
import pandas as pd
import json
import logging
import unittest
import tempfile
import shutil
from datetime import datetime, timedelta

from src.validators.lead_flow_optimizer import LeadFlowOptimizer, prepare_lead_data
from src.ml.lead_model import LeadOutcomePredictor, load_test_data, prepare_training_data
from src.notifications.escalation import AlertEscalationRouter

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_lead_flow_integration():
    """
    End-to-end test of the lead flow optimizer using CSV test data.
    """
    try:
        # Define paths
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        test_data_path = os.path.join(base_path, "assets", "test_lead_flow_data.csv")
        results_path = os.path.join(base_path, "assets", "lead_flow_results.json")
        
        # Load test data
        if not os.path.exists(test_data_path):
            logger.error(f"Test data file not found: {test_data_path}")
            return False
        
        df = pd.read_csv(test_data_path)
        logger.info(f"Loaded {len(df)} lead records from {test_data_path}")
        
        # Initialize the optimizer
        optimizer = LeadFlowOptimizer()
        
        # Process the data
        logger.info("Processing lead flow data...")
        results = optimizer.process_lead_data(df)
        
        # Get summary
        summary = optimizer.get_summary()
        
        # Get recommendations
        recommendations = optimizer.generate_recommendations()
        
        # Save results
        output = {
            "summary": summary,
            "recommendations": recommendations,
            "bottlenecks": optimizer.identify_bottlenecks(),
            "aged_leads": optimizer.flag_aged_leads(),
            "rep_performance": optimizer.get_rep_performance(),
            "source_performance": optimizer.get_source_performance(),
            "model_performance": optimizer.get_model_performance(),
            "timestamp": datetime.now().isoformat()
        }
        
        with open(results_path, 'w') as f:
            json.dump(output, f, indent=2, default=str)
        
        logger.info(f"Lead flow analysis results saved to {results_path}")
        
        # Log key statistics
        logger.info(f"Overall lead conversion rate: {summary.get('overall_conversion_rate', 0):.1f}%")
        logger.info(f"Aged leads count: {summary.get('aged_leads_count', 0)}")
        
        if "top_bottlenecks" in summary:
            logger.info("Top bottlenecks:")
            for bottleneck in summary["top_bottlenecks"]:
                logger.info(f"- {bottleneck.get('stage')}: {bottleneck.get('average_days', 0):.1f} days")
        
        if recommendations:
            logger.info("Top recommendations:")
            for i, rec in enumerate(recommendations[:3], 1):
                logger.info(f"{i}. {rec.get('title')}: {rec.get('action')}")
        
        return True
    
    except Exception as e:
        logger.error(f"Error in lead flow integration test: {e}")
        return False

class TestLeadFlowPredictionSystem(unittest.TestCase):
    """Integration tests for the Lead Flow Prediction System."""
    
    def setUp(self):
        """Set up test environment."""
        # Create temporary directory for test data
        self.test_dir = tempfile.mkdtemp()
        
        # Paths for model and config
        self.model_path = os.path.join(self.test_dir, "test_model.pkl")
        self.config_path = os.path.join(self.test_dir, "test_config.json")
        self.log_path = os.path.join(self.test_dir, "test_log.json")
        
        # Load test data
        self.test_data = load_test_data()
        
        # Prepare training data
        self.training_data = prepare_training_data(self.test_data, outcome_days=30)
        
        # Create predictor
        self.predictor = LeadOutcomePredictor(model_path=self.model_path)
        
        # Train the model
        self.predictor.train(self.training_data)
        
        # Create escalation router
        self.router = AlertEscalationRouter(
            config_path=self.config_path,
            log_path=self.log_path,
            predictor=self.predictor
        )
        
        # Create lead flow optimizer
        self.optimizer = LeadFlowOptimizer()
    
    def tearDown(self):
        """Clean up after tests."""
        # Shutdown router
        self.router.shutdown()
        
        # Remove temporary directory
        shutil.rmtree(self.test_dir)
    
    def test_end_to_end_prediction_flow(self):
        """Test the end-to-end lead flow with predictions and escalations."""
        # Process leads for probability prediction
        high_risk_leads = []
        processed_count = 0
        escalation_count = 0
        
        for _, row in self.test_data.iterrows():
            # Convert row to dictionary for processing
            lead_dict = row.to_dict()
            
            # Make sure lead_id is string
            if "LeadID" in lead_dict:
                lead_dict["lead_id"] = str(lead_dict["LeadID"])
            elif "lead_id" not in lead_dict:
                lead_dict["lead_id"] = f"LEAD_{processed_count}"
            
            # Process with predictor to get probabilities
            lead_df = pd.DataFrame([lead_dict])
            pred_df = self.predictor.predict(lead_df)
            
            # Extract probability
            if "sale_probability_medium" in pred_df.columns:
                prob = float(pred_df["sale_probability_medium"].iloc[0])
                lead_dict["sale_probability_medium"] = prob
                
                # Flag high-risk leads
                if prob < 0.2:
                    high_risk_leads.append(lead_dict)
            
            # Process lead for escalation
            result = self.router.process_lead(lead_dict)
            
            # Count escalations
            if result["status"] != "no_escalation":
                escalation_count += 1
            
            processed_count += 1
            
            # Only process a subset for efficiency
            if processed_count >= 10:
                break
        
        # Check that we have some high-risk leads
        self.assertTrue(len(high_risk_leads) > 0, "Should identify some high-risk leads")
        
        # Check that we have some escalations
        self.assertTrue(escalation_count > 0, "Should generate some escalations")
        
        # Get escalation stats
        stats = self.router.get_escalation_stats()
        
        # Check stats
        self.assertEqual(stats["total_escalations"], escalation_count)
    
    def test_critical_lead_escalation(self):
        """Test immediate escalation of a critical lead."""
        # Create a critical lead
        critical_lead = {
            "lead_id": "CRITICAL_TEST_LEAD",
            "rep": "Test Rep",
            "rep_email": "test.rep@example.com",
            "source": "Website",
            "vehicle": "Luxury Z",
            "created_date": datetime.now() - timedelta(days=5),
            "contacted_date": datetime.now() - timedelta(days=4),
            "sale_probability_medium": 0.05,  # Very low probability
            "vehicle_price": 65000  # High value
        }
        
        # Process with immediate escalation
        result = self.router.escalate_lead(critical_lead, immediate=True)
        
        # Check escalation result
        self.assertIn(result["status"], ["completed", "failed"])
        self.assertEqual(result["lead_id"], "CRITICAL_TEST_LEAD")
        self.assertEqual(result["level"], "critical")
        
        # Get recent escalations
        escalations = self.router.get_recent_escalations(1)
        self.assertEqual(len(escalations), 1)
        
        # Check escalation details
        escalation = escalations[0]
        self.assertEqual(escalation["lead_id"], "CRITICAL_TEST_LEAD")
        self.assertEqual(escalation["level"], "critical")

if __name__ == "__main__":
    # Run the original test function
    test_lead_flow_integration()
    
    # Run the new unittest class
    unittest.main()