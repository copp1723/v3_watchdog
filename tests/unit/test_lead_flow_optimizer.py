"""
Unit tests for the lead flow optimizer module.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
import tempfile

from src.validators.lead_flow_optimizer import (
    LeadFlowMetrics,
    LeadFlowOptimizer,
    prepare_lead_data,
    load_test_data
)

class TestLeadFlowMetrics(unittest.TestCase):
    """Test cases for the LeadFlowMetrics class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary file for metrics storage
        self.temp_dir = tempfile.TemporaryDirectory()
        self.metrics_path = os.path.join(self.temp_dir.name, "test_metrics.json")
        
        # Create LeadFlowMetrics instance
        self.metrics = LeadFlowMetrics(storage_path=self.metrics_path)
        
        # Create a test DataFrame
        self.create_test_data()
    
    def tearDown(self):
        """Clean up after tests."""
        # Remove temporary directory
        self.temp_dir.cleanup()
    
    def create_test_data(self):
        """Create a test DataFrame for lead flow analysis."""
        # Create sample data
        now = datetime.now()
        
        data = [
            # Normal progression lead
            {
                "LeadID": "L1001",
                "LeadSource": "Website",
                "SalesRep": "John Smith",
                "Model": "Sedan X",
                "created_date": now - timedelta(days=30),
                "contacted_date": now - timedelta(days=29),
                "appointment_date": now - timedelta(days=28),
                "test_drive_date": now - timedelta(days=27),
                "offer_date": now - timedelta(days=26),
                "sold_date": now - timedelta(days=25),
                "delivered_date": now - timedelta(days=20),
                "closed_date": now - timedelta(days=15)
            },
            # Slow progression lead
            {
                "LeadID": "L1002",
                "LeadSource": "AutoTrader",
                "SalesRep": "Jane Doe",
                "Model": "SUV Pro",
                "created_date": now - timedelta(days=45),
                "contacted_date": now - timedelta(days=42),
                "appointment_date": now - timedelta(days=35),
                "test_drive_date": now - timedelta(days=30),
                "offer_date": now - timedelta(days=20),
                "sold_date": now - timedelta(days=15),
                "delivered_date": now - timedelta(days=10),
                "closed_date": now - timedelta(days=5)
            },
            # Stalled lead
            {
                "LeadID": "L1003",
                "LeadSource": "Facebook",
                "SalesRep": "Mike Johnson",
                "Model": "Truck Max",
                "created_date": now - timedelta(days=60),
                "contacted_date": now - timedelta(days=58),
                "appointment_date": now - timedelta(days=50),
                "test_drive_date": None,
                "offer_date": None,
                "sold_date": None,
                "delivered_date": None,
                "closed_date": None
            },
            # Fast progression lead
            {
                "LeadID": "L1004",
                "LeadSource": "Walk-in",
                "SalesRep": "John Smith",
                "Model": "Luxury Z",
                "created_date": now - timedelta(days=20),
                "contacted_date": now - timedelta(days=20),  # Same day
                "appointment_date": now - timedelta(days=20),  # Same day
                "test_drive_date": now - timedelta(days=20),  # Same day
                "offer_date": now - timedelta(days=20),  # Same day
                "sold_date": now - timedelta(days=19),
                "delivered_date": now - timedelta(days=15),
                "closed_date": now - timedelta(days=10)
            },
            # Recently created lead
            {
                "LeadID": "L1005",
                "LeadSource": "CarGurus",
                "SalesRep": "Sarah Williams",
                "Model": "Sedan X",
                "created_date": now - timedelta(days=5),
                "contacted_date": now - timedelta(days=4),
                "appointment_date": now - timedelta(days=2),
                "test_drive_date": now - timedelta(days=1),
                "offer_date": None,
                "sold_date": None,
                "delivered_date": None,
                "closed_date": None
            }
        ]
        
        self.test_df = pd.DataFrame(data)
    
    def test_process_lead_data(self):
        """Test processing lead data and identifying metrics."""
        # Process the test data
        results = self.metrics.process_lead_data(self.test_df)
        
        # Check that results have expected sections
        self.assertIn("bottlenecks", results)
        self.assertIn("outliers", results)
        self.assertIn("stage_metrics", results)
        self.assertIn("rep_metrics", results)
        self.assertIn("source_metrics", results)
        self.assertIn("timestamp", results)
        
        # Check bottlenecks were detected
        if "time_created_to_contacted" in results["bottlenecks"]:
            self.assertIsNotNone(results["bottlenecks"]["time_created_to_contacted"]["average_days"])
        
        # Verify metrics were saved to file
        self.assertTrue(os.path.exists(self.metrics_path))
        
        # Load saved metrics and verify
        with open(self.metrics_path, 'r') as f:
            saved_metrics = json.load(f)
        
        self.assertIn("bottlenecks", saved_metrics)
        self.assertIn("outliers", saved_metrics)
        self.assertIn("stage_metrics", saved_metrics)
    
    def test_identify_bottlenecks(self):
        """Test identification of bottlenecks in lead flow."""
        # Process the test data
        self.metrics.process_lead_data(self.test_df)
        
        # Get bottlenecks
        bottlenecks = self.metrics.get_bottlenecks()
        
        # Verify bottlenecks data structure
        self.assertIsInstance(bottlenecks, dict)
        
        # At least one stage should have metrics
        if "time_created_to_contacted" in bottlenecks:
            stage_data = bottlenecks["time_created_to_contacted"]
            self.assertIn("average_days", stage_data)
            self.assertIn("median_days", stage_data)
            self.assertIn("threshold", stage_data)
            self.assertIn("is_bottleneck", stage_data)
    
    def test_identify_outliers(self):
        """Test identification of outlier leads."""
        # Process the test data
        self.metrics.process_lead_data(self.test_df)
        
        # Get outliers
        outliers = self.metrics.get_outliers()
        
        # Verify outliers data structure
        self.assertIsInstance(outliers, dict)
        
        # Check aged leads
        if "aged_leads" in outliers:
            aged_data = outliers["aged_leads"]
            self.assertIn("count", aged_data)
            self.assertIn("percentage", aged_data)
            self.assertIn("threshold_days", aged_data)
            
            # Our test data has at least one aged lead
            self.assertGreaterEqual(aged_data["count"], 1)
    
    def test_rep_performance(self):
        """Test calculating performance metrics by sales rep."""
        # Process the test data
        self.metrics.process_lead_data(self.test_df)
        
        # Get rep performance
        rep_metrics = self.metrics.get_rep_performance()
        
        # Verify rep metrics
        self.assertIsInstance(rep_metrics, dict)
        self.assertGreaterEqual(len(rep_metrics), 2)  # At least 2 reps
        
        # Verify specific rep data
        if "John Smith" in rep_metrics:
            rep_data = rep_metrics["John Smith"]
            self.assertIn("total_leads", rep_data)
            self.assertIn("closed_leads", rep_data)
            self.assertIn("conversion_rate", rep_data)
            
            # John Smith has 2 leads in our test data
            self.assertEqual(rep_data["total_leads"], 2)
    
    def test_source_performance(self):
        """Test calculating performance metrics by lead source."""
        # Process the test data
        self.metrics.process_lead_data(self.test_df)
        
        # Get source performance
        source_metrics = self.metrics.get_source_performance()
        
        # Verify source metrics
        self.assertIsInstance(source_metrics, dict)
        self.assertGreaterEqual(len(source_metrics), 3)  # At least 3 sources
        
        # Verify specific source data
        if "Website" in source_metrics:
            source_data = source_metrics["Website"]
            self.assertIn("total_leads", source_data)
            self.assertIn("closed_leads", source_data)
            self.assertIn("conversion_rate", source_data)
    
    def test_summary(self):
        """Test generating a summary of lead flow metrics."""
        # Process the test data
        self.metrics.process_lead_data(self.test_df)
        
        # Get summary
        summary = self.metrics.get_summary()
        
        # Verify summary structure
        self.assertIsInstance(summary, dict)
        self.assertIn("overall_conversion_rate", summary)
        self.assertIn("aged_leads_count", summary)
        self.assertIn("top_bottlenecks", summary)
        self.assertIn("best_performing_reps", summary)
        self.assertIn("best_performing_sources", summary)
        self.assertIn("last_updated", summary)


class TestLeadFlowOptimizer(unittest.TestCase):
    """Test cases for the LeadFlowOptimizer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary file for metrics storage
        self.temp_dir = tempfile.TemporaryDirectory()
        self.metrics_path = os.path.join(self.temp_dir.name, "test_optimizer.json")
        
        # Create LeadFlowOptimizer instance
        self.optimizer = LeadFlowOptimizer(storage_path=self.metrics_path)
        
        # Load test data
        self.test_df = load_test_data()
    
    def tearDown(self):
        """Clean up after tests."""
        # Remove temporary directory
        self.temp_dir.cleanup()
    
    def test_process_lead_data(self):
        """Test processing lead data with the optimizer."""
        # Process the test data
        results = self.optimizer.process_lead_data(self.test_df)
        
        # Check results format
        self.assertIsInstance(results, dict)
        self.assertIn("bottlenecks", results)
        self.assertIn("outliers", results)
        self.assertIn("stage_metrics", results)
    
    def test_recommendations(self):
        """Test generation of optimization recommendations."""
        # Process the test data
        self.optimizer.process_lead_data(self.test_df)
        
        # Generate recommendations
        recommendations = self.optimizer.generate_recommendations()
        
        # Check recommendations format
        self.assertIsInstance(recommendations, list)
        self.assertGreater(len(recommendations), 0)
        
        # Verify recommendation structure
        first_rec = recommendations[0]
        self.assertIn("type", first_rec)
        self.assertIn("priority", first_rec)
        self.assertIn("title", first_rec)
        self.assertIn("description", first_rec)
        self.assertIn("action", first_rec)
    
    def test_flag_aged_leads(self):
        """Test flagging aged leads."""
        # Process the test data
        self.optimizer.process_lead_data(self.test_df)
        
        # Flag aged leads
        aged_leads = self.optimizer.flag_aged_leads(threshold=30.0)
        
        # Check aged leads format
        self.assertIsInstance(aged_leads, dict)
        self.assertIn("count", aged_leads)
        self.assertIn("percentage", aged_leads)
        
        # With our test data, we should have some aged leads
        self.assertGreaterEqual(aged_leads.get("count", 0), 1)


class TestLeadDataUtilities(unittest.TestCase):
    """Test cases for lead data utility functions."""
    
    def test_prepare_lead_data(self):
        """Test preparing raw lead data for analysis."""
        # Create raw data with different column names
        now = datetime.now()
        raw_data = [
            {
                "ID": "L2001",
                "Source": "Website",
                "Rep": "John Doe",
                "CreatedDate": now - timedelta(days=10),
                "FirstContactDate": now - timedelta(days=9),
                "FirstAppointment": now - timedelta(days=8),
                "VehicleDemo": now - timedelta(days=7),
                "ProposalDate": now - timedelta(days=6),
                "SaleDate": now - timedelta(days=5)
            }
        ]
        raw_df = pd.DataFrame(raw_data)
        
        # Define column mappings
        mappings = {
            "CreatedDate": "created_date",
            "FirstContactDate": "contacted_date",
            "FirstAppointment": "appointment_date",
            "VehicleDemo": "test_drive_date",
            "ProposalDate": "offer_date",
            "SaleDate": "sold_date"
        }
        
        # Prepare the data
        prepared_df = prepare_lead_data(raw_df, mappings)
        
        # Verify the prepared data
        self.assertIn("LeadID", prepared_df.columns)
        self.assertIn("created_date", prepared_df.columns)
        self.assertIn("contacted_date", prepared_df.columns)
        self.assertIn("appointment_date", prepared_df.columns)
        self.assertIn("test_drive_date", prepared_df.columns)
        self.assertIn("offer_date", prepared_df.columns)
        self.assertIn("sold_date", prepared_df.columns)
        
        # Verify data transformation
        self.assertEqual(prepared_df.iloc[0]["LeadID"], "L2001")
    
    def test_load_test_data(self):
        """Test loading sample test data."""
        # Load the test data
        test_df = load_test_data()
        
        # Verify the test data
        self.assertIsInstance(test_df, pd.DataFrame)
        self.assertGreater(len(test_df), 0)
        self.assertIn("LeadID", test_df.columns)
        self.assertIn("LeadSource", test_df.columns)
        self.assertIn("SalesRep", test_df.columns)
        self.assertIn("Model", test_df.columns)
        self.assertIn("created_date", test_df.columns)


if __name__ == "__main__":
    unittest.main()