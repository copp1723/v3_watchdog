"""
Test script for validating business queries with the enhanced LLM prompt format.

Validates common automotive business questions and checks the formatting, accuracy,
and usefulness of the responses.
"""

import sys
import os
import json
import unittest
from datetime import datetime
import pandas as pd

# Add the project root to Python path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.insight_conversation_enhanced import ConversationManager
from src.insight_card_improved import EnhancedInsightOutputFormatter

class TestBusinessQueries(unittest.TestCase):
    """Test business queries with the enhanced prompt system."""
    
    def setUp(self):
        """Set up the test environment with sample data."""
        # Initialize conversation manager with mock mode enabled
        self.conv_manager = ConversationManager(use_mock=True)
        
        # Create sample data
        self.create_sample_data()
        
        # Test queries
        self.test_queries = [
            "Which sales rep closed the most deals last month?",
            "What was the average front gross by lead source?",
            "Which day had the highest volume of sales?",
            "What is the breakdown of vehicle types sold last month?",
            "Which finance products had the highest penetration rate?"
        ]
        
        # Required fields in responses
        self.required_fields = ["summary", "value_insights", "actionable_flags", "confidence"]
    
    def create_sample_data(self):
        """Create sample automotive sales data for testing."""
        # Create a sample DataFrame with automotive sales data
        data = {
            'SaleDate': pd.date_range(start='2023-01-01', periods=100, freq='D'),
            'SalesRepName': ['Alice Johnson', 'Bob Smith', 'Charlie Davis', 'Diana Martinez'] * 25,
            'VehicleType': ['SUV', 'Sedan', 'Truck', 'Compact', 'SUV'] * 20,
            'LeadSource': ['Internet', 'Walk-in', 'Referral', 'Phone', 'Return Customer'] * 20,
            'FrontGross': [round(1000 + 2000 * pd.np.random.random(), 2) for _ in range(100)],
            'BackGross': [round(500 + 1500 * pd.np.random.random(), 2) for _ in range(100)],
            'TotalGross': [0] * 100,  # Will be calculated
            'VIN': [f'VIN{i:05d}' for i in range(1, 101)],
            'Make': ['Toyota', 'Honda', 'Ford', 'Chevrolet', 'Nissan'] * 20,
            'Model': ['Camry', 'Civic', 'F-150', 'Silverado', 'Altima'] * 20,
            'Year': [2020, 2021, 2022, 2023] * 25,
            'ListPrice': [round(20000 + 30000 * pd.np.random.random(), 2) for _ in range(100)],
            'SalePrice': [round(18000 + 28000 * pd.np.random.random(), 2) for _ in range(100)],
            'Days_on_Lot': [int(pd.np.random.randint(1, 120)) for _ in range(100)],
            'FinanceProduct': ['Extended Warranty', 'GAP Insurance', 'Prepaid Maintenance', 'None', 'Multiple'] * 20,
            'FinanceIncome': [round(0 + 1200 * pd.np.random.random(), 2) for _ in range(100)]
        }
        
        # Calculate TotalGross
        for i in range(100):
            data['TotalGross'][i] = data['FrontGross'][i] + data['BackGross'][i]
        
        # Create DataFrame
        self.sample_data = pd.DataFrame(data)
        
        # Add to session state for testing
        import streamlit as st
        if 'validated_data' not in st.session_state:
            st.session_state.validated_data = self.sample_data
    
    def check_response_format(self, response):
        """Verify the response has the correct format and structure."""
        # Check that all required fields are present
        for field in self.required_fields:
            self.assertIn(field, response, f"Response missing required field: {field}")
        
        # Check that summary is a non-empty string
        self.assertIsInstance(response['summary'], str)
        self.assertTrue(len(response['summary']) > 0, "Summary is empty")
        
        # Check that value_insights is a non-empty list
        self.assertIsInstance(response['value_insights'], list)
        self.assertTrue(len(response['value_insights']) > 0, "No value insights provided")
        
        # Check that actionable_flags is a list (can be empty)
        self.assertIsInstance(response['actionable_flags'], list)
        
        # Check that confidence is a valid value
        self.assertIn(response['confidence'].lower(), ['high', 'medium', 'low'], 
                     f"Invalid confidence level: {response['confidence']}")
        
        # Check for markdown formatting presence in insights
        has_markdown = False
        for insight in response['value_insights']:
            if '**' in insight:  # Check for bold formatting
                has_markdown = True
                break
        
        # Only issue a warning, not a failure, if no markdown found
        if not has_markdown:
            print(f"WARNING: No markdown formatting found in value insights")
    
    def check_response_relevance(self, query, response):
        """Check that the response is relevant to the query."""
        # Look for query keywords in the response summary
        query_keywords = query.lower().split()
        query_keywords = [k for k in query_keywords if len(k) > 3]  # Filter out short words
        
        # Check if any keywords appear in the summary
        summary_has_keyword = False
        summary_lower = response['summary'].lower()
        for keyword in query_keywords:
            if keyword in summary_lower:
                summary_has_keyword = True
                break
        
        self.assertTrue(summary_has_keyword, 
                       f"Response summary doesn't seem relevant to the query: {query}")
    
    def test_queries(self):
        """Test each business query and validate responses."""
        print("\n\n=== Testing Business Queries ===\n")
        
        validation_context = {
            'data_shape': self.sample_data.shape,
            'columns': self.sample_data.columns.tolist(),
            'lead_source_breakdown': self.sample_data['LeadSource'].value_counts().to_dict()
        }
        
        for i, query in enumerate(self.test_queries):
            print(f"\nTest Query {i+1}: {query}")
            
            # Generate insight
            response = self.conv_manager.generate_insight(query, validation_context)
            
            # Print the response for manual inspection
            print(f"Response:\n{json.dumps(response, indent=2)}")
            
            # Check response format
            self.check_response_format(response)
            
            # Check response relevance
            self.check_response_relevance(query, response)
            
            print(f"Query {i+1} passed format and relevance checks.")
    
    def test_enhanced_formatter(self):
        """Test that the enhanced formatter correctly processes responses."""
        # Create a test response with metrics but no markdown
        test_response = {
            "summary": "Alice Johnson leads with 25 deals and $75,000 total gross profit.",
            "value_insights": [
                "Alice averaged $3000 per deal, 15% above team average.",
                "Her SUV sales were 60% of total deals with $45,000 in gross.",
                "Top lead source was internet at 40% of deals.",
            ],
            "actionable_flags": [
                "Examine Alice's internet lead handling for training others.",
                "Consider allowing Alice to mentor newer team members."
            ],
            "confidence": "high"
        }
        
        # Format with enhanced formatter
        formatter = EnhancedInsightOutputFormatter()
        formatted = formatter._apply_markdown_formatting(test_response)
        
        # Check that formatting was applied
        self.assertIn("**", formatted['summary'], "No markdown added to summary")
        
        markdown_count = 0
        for insight in formatted['value_insights']:
            if "**" in insight:
                markdown_count += 1
        
        self.assertTrue(markdown_count > 0, "No markdown added to any insights")
        print(f"Enhanced formatter successfully added markdown to {markdown_count} insights")

if __name__ == "__main__":
    unittest.main()
