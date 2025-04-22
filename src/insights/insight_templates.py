"""
Template management module for V3 Watchdog AI.

Provides functionality for loading, managing and applying insight templates.
"""

import os
import json
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
import streamlit as st
import re
from src.trend_analysis import (
    analyze_sales_trend, 
    analyze_gross_profit,
    analyze_lead_sources,
    analyze_inventory_health
)

@dataclass
class InsightTemplate:
    """Represents a pre-built insight template."""
    template_id: str
    name: str
    description: str
    required_columns: List[str]
    optional_columns: List[str]
    prompt_template: str
    expected_response_format: Dict[str, Any]
    example_response: Dict[str, Any]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'InsightTemplate':
        """Create a template from a dictionary."""
        return cls(
            template_id=data.get("template_id", ""),
            name=data.get("name", ""),
            description=data.get("description", ""),
            required_columns=data.get("required_columns", []),
            optional_columns=data.get("optional_columns", []),
            prompt_template=data.get("prompt_template", ""),
            expected_response_format=data.get("expected_response_format", {}),
            example_response=data.get("example_response", {})
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert template to a dictionary."""
        return {
            "template_id": self.template_id,
            "name": self.name,
            "description": self.description,
            "required_columns": self.required_columns,
            "optional_columns": self.optional_columns,
            "prompt_template": self.prompt_template,
            "expected_response_format": self.expected_response_format,
            "example_response": self.example_response
        }

    def is_applicable(self, df: pd.DataFrame) -> bool:
        """
        Check if this template can be applied to the given DataFrame.
        
        Args:
            df: DataFrame to check against template requirements
            
        Returns:
            True if all required columns are present, False otherwise
        """
        if df is None:
            return False
            
        # Check if all required columns exist in the DataFrame
        columns = df.columns.tolist()
        return all(col in columns for col in self.required_columns)

    def get_compatibility_score(self, df: pd.DataFrame) -> float:
        """
        Calculate a compatibility score between the template and DataFrame.
        
        Args:
            df: DataFrame to calculate compatibility with
            
        Returns:
            Score between 0 and 1, where 1 is perfect compatibility
        """
        if df is None:
            return 0.0
            
        columns = df.columns.tolist()
        
        # Check required columns (critical)
        required_matched = sum(1 for col in self.required_columns if col in columns)
        required_score = required_matched / len(self.required_columns) if self.required_columns else 1.0
        
        # If any required column is missing, compatibility is zero
        if required_score < 1.0:
            return 0.0
        
        # Check optional columns (bonus)
        optional_matched = sum(1 for col in self.optional_columns if col in columns)
        optional_score = optional_matched / len(self.optional_columns) if self.optional_columns else 0.0
        
        # Weight required columns more heavily than optional
        return 0.7 + (0.3 * optional_score)


class TemplateManager:
    """Manages loading and applying insight templates."""
    
    def __init__(self, templates_dir: str = None):
        """
        Initialize the template manager.
        
        Args:
            templates_dir: Directory containing template JSON files
        """
        self.templates_dir = templates_dir or os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                                         "docs", "insight_templates")
        self.templates = {}
        self.load_templates()
    
    def load_templates(self) -> None:
        """Load all templates from the templates directory."""
        if not os.path.exists(self.templates_dir):
            print(f"Templates directory not found: {self.templates_dir}")
            return
            
        for filename in os.listdir(self.templates_dir):
            if filename.endswith('.json'):
                try:
                    file_path = os.path.join(self.templates_dir, filename)
                    with open(file_path, 'r') as f:
                        template_data = json.load(f)
                        
                    template = InsightTemplate.from_dict(template_data)
                    self.templates[template.template_id] = template
                    print(f"Loaded template: {template.name}")
                except Exception as e:
                    print(f"Error loading template from {filename}: {str(e)}")
    
    def get_template(self, template_id: str) -> Optional[InsightTemplate]:
        """
        Get a template by its ID.
        
        Args:
            template_id: The template ID to retrieve
            
        Returns:
            The template if found, None otherwise
        """
        return self.templates.get(template_id)
    
    def get_all_templates(self) -> List[InsightTemplate]:
        """
        Get all available templates.
        
        Returns:
            List of all templates
        """
        return list(self.templates.values())
    
    def get_applicable_templates(self, df: pd.DataFrame) -> List[Tuple[InsightTemplate, float]]:
        """
        Get templates that can be applied to the given DataFrame, sorted by compatibility.
        
        Args:
            df: DataFrame to check templates against
            
        Returns:
            List of tuples containing (template, compatibility_score), sorted by score
        """
        if df is None:
            return []
            
        # Calculate compatibility scores for all templates
        scored_templates = []
        for template in self.templates.values():
            score = template.get_compatibility_score(df)
            if score > 0:  # Only include applicable templates
                scored_templates.append((template, score))
        
        # Sort by compatibility score (descending)
        return sorted(scored_templates, key=lambda x: x[1], reverse=True)
    
    def apply_template(self, template_id: str, df: pd.DataFrame) -> str:
        """
        Apply a template to generate a prompt for the given DataFrame.
        
        Args:
            template_id: ID of the template to apply
            df: DataFrame to analyze
            
        Returns:
            Generated prompt string or error message
        """
        template = self.get_template(template_id)
        if not template:
            return f"Template not found: {template_id}"
        
        if not template.is_applicable(df):
            missing_cols = [col for col in template.required_columns if col not in df.columns]
            return f"Cannot apply template: missing required columns {missing_cols}"
        
        # Apply the template's prompt
        prompt = template.prompt_template
        
        # Add relevant data summaries based on the DataFrame
        data_context = self._generate_data_context(df, template)
        
        # Combine prompt with data context
        final_prompt = f"""Based on the following dealership data, please provide insights:

{prompt}

{data_context}

Important:
1. Base your answer ONLY on the data provided above
2. Be specific and include numbers from the data
3. Format your response as a JSON object with the following structure:
   - summary: A concise summary of key findings
   - chart_data: Visualization data with type, data points and title
   - recommendation: Actionable recommendations based on the analysis
   - risk_flag: Boolean indicating if there are concerning trends
"""
        
        return final_prompt
    
    def _generate_data_context(self, df: pd.DataFrame, template: InsightTemplate) -> str:
        """
        Generate a data context summary based on the template and DataFrame.
        
        Args:
            df: DataFrame to analyze
            template: Template to use for context generation
            
        Returns:
            Formatted data context string
        """
        context_parts = []
        analysis_data = {}
        
        # Add basic dataset info
        context_parts.append(f"\nDataset Overview:")
        context_parts.append(f"- Total Records: {len(df)}")
        
        # Generate summaries based on template type
        if template.template_id == "sales_trend_analysis":
            # Use the advanced sales trend analysis
            sales_analysis = analyze_sales_trend(df)
            analysis_data = sales_analysis  # Store for chart data
            
            if 'error' not in sales_analysis:
                # Extract key insights
                context_parts.append("\nSales Trend Analysis:")
                context_parts.append(f"- Total Sales: {sales_analysis.get('total_sales', 'N/A')}")
                context_parts.append(f"- Average Monthly Sales: {sales_analysis.get('average_monthly', 'N/A'):.1f}")
                context_parts.append(f"- Trend Direction: {sales_analysis.get('trend_direction', 'N/A').title()}")
                
                # Add quarter-over-quarter and year-over-year if available
                qoq = sales_analysis.get('quarter_over_quarter_change')
                if qoq is not None:
                    direction = "increase" if qoq > 0 else "decrease"
                    context_parts.append(f"- Quarter-over-Quarter: {abs(qoq):.1f}% {direction}")
                
                yoy = sales_analysis.get('year_over_year_change')
                if yoy is not None:
                    direction = "increase" if yoy > 0 else "decrease"
                    context_parts.append(f"- Year-over-Year: {abs(yoy):.1f}% {direction}")
                
                # Add seasonality information if available
                if sales_analysis.get('has_seasonality'):
                    peak_periods = sales_analysis.get('peak_periods', [])
                    if peak_periods:
                        context_parts.append(f"- Peak Periods: {', '.join(peak_periods)}")
            else:
                # Fallback to simple summary if advanced analysis fails
                if 'Sale_Date' in df.columns:
                    try:
                        df['Sale_Date'] = pd.to_datetime(df['Sale_Date'])
                        df['Month'] = df['Sale_Date'].dt.to_period('M')
                        monthly_counts = df.groupby('Month').size()
                        
                        context_parts.append("\nSales by Month:")
                        for month, count in monthly_counts.items():
                            context_parts.append(f"- {month}: {count} sales")
                    except:
                        pass
        
        elif template.template_id == "gross_profit_analysis":
            # Use the advanced gross profit analysis
            gross_analysis = analyze_gross_profit(df)
            analysis_data = gross_analysis  # Store for chart data
            
            if 'error' not in gross_analysis:
                # Extract key insights
                context_parts.append("\nGross Profit Analysis:")
                context_parts.append(f"- Total Gross Profit: ${gross_analysis.get('total_gross', 0):.2f}")
                context_parts.append(f"- Average Gross: ${gross_analysis.get('average_gross', 0):.2f}")
                context_parts.append(f"- Median Gross: ${gross_analysis.get('median_gross', 0):.2f}")
                
                # Add negative gross info
                neg_count = gross_analysis.get('negative_gross_count', 0)
                neg_pct = gross_analysis.get('negative_gross_percentage', 0)
                context_parts.append(f"- Negative Gross Deals: {neg_count} ({neg_pct:.1f}%)")
                
                # Add trend if available
                time_trend = gross_analysis.get('time_trend', {})
                if time_trend and 'trend_direction' in time_trend:
                    context_parts.append(f"- Trend Direction: {time_trend.get('trend_direction', 'unknown').title()}")
                
                # Add vehicle breakdown if available
                vehicle_data = gross_analysis.get('vehicle_breakdown', [])
                if vehicle_data:
                    context_parts.append("\nGross Profit by Vehicle Type:")
                    # Sort by average gross (descending)
                    sorted_data = sorted(vehicle_data, key=lambda x: x.get('AvgGross', 0), reverse=True)
                    for item in sorted_data[:5]:  # Show top 5
                        context_parts.append(f"- {item.get('VehicleType', 'Unknown')}: ${item.get('AvgGross', 0):.2f} avg ({item.get('Count', 0)} units)")
            else:
                # Fallback to simple summary if advanced analysis fails
                if 'Gross_Profit' in df.columns:
                    try:
                        gross_avg = df['Gross_Profit'].mean()
                        gross_median = df['Gross_Profit'].median()
                        gross_min = df['Gross_Profit'].min()
                        gross_max = df['Gross_Profit'].max()
                        negative_gross_count = (df['Gross_Profit'] < 0).sum()
                        
                        context_parts.append("\nGross Profit Summary:")
                        context_parts.append(f"- Average Gross Profit: ${gross_avg:.2f}")
                        context_parts.append(f"- Median Gross Profit: ${gross_median:.2f}")
                        context_parts.append(f"- Range: ${gross_min:.2f} to ${gross_max:.2f}")
                        context_parts.append(f"- Negative Gross Count: {negative_gross_count} ({negative_gross_count/len(df)*100:.1f}%)")
                    except:
                        pass
        
        elif template.template_id == "lead_source_roi":
            # Use the advanced lead source analysis
            lead_analysis = analyze_lead_sources(df)
            analysis_data = lead_analysis  # Store for chart data
            
            if 'error' not in lead_analysis:
                # Extract key insights
                context_parts.append("\nLead Source Analysis:")
                context_parts.append(f"- Top Lead Source: {lead_analysis.get('top_source', 'Unknown')}")
                
                # Add lead source breakdown
                lead_summary = lead_analysis.get('lead_summary', [])
                if lead_summary:
                    context_parts.append("\nSales by Lead Source:")
                    for item in lead_summary:
                        source = item.get('source', 'Unknown')
                        count = item.get('count', 0)
                        pct = item.get('percentage', 0)
                        context_parts.append(f"- {source}: {count} sales ({pct:.1f}%)")
                
                # Add gross profit by lead source if available
                if lead_summary and 'avg_gross' in lead_summary[0]:
                    context_parts.append("\nAverage Gross by Lead Source:")
                    # Sort by average gross (descending)
                    sorted_data = sorted(lead_summary, key=lambda x: x.get('avg_gross', 0), reverse=True)
                    for item in sorted_data[:5]:  # Show top 5
                        source = item.get('source', 'Unknown')
                        avg_gross = item.get('avg_gross', 0)
                        context_parts.append(f"- {source}: ${avg_gross:.2f} average gross")
                
                # Add time trend if available
                time_trend = lead_analysis.get('time_trend', {})
                if time_trend and 'growth_rates' in time_trend:
                    growth_rates = time_trend.get('growth_rates', {})
                    if growth_rates:
                        context_parts.append("\nLead Source Growth Rates:")
                        for source, rate in growth_rates.items():
                            direction = "increase" if rate > 0 else "decrease"
                            context_parts.append(f"- {source}: {abs(rate):.1f}% {direction}")
            else:
                # Fallback to simple summary if advanced analysis fails
                if 'LeadSource' in df.columns:
                    try:
                        lead_counts = df['LeadSource'].value_counts()
                        
                        context_parts.append("\nSales by Lead Source:")
                        for source, count in lead_counts.items():
                            context_parts.append(f"- {source}: {count} sales ({count/len(df)*100:.1f}%)")
                    except:
                        pass
        
        elif template.template_id == "inventory_health":
            # Use the advanced inventory health analysis
            inventory_analysis = analyze_inventory_health(df)
            analysis_data = inventory_analysis  # Store for chart data
            
            if 'error' not in inventory_analysis:
                # Extract key insights
                context_parts.append("\nInventory Health Analysis:")
                context_parts.append(f"- Total Units: {inventory_analysis.get('total_units', 0)}")
                context_parts.append(f"- Average Days in Inventory: {inventory_analysis.get('average_days', 0):.1f}")
                context_parts.append(f"- Median Days in Inventory: {inventory_analysis.get('median_days', 0):.1f}")
                
                # Add aged inventory info
                aged = inventory_analysis.get('aged_inventory', 0)
                aged_pct = inventory_analysis.get('aged_percentage', 0)
                context_parts.append(f"- Units Over 90 Days: {aged} ({aged_pct:.1f}%)")
                
                # Add turn rate
                turn_rate = inventory_analysis.get('turn_rate', 0)
                context_parts.append(f"- Estimated Annual Turn Rate: {turn_rate:.1f}x")
                
                # Add age distribution
                age_buckets = inventory_analysis.get('age_buckets', {})
                age_percentages = inventory_analysis.get('age_percentages', {})
                if age_buckets and age_percentages:
                    context_parts.append("\nInventory Age Distribution:")
                    for bucket in ['<30 days', '30-60 days', '61-90 days', '>90 days']:
                        count = age_buckets.get(bucket, 0)
                        pct = age_percentages.get(bucket, 0)
                        context_parts.append(f"- {bucket}: {count} units ({pct:.1f}%)")
                
                # Add vehicle type breakdown if available
                vehicle_data = inventory_analysis.get('vehicle_breakdown', [])
                if vehicle_data:
                    context_parts.append("\nAverage Days in Inventory by Vehicle Type:")
                    # Sort by average days (descending)
                    sorted_data = sorted(vehicle_data, key=lambda x: x.get('AvgDays', 0), reverse=True)
                    for item in sorted_data[:5]:  # Show top 5
                        context_parts.append(f"- {item.get('VehicleType', 'Unknown')}: {item.get('AvgDays', 0):.1f} days")
            else:
                # Fallback to simple summary if advanced analysis fails
                if 'DaysInInventory' in df.columns:
                    try:
                        avg_days = df['DaysInInventory'].mean()
                        aged_90_plus = (df['DaysInInventory'] > 90).sum()
                        
                        context_parts.append("\nInventory Age Summary:")
                        context_parts.append(f"- Average Days in Inventory: {avg_days:.1f}")
                        context_parts.append(f"- Units Over 90 Days: {aged_90_plus} ({aged_90_plus/len(df)*100:.1f}%)")
                        
                        # Age buckets
                        age_buckets = {
                            '<30 days': (df['DaysInInventory'] < 30).sum(),
                            '30-60 days': ((df['DaysInInventory'] >= 30) & (df['DaysInInventory'] < 60)).sum(),
                            '61-90 days': ((df['DaysInInventory'] >= 60) & (df['DaysInInventory'] < 90)).sum(),
                            '>90 days': (df['DaysInInventory'] >= 90).sum()
                        }
                        
                        context_parts.append("\nInventory Age Distribution:")
                        for bucket, count in age_buckets.items():
                            context_parts.append(f"- {bucket}: {count} units ({count/len(df)*100:.1f}%)")
                    except:
                        pass
        
        elif template.template_id == "service_revenue_analysis":
            # For service, summarize revenue
            if 'ServiceTotal' in df.columns:
                try:
                    total_revenue = df['ServiceTotal'].sum()
                    avg_ro = df['ServiceTotal'].mean()
                    
                    context_parts.append("\nService Revenue Summary:")
                    context_parts.append(f"- Total Service Revenue: ${total_revenue:.2f}")
                    context_parts.append(f"- Average Repair Order: ${avg_ro:.2f}")
                    
                    # If service type exists, break down by type
                    if 'ServiceType' in df.columns:
                        type_summary = df.groupby('ServiceType')['ServiceTotal'].agg(['sum', 'mean'])
                        
                        context_parts.append("\nRevenue by Service Type:")
                        for service_type, row in type_summary.iterrows():
                            total = row['sum']
                            avg = row['mean']
                            context_parts.append(f"- {service_type}: ${total:.2f} total, ${avg:.2f} average")
                except:
                    pass
        
        # If we have chart data from the analysis, add it to the template
        if analysis_data and 'chart_data' in analysis_data and isinstance(analysis_data['chart_data'], dict):
            # Add chart data to the template context
            chart_data_str = json.dumps(analysis_data['chart_data'])
            context_parts.append(f"\nChart Data: {chart_data_str}")
        
        # Combine all parts
        return "\n".join(context_parts)


def render_template_selector(df: pd.DataFrame) -> Optional[str]:
    """
    Render a UI component for selecting templates.
    
    Args:
        df: DataFrame to check templates against
        
    Returns:
        Selected template ID or None if no selection was made
    """
    template_manager = TemplateManager()
    applicable_templates = template_manager.get_applicable_templates(df)
    
    if not applicable_templates:
        st.warning("No applicable templates found for this dataset.")
        return None
    
    st.subheader("📊 Select an Insight Template")
    
    # Group templates by category (inferred from template ID)
    categories = {}
    for template, score in applicable_templates:
        # Extract category from template ID (e.g., "sales_trend_analysis" -> "sales")
        category = template.template_id.split('_')[0].capitalize()
        if category not in categories:
            categories[category] = []
        categories[category].append((template, score))
    
    # Create tabs for each category
    if len(categories) > 1:
        tabs = st.tabs(list(categories.keys()))
        
        selected_template_id = None
        for i, (category, templates) in enumerate(categories.items()):
            with tabs[i]:
                for template, score in templates:
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        st.markdown(f"**{template.name}**")
                        st.caption(template.description)
                    with col2:
                        if st.button("Select", key=f"template_{template.template_id}"):
                            selected_template_id = template.template_id
        
        return selected_template_id
    else:
        # If only one category, display templates directly
        category = list(categories.keys())[0]
        templates = categories[category]
        
        for template, score in templates:
            col1, col2 = st.columns([4, 1])
            with col1:
                st.markdown(f"**{template.name}**")
                st.caption(template.description)
            with col2:
                if st.button("Select", key=f"template_{template.template_id}"):
                    return template.template_id
    
    return None


if __name__ == "__main__":
    # Example usage
    import streamlit as st
    
    st.set_page_config(page_title="Template Demo", layout="wide")
    st.title("Insight Template Demo")
    
    # Create sample data
    data = {
        'Sale_Date': pd.date_range(start='2023-01-01', periods=100, freq='D'),
        'VIN': [f'VIN{i}' for i in range(100)],
        'Gross_Profit': [1500 + 1000 * (i % 10) / 10 for i in range(100)],
        'LeadSource': ['Website', 'Walk-in', 'Referral', 'Third-party'] * 25
    }
    df = pd.DataFrame(data)
    
    # Display template selector
    template_id = render_template_selector(df)
    
    if template_id:
        st.success(f"Selected template: {template_id}")
        
        # Display the generated prompt
        template_manager = TemplateManager()
        prompt = template_manager.apply_template(template_id, df)
        
        st.subheader("Generated Prompt")
        st.text_area("Prompt", prompt, height=300)