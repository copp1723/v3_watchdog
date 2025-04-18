"""
Executive insight functions for Watchdog AI.

Provides high-value, actionable insights for automotive dealership data.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime, timedelta
import logging
import sentry_sdk
from .base_insight import InsightBase, ChartableInsight, find_column_by_pattern

# Configure logger
logger = logging.getLogger(__name__)

class MonthlyGrossMarginInsight(ChartableInsight):
    """
    Monthly Gross Margin vs. Target insight.
    
    Analyzes monthly gross margin trends compared to targets, 
    providing visualizations of performance over time.
    """
    
    def __init__(self):
        """Initialize the Monthly Gross Margin insight."""
        super().__init__("monthly_gross_margin")
    
    def _validate_insight_input(self, df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """Validate input data for monthly gross margin insight."""
        # Find date column
        date_col = find_column_by_pattern(df, ['date', 'saledate', 'sale_date', 'transaction_date'])
        if not date_col:
            return {"error": "No date column found in data"}
        
        # Find gross profit column
        gross_col = find_column_by_pattern(df, ['gross', 'profit', 'totalgross', 'total_gross', 'grossprofit'])
        if not gross_col:
            return {"error": "No gross profit column found in data"}
        
        # Find cost or sale price column (for margin calculation)
        price_col = find_column_by_pattern(df, ['price', 'sale_price', 'saleprice', 'amount', 'revenue'], False)
        cost_col = find_column_by_pattern(df, ['cost', 'vehicle_cost', 'vehiclecost', 'expense'], False)
        
        if not price_col and not cost_col and 'target_margin' not in kwargs:
            return {"error": "Cannot calculate margin - no price or cost column found and no target margin provided"}
        
        return {}
    
    def preprocess_data(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Preprocess data for monthly gross margin insight."""
        processed_df = df.copy()
        
        # Find date column
        date_col = find_column_by_pattern(processed_df, ['date', 'saledate', 'sale_date', 'transaction_date'])
        
        # Convert date column to datetime
        processed_df[date_col] = pd.to_datetime(processed_df[date_col], errors='coerce')
        
        # Drop rows with invalid dates
        processed_df = processed_df.dropna(subset=[date_col])
        
        # Find gross profit column
        gross_col = find_column_by_pattern(processed_df, ['gross', 'profit', 'totalgross', 'total_gross', 'grossprofit'])
        
        # Convert gross to numeric
        processed_df[gross_col] = pd.to_numeric(processed_df[gross_col].astype(str).str.replace(r'[$,]', '', regex=True), 
                                         errors='coerce')
        
        # Find price column if available
        price_col = find_column_by_pattern(processed_df, ['price', 'sale_price', 'saleprice', 'amount', 'revenue'], False)
        if price_col:
            processed_df[price_col] = pd.to_numeric(processed_df[price_col].astype(str).str.replace(r'[$,]', '', regex=True), 
                                            errors='coerce')
        
        # Find cost column if available
        cost_col = find_column_by_pattern(processed_df, ['cost', 'vehicle_cost', 'vehiclecost', 'expense'], False)
        if cost_col:
            processed_df[cost_col] = pd.to_numeric(processed_df[cost_col].astype(str).str.replace(r'[$,]', '', regex=True), 
                                           errors='coerce')
        
        # Drop rows with missing values in critical columns
        critical_cols = [date_col, gross_col]
        if price_col:
            critical_cols.append(price_col)
        if cost_col:
            critical_cols.append(cost_col)
        
        processed_df = processed_df.dropna(subset=critical_cols)
        
        return processed_df
    
    def compute_insight(self, df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """Compute monthly gross margin insight."""
        # Find column names
        date_col = find_column_by_pattern(df, ['date', 'saledate', 'sale_date', 'transaction_date'])
        gross_col = find_column_by_pattern(df, ['gross', 'profit', 'totalgross', 'total_gross', 'grossprofit'])
        price_col = find_column_by_pattern(df, ['price', 'sale_price', 'saleprice', 'amount', 'revenue'], False)
        cost_col = find_column_by_pattern(df, ['cost', 'vehicle_cost', 'vehiclecost', 'expense'], False)
        
        # Create month field
        df['month'] = df[date_col].dt.to_period('M')
        
        # Get target margin from kwargs or use default
        target_margin = kwargs.get('target_margin', 0.20)  # Default 20%
        
        # Calculate actual margin if we have price data
        if price_col:
            df['margin'] = df[gross_col] / df[price_col]
        # Or calculate based on cost if available
        elif cost_col:
            df['margin'] = df[gross_col] / (df[cost_col] + df[gross_col])
        # Otherwise use gross profit directly (can't calculate true margin)
        else:
            # Skip margin calculation
            pass
        
        # Group by month
        monthly_data = df.groupby('month').agg({
            gross_col: ['sum', 'count'],
        }).reset_index()
        
        # Flatten column names
        monthly_data.columns = ['month', 'total_gross', 'deal_count']
        
        # Calculate average gross per deal
        monthly_data['avg_gross_per_deal'] = monthly_data['total_gross'] / monthly_data['deal_count']
        
        # Calculate margin by month if possible
        if 'margin' in df.columns:
            margins = df.groupby('month')['margin'].mean().reset_index()
            monthly_data = monthly_data.merge(margins, on='month')
            
            # Calculate difference from target
            monthly_data['margin_delta'] = monthly_data['margin'] - target_margin
            monthly_data['margin_delta_pct'] = (monthly_data['margin'] / target_margin - 1) * 100
        
        # Convert period to string for serialization
        monthly_data['month_str'] = monthly_data['month'].astype(str)
        
        # Analyze trend
        trend_data = {}
        if len(monthly_data) >= 2:
            # Sort by month
            monthly_data = monthly_data.sort_values('month')
            
            # Get first and last months
            first_month = monthly_data.iloc[0]
            last_month = monthly_data.iloc[-1]
            
            # Calculate trend in gross profit
            gross_change = (last_month['total_gross'] - first_month['total_gross']) / first_month['total_gross']
            trend_data['gross_change_pct'] = float(gross_change * 100)
            trend_data['gross_trend'] = "increasing" if gross_change > 0.05 else "decreasing" if gross_change < -0.05 else "stable"
            
            # Calculate trend in margin if available
            if 'margin' in monthly_data.columns:
                margin_change = last_month['margin'] - first_month['margin']
                trend_data['margin_change_ppt'] = float(margin_change * 100)  # Percentage points
                trend_data['margin_trend'] = "improving" if margin_change > 0.01 else "declining" if margin_change < -0.01 else "stable"
        
        # Generate insights text
        insights = []
        
        # Current month performance
        if len(monthly_data) > 0:
            current = monthly_data.iloc[-1]
            current_month = current['month_str']
            
            if 'margin' in monthly_data.columns:
                current_margin = current['margin']
                target_diff = current_margin - target_margin
                
                insights.append({
                    "type": "current_performance",
                    "title": f"Gross Margin Performance - {current_month}",
                    "description": (
                        f"Current gross margin is {current_margin:.1%}, which is "
                        f"{'above' if target_diff > 0 else 'below'} the target of {target_margin:.1%} "
                        f"by {abs(target_diff):.1%}."
                    )
                })
            else:
                # No margin data available, report on gross profit
                insights.append({
                    "type": "current_performance",
                    "title": f"Gross Profit Performance - {current_month}",
                    "description": (
                        f"Current month has ${current['total_gross']:,.2f} in total gross profit "
                        f"from {current['deal_count']} deals, averaging ${current['avg_gross_per_deal']:,.2f} per deal."
                    )
                })
        
        # Trend insight
        if trend_data:
            if 'margin_trend' in trend_data:
                insights.append({
                    "type": "trend",
                    "title": "Gross Margin Trend",
                    "description": (
                        f"Gross margin is {trend_data['margin_trend']} with a {trend_data['margin_change_ppt']:.1f} "
                        f"percentage point change over the analyzed period. Total gross profit is "
                        f"{trend_data['gross_trend']} ({trend_data['gross_change_pct']:.1f}%)."
                    )
                })
            else:
                insights.append({
                    "type": "trend",
                    "title": "Gross Profit Trend",
                    "description": (
                        f"Total gross profit is {trend_data['gross_trend']} ({trend_data['gross_change_pct']:.1f}%) "
                        f"over the analyzed period."
                    )
                })
        
        # Generate recommendations
        recommendations = []
        
        # Add recommendation based on margin performance
        if 'margin' in monthly_data.columns:
            latest_margin = monthly_data.iloc[-1]['margin']
            if latest_margin < target_margin * 0.9:  # More than 10% below target
                recommendations.append(
                    "Gross margin is significantly below target. Review pricing strategy and consider "
                    "adjusting vehicle acquisition costs or sales tactics to improve margins."
                )
            elif latest_margin < target_margin:
                recommendations.append(
                    "Gross margin is below target. Consider focusing sales team on higher-margin vehicles "
                    "and reviewing discount approvals."
                )
        
        # Add recommendation based on trend
        if trend_data and trend_data.get('gross_trend') == 'decreasing':
            recommendations.append(
                "Gross profit shows a declining trend. Analyze changes in sales mix or negotiation "
                "patterns that might be affecting overall profitability."
            )
        
        # Seasonal recommendation if we have enough data
        if len(monthly_data) >= 6:
            recommendations.append(
                "Consider analyzing gross margin by vehicle model to identify highest and lowest "
                "performing inventory segments."
            )
        
        # Return complete insight
        return {
            "monthly_data": monthly_data.to_dict('records'),
            "trend_data": trend_data,
            "target_margin": float(target_margin),
            "insights": insights,
            "recommendations": recommendations,
            "data_rows": len(df)
        }
    
    def create_chart_data(self, insight_result: Dict[str, Any], original_df: pd.DataFrame, **kwargs) -> Optional[pd.DataFrame]:
        """Create chart data for monthly gross margin insight."""
        if "monthly_data" not in insight_result:
            return None
            
        # Convert records back to DataFrame
        monthly_data = pd.DataFrame(insight_result["monthly_data"])
        
        # Keep only needed columns for charting
        chart_columns = ['month_str', 'total_gross', 'avg_gross_per_deal']
        if 'margin' in monthly_data.columns:
            chart_columns.append('margin')
            # Add target margin as a column
            monthly_data['target_margin'] = insight_result.get('target_margin', 0.2)
            chart_columns.append('target_margin')
        
        chart_df = monthly_data[chart_columns].copy()
        
        # Sort by month for proper display
        try:
            # Try to convert back to period for proper sorting
            chart_df['month_period'] = pd.PeriodIndex(chart_df['month_str'], freq='M')
            chart_df = chart_df.sort_values('month_period')
            chart_df = chart_df.drop('month_period', axis=1)
        except:
            # Fallback if conversion fails
            pass
        
        return chart_df
    
    def create_chart_encoding(self, insight_result: Dict[str, Any], chart_data: pd.DataFrame, **kwargs) -> Optional[Dict[str, Any]]:
        """Create chart encoding for monthly gross margin insight."""
        encoding = {
            "chart_type": "line",
            "x": "month_str",
            "title": "Monthly Gross Performance"
        }
        
        # Determine primary y-axis based on data availability
        if 'margin' in chart_data.columns:
            encoding["y"] = "margin"
            encoding["y_title"] = "Gross Margin %"
            
            # Format margin as percentage if it's a decimal
            if chart_data['margin'].max() < 1:
                # This is a hint for the UI to format as percentage
                encoding["y_format"] = "percentage"
        else:
            encoding["y"] = "avg_gross_per_deal"
            encoding["y_title"] = "Average Gross per Deal"
            encoding["y_format"] = "currency"
        
        return encoding


class LeadConversionRateInsight(ChartableInsight):
    """
    Lead Conversion Rate by Source with Trend insight.
    
    Analyzes lead conversion rates across different sources,
    identifies trends, and provides actionable recommendations.
    """
    
    def __init__(self):
        """Initialize the Lead Conversion Rate insight."""
        super().__init__("lead_conversion_rate")
    
    def _validate_insight_input(self, df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """Validate input data for lead conversion rate insight."""
        # Find lead source column
        source_col = find_column_by_pattern(df, ['leadsource', 'lead_source', 'source'])
        if not source_col:
            return {"error": "No lead source column found in data"}
        
        # Find date column 
        date_col = find_column_by_pattern(df, ['date', 'saledate', 'sale_date', 'lead_date'], False)
        
        # Find conversion status column - this could be 'status', 'converted', 'sold', etc.
        # Or we might infer conversion from presence of a sale date or sale amount
        conversion_col = find_column_by_pattern(
            df, ['status', 'converted', 'sold', 'purchase', 'sale'], False
        )
        
        # If we can't find either a conversion column or date, we can't analyze conversion
        if not conversion_col and not date_col:
            return {"error": "No conversion status or date column found in data"}
        
        return {}
    
    def preprocess_data(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Preprocess data for lead conversion rate insight."""
        processed_df = df.copy()
        
        # Find columns
        source_col = find_column_by_pattern(processed_df, ['leadsource', 'lead_source', 'source'])
        date_col = find_column_by_pattern(processed_df, ['date', 'saledate', 'sale_date', 'lead_date'], False)
        conversion_col = find_column_by_pattern(
            processed_df, ['status', 'converted', 'sold', 'purchase', 'sale'], False
        )
        
        # Normalize lead source values
        processed_df[source_col] = processed_df[source_col].str.strip()
        processed_df[source_col] = processed_df[source_col].fillna('Unknown')
        
        # Attempt to create consistent source categories by normalizing common variations
        processed_df[source_col] = processed_df[source_col].str.lower()
        
        # Replace common variations with standardized values
        source_mapping = {
            'web': 'Website',
            'website': 'Website',
            'dealership website': 'Website',
            'dealer website': 'Website',
            'walk in': 'Walk-In',
            'walkin': 'Walk-In',
            'walk-in': 'Walk-In',
            'facebook': 'Facebook',
            'fb': 'Facebook',
            'instagram': 'Social Media',
            'twitter': 'Social Media',
            'social': 'Social Media',
            'google': 'Google',
            'google ads': 'Google',
            'google adwords': 'Google',
            'autotrader': 'AutoTrader',
            'auto trader': 'AutoTrader',
            'cars.com': 'Cars.com',
            'carscom': 'Cars.com',
            'cars': 'Cars.com',
            'craigslist': 'Craigslist',
            'referral': 'Referral',
            'refer': 'Referral',
            'friend': 'Referral',
            'family': 'Referral',
            'repeat': 'Repeat Customer',
            'repeat customer': 'Repeat Customer',
            'return': 'Repeat Customer',
            'phone': 'Phone Call',
            'call': 'Phone Call',
            'phone call': 'Phone Call',
            'email': 'Email',
            'mail': 'Email',
            'direct mail': 'Direct Mail',
            'carfax': 'Carfax',
            'cargurus': 'CarGurus',
            'car gurus': 'CarGurus',
            'truecar': 'TrueCar',
            'true car': 'TrueCar',
        }
        
        # Apply mapping
        for k, v in source_mapping.items():
            mask = processed_df[source_col].str.contains(k, case=False, na=False)
            processed_df.loc[mask, source_col] = v
        
        # Convert date column to datetime if it exists
        if date_col:
            processed_df[date_col] = pd.to_datetime(processed_df[date_col], errors='coerce')
        
        # Determine conversion status
        if conversion_col:
            # If conversion column exists, ensure it's properly formatted
            # Check if it's already boolean
            if processed_df[conversion_col].dtype == bool:
                processed_df['converted'] = processed_df[conversion_col]
            else:
                # Check if it's a status field
                status_values = processed_df[conversion_col].dropna().unique()
                
                # Check for common "success" status indicators
                success_terms = ['sold', 'purchase', 'deal', 'close', 'won', 'success', 'converted', 'yes']
                
                # Find status values that indicate conversion
                conversion_statuses = [
                    status for status in status_values
                    if any(term in str(status).lower() for term in success_terms)
                ]
                
                # Create converted flag
                processed_df['converted'] = processed_df[conversion_col].isin(conversion_statuses)
        else:
            # Infer conversion from sale date or price - if present and not NA, consider it converted
            potential_indicators = [
                col for col in processed_df.columns 
                if any(term in col.lower() for term in ['saleprice', 'sale_price', 'sold_price', 'soldprice'])
            ]
            
            if potential_indicators:
                indicator_col = potential_indicators[0]
                processed_df['converted'] = ~processed_df[indicator_col].isna()
            elif date_col:
                # Use date as a proxy - if date exists, consider it converted
                processed_df['converted'] = ~processed_df[date_col].isna()
            else:
                # We can't determine conversion status
                return pd.DataFrame()
        
        # Drop rows with missing source
        processed_df = processed_df.dropna(subset=[source_col])
        
        # Create month field if date exists
        if date_col:
            processed_df['month'] = processed_df[date_col].dt.to_period('M')
        
        return processed_df
    
    def compute_insight(self, df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """Compute lead conversion rate insight."""
        # Find source column
        source_col = find_column_by_pattern(df, ['leadsource', 'lead_source', 'source'])
        
        # Check if we have month data for trend analysis
        has_trend_data = 'month' in df.columns
        
        # Calculate overall conversion rate
        total_leads = len(df)
        
        # Ensure 'converted' is treated as boolean
        if 'converted' in df.columns:
            if df['converted'].dtype != bool:
                df['converted'] = df['converted'].astype(bool)
            converted_leads = int(df['converted'].sum())
        else:
            converted_leads = 0
            
        conversion_rate = converted_leads / total_leads if total_leads > 0 else 0
        
        # Calculate conversion rate by source
        source_conversion = df.groupby(source_col).agg({
            'converted': ['sum', 'count']
        }).reset_index()
        
        # Flatten columns
        source_conversion.columns = [source_col, 'converted_count', 'total_count']
        
        # Calculate conversion rate
        source_conversion['conversion_rate'] = source_conversion['converted_count'] / source_conversion['total_count']
        
        # Calculate volume percentage
        source_conversion['volume_percentage'] = source_conversion['total_count'] / total_leads * 100
        
        # Sort by conversion rate (descending)
        source_conversion = source_conversion.sort_values('conversion_rate', ascending=False)
        
        # Analyze trend data if available
        trend_data = {}
        monthly_conversion = None
        
        if has_trend_data:
            # Calculate conversion rate by month
            monthly_conversion = df.groupby('month').agg({
                'converted': ['sum', 'count']
            }).reset_index()
            
            # Flatten columns
            monthly_conversion.columns = ['month', 'converted_count', 'total_count']
            
            # Calculate conversion rate
            monthly_conversion['conversion_rate'] = monthly_conversion['converted_count'] / monthly_conversion['total_count']
            
            # Convert period to string for serialization
            monthly_conversion['month_str'] = monthly_conversion['month'].astype(str)
            
            # Sort by month
            monthly_conversion = monthly_conversion.sort_values('month')
            
            # Calculate trend if we have at least 2 months
            if len(monthly_conversion) >= 2:
                first_month = monthly_conversion.iloc[0]
                last_month = monthly_conversion.iloc[-1]
                
                # Calculate change in conversion rate (percentage points)
                rate_change_ppt = last_month['conversion_rate'] - first_month['conversion_rate']
                
                # Calculate relative change (percent)
                baseline = first_month['conversion_rate']
                if baseline > 0:
                    rate_change_pct = (last_month['conversion_rate'] / baseline - 1) * 100
                else:
                    rate_change_pct = 0  # Handle division by zero
                
                trend_data = {
                    'rate_change_ppt': float(rate_change_ppt * 100),  # Convert to percentage points
                    'rate_change_pct': float(rate_change_pct),
                    'trend_direction': 'improving' if rate_change_ppt > 0.01 else 'declining' if rate_change_ppt < -0.01 else 'stable'
                }
        
        # Calculate source-specific trends if available
        source_trends = []
        if has_trend_data and len(df[source_col].unique()) > 1:
            top_sources = source_conversion.head(5)[source_col].tolist()
            
            for source in top_sources:
                source_df = df[df[source_col] == source]
                
                # Only analyze if we have enough data
                if len(source_df) < 10:
                    continue
                
                # Calculate conversion rate by month for this source
                source_monthly = source_df.groupby('month').agg({
                    'converted': ['sum', 'count']
                }).reset_index()
                
                # Skip if less than 2 months of data
                if len(source_monthly) < 2:
                    continue
                
                # Flatten columns
                source_monthly.columns = ['month', 'converted_count', 'total_count']
                
                # Calculate conversion rate
                source_monthly['conversion_rate'] = source_monthly['converted_count'] / source_monthly['total_count']
                
                # Sort by month
                source_monthly = source_monthly.sort_values('month')
                
                # Calculate trend
                first_month = source_monthly.iloc[0]
                last_month = source_monthly.iloc[-1]
                
                # Calculate change in conversion rate (percentage points)
                rate_change_ppt = last_month['conversion_rate'] - first_month['conversion_rate']
                
                source_trends.append({
                    'source': source,
                    'rate_change_ppt': float(rate_change_ppt * 100),  # Convert to percentage points
                    'current_rate': float(last_month['conversion_rate'] * 100),  # Convert to percentage
                    'trend_direction': 'improving' if rate_change_ppt > 0.01 else 'declining' if rate_change_ppt < -0.01 else 'stable'
                })
        
        # Generate insights
        insights = []
        
        # Overall conversion rate insight
        insights.append({
            "type": "overall_conversion",
            "title": "Overall Lead Conversion Rate",
            "description": (
                f"Overall lead conversion rate is {conversion_rate:.1%} "
                f"({converted_leads:,} conversions from {total_leads:,} leads)."
            )
        })
        
        # Top and bottom sources insight
        if len(source_conversion) >= 2:
            top_source = source_conversion.iloc[0]
            bottom_source = source_conversion.iloc[-1]
            
            insights.append({
                "type": "source_comparison",
                "title": "Lead Source Performance",
                "description": (
                    f"{top_source[source_col]} has the highest conversion rate at {top_source['conversion_rate']:.1%}, "
                    f"while {bottom_source[source_col]} has the lowest at {bottom_source['conversion_rate']:.1%}."
                )
            })
        
        # Trend insight
        if trend_data:
            insights.append({
                "type": "trend",
                "title": "Conversion Rate Trend",
                "description": (
                    f"Lead conversion rate is {trend_data['trend_direction']} with a "
                    f"{trend_data['rate_change_ppt']:.1f} percentage point change over the analyzed period "
                    f"({trend_data['rate_change_pct']:.1f}% relative change)."
                )
            })
        
        # Source-specific trend insights
        if source_trends:
            # Find the most interesting source trend (largest change)
            interesting_trends = sorted(source_trends, key=lambda x: abs(x['rate_change_ppt']), reverse=True)
            
            if interesting_trends:
                trend = interesting_trends[0]
                insights.append({
                    "type": "source_trend",
                    "title": f"{trend['source']} Conversion Trend",
                    "description": (
                        f"{trend['source']} conversion rate is {trend['trend_direction']}, "
                        f"with a {abs(trend['rate_change_ppt']):.1f} percentage point "
                        f"{'increase' if trend['rate_change_ppt'] > 0 else 'decrease'} over time. "
                        f"Current rate is {trend['current_rate']:.1f}%."
                    )
                })
        
        # Generate recommendations
        recommendations = []
        
        # Add recommendation based on source performance
        if len(source_conversion) >= 3:
            low_performers = source_conversion.iloc[-3:][source_col].tolist()
            low_performer_list = ", ".join(low_performers)
            
            recommendations.append(
                f"Focus on improving lead quality or follow-up processes for low-converting sources: {low_performer_list}."
            )
        
        # Add recommendation based on trend
        if trend_data and trend_data.get('trend_direction') == 'declining':
            recommendations.append(
                "Overall conversion rate is declining. Review lead follow-up process and sales training to address this trend."
            )
        
        # Add source-specific recommendation if available
        if source_trends:
            declining_sources = [s for s in source_trends if s['trend_direction'] == 'declining']
            if declining_sources:
                # Focus on the highest volume declining source
                source_volumes = {s: df[df[source_col] == s].shape[0] for s in [d['source'] for d in declining_sources]}
                highest_volume = max(source_volumes.items(), key=lambda x: x[1])[0]
                
                recommendations.append(
                    f"Investigate decline in {highest_volume} lead conversions. Consider reviewing lead quality, "
                    f"response time, and follow-up consistency."
                )
        
        # Add general recommendation for improvement
        top_volume_sources = source_conversion.sort_values('total_count', ascending=False).head(3)[source_col].tolist()
        top_source_list = ", ".join(top_volume_sources)
        
        recommendations.append(
            f"Implement A/B testing for lead follow-up strategies on your highest volume sources: {top_source_list}."
        )
        
        # Return complete insight
        result = {
            "overall_conversion_rate": float(conversion_rate),
            "total_leads": int(total_leads),
            "converted_leads": int(converted_leads),
            "source_data": source_conversion.to_dict('records'),
            "insights": insights,
            "recommendations": recommendations,
            "data_rows": len(df)
        }
        
        # Add trend data if available
        if trend_data:
            result["trend_data"] = trend_data
        
        if monthly_conversion is not None:
            result["monthly_data"] = monthly_conversion.to_dict('records')
            
        if source_trends:
            result["source_trends"] = source_trends
        
        return result
    
    def create_chart_data(self, insight_result: Dict[str, Any], original_df: pd.DataFrame, **kwargs) -> Optional[pd.DataFrame]:
        """Create chart data for lead conversion rate insight."""
        # Primary chart: conversion rate by source
        if "source_data" in insight_result:
            # Convert source data to DataFrame
            source_data = pd.DataFrame(insight_result["source_data"])
            
            # Find the column name that contains the lead source
            source_col = [col for col in source_data.columns if 'source' in col.lower()][0]
            
            # Keep only needed columns and rename for clarity
            chart_df = source_data[[source_col, 'conversion_rate', 'total_count']].copy()
            chart_df['conversion_rate_pct'] = chart_df['conversion_rate'] * 100
            
            # Sort by conversion rate (descending)
            chart_df = chart_df.sort_values('conversion_rate', ascending=False)
            
            # Limit to top sources for readability
            return chart_df.head(10)
        
        # Fallback to monthly trend if available
        elif "monthly_data" in insight_result:
            monthly_data = pd.DataFrame(insight_result["monthly_data"])
            
            # Sort by month for proper display
            try:
                # Try to convert back to period for proper sorting
                monthly_data['month_period'] = pd.PeriodIndex(monthly_data['month_str'], freq='M')
                monthly_data = monthly_data.sort_values('month_period')
                monthly_data = monthly_data.drop('month_period', axis=1)
            except:
                # Fallback if conversion fails
                pass
            
            # Add percentage column
            monthly_data['conversion_rate_pct'] = monthly_data['conversion_rate'] * 100
            
            return monthly_data
        
        return None
    
    def create_chart_encoding(self, insight_result: Dict[str, Any], chart_data: pd.DataFrame, **kwargs) -> Optional[Dict[str, Any]]:
        """Create chart encoding for lead conversion rate insight."""
        # Determine which chart we're creating based on the data
        source_col = next((col for col in chart_data.columns if 'source' in col.lower()), None)
        if source_col:
            # Source-based chart
            encoding = {
                "chart_type": "bar",
                "x": source_col,
                "y": "conversion_rate_pct",
                "y_title": "Conversion Rate (%)",
                "title": "Lead Conversion Rate by Source"
            }
        elif 'month_str' in chart_data.columns:
            # Time series chart
            encoding = {
                "chart_type": "line",
                "x": "month_str",
                "y": "conversion_rate_pct",
                "y_title": "Conversion Rate (%)",
                "title": "Lead Conversion Rate Over Time"
            }
        else:
            # Default encoding
            encoding = {
                "chart_type": "bar",
                "title": "Lead Conversion Analysis"
            }
        
        return encoding