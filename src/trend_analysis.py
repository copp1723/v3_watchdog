"""
Trend Analysis Module for V3 Watchdog AI.

Provides functionality for analyzing trends in time series data,
identifying patterns, and generating insights from automotive retail data.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import json

def analyze_time_series(df: pd.DataFrame, date_col: str, value_col: str, 
                       aggregation: str = 'mean', freq: str = 'M') -> pd.DataFrame:
    """
    Analyze a time series column in a DataFrame.
    
    Args:
        df: Input DataFrame
        date_col: Name of the date column
        value_col: Name of the value column to analyze
        aggregation: Aggregation method ('mean', 'sum', 'count', etc.)
        freq: Frequency for resampling ('D' for daily, 'W' for weekly, 'M' for monthly, etc.)
        
    Returns:
        DataFrame with aggregated time series data
    """
    if df is None or date_col not in df.columns or value_col not in df.columns:
        return pd.DataFrame()
    
    # Ensure date column is datetime type
    try:
        df[date_col] = pd.to_datetime(df[date_col])
    except:
        print(f"Error converting {date_col} to datetime")
        return pd.DataFrame()
    
    # Set date as index
    df_ts = df.set_index(date_col)
    
    # Resample and aggregate
    if aggregation == 'mean':
        result = df_ts[value_col].resample(freq).mean()
    elif aggregation == 'sum':
        result = df_ts[value_col].resample(freq).sum()
    elif aggregation == 'count':
        result = df_ts[value_col].resample(freq).count()
    else:
        result = df_ts[value_col].resample(freq).mean()  # Default to mean
    
    # Convert back to DataFrame with reset index
    result_df = result.reset_index()
    result_df.columns = [date_col, value_col]
    
    return result_df

def calculate_change_metrics(series: pd.Series) -> Dict[str, float]:
    """
    Calculate change metrics for a time series.
    
    Args:
        series: Time series data
        
    Returns:
        Dictionary of change metrics
    """
    if series.empty or len(series) < 2:
        return {
            'absolute_change': None,
            'percentage_change': None,
            'average': None,
            'trend_direction': 'insufficient_data'
        }
    
    # Calculate metrics
    first_value = series.iloc[0]
    last_value = series.iloc[-1]
    absolute_change = last_value - first_value
    
    # Avoid division by zero
    if first_value == 0:
        percentage_change = float('inf') if absolute_change > 0 else float('-inf')
    else:
        percentage_change = (absolute_change / first_value) * 100
    
    average = series.mean()
    
    # Determine trend direction
    if percentage_change > 5:
        trend_direction = 'increasing'
    elif percentage_change < -5:
        trend_direction = 'decreasing'
    else:
        trend_direction = 'stable'
    
    return {
        'absolute_change': absolute_change,
        'percentage_change': percentage_change,
        'average': average,
        'trend_direction': trend_direction
    }

def detect_seasonality(df: pd.DataFrame, date_col: str, value_col: str, freq: str = 'M') -> Dict[str, Any]:
    """
    Detect seasonal patterns in time series data.
    
    Args:
        df: Input DataFrame
        date_col: Name of the date column
        value_col: Name of the value column to analyze
        freq: Frequency for seasonality detection
        
    Returns:
        Dictionary with seasonality information
    """
    if df is None or date_col not in df.columns or value_col not in df.columns:
        return {'has_seasonality': False, 'pattern': None, 'peak_periods': []}
    
    try:
        # Ensure date column is datetime type
        df[date_col] = pd.to_datetime(df[date_col])
        
        # Extract relevant time components based on frequency
        if freq == 'D':
            # Daily data - look for day of week patterns
            df['day_of_week'] = df[date_col].dt.day_name()
            grouped = df.groupby('day_of_week')[value_col].mean()
            
            # Sort by day of week
            days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            grouped = grouped.reindex(days_order)
            
        elif freq == 'M':
            # Monthly data - look for month of year patterns
            df['month'] = df[date_col].dt.month_name()
            grouped = df.groupby('month')[value_col].mean()
            
            # Sort by month
            months_order = ['January', 'February', 'March', 'April', 'May', 'June', 
                           'July', 'August', 'September', 'October', 'November', 'December']
            grouped = grouped.reindex(months_order)
            
        elif freq == 'Q':
            # Quarterly data
            df['quarter'] = 'Q' + df[date_col].dt.quarter.astype(str)
            grouped = df.groupby('quarter')[value_col].mean()
        else:
            # Default to monthly
            df['month'] = df[date_col].dt.month_name()
            grouped = df.groupby('month')[value_col].mean()
        
        # Detect seasonality by comparing std to mean
        std_to_mean_ratio = grouped.std() / grouped.mean()
        has_seasonality = std_to_mean_ratio > 0.1
        
        # Find peak periods (top 25%)
        threshold = grouped.quantile(0.75)
        peak_periods = grouped[grouped >= threshold].index.tolist()
        
        return {
            'has_seasonality': bool(has_seasonality),
            'pattern': grouped.to_dict(),
            'peak_periods': peak_periods
        }
        
    except Exception as e:
        print(f"Error detecting seasonality: {e}")
        return {'has_seasonality': False, 'pattern': None, 'peak_periods': []}

def analyze_sales_trend(df: pd.DataFrame, date_col: str = 'Sale_Date') -> Dict[str, Any]:
    """
    Comprehensive sales trend analysis.
    
    Args:
        df: Input DataFrame with sales data
        date_col: Name of the date column
        
    Returns:
        Dictionary with sales trend analysis results
    """
    if df is None or date_col not in df.columns:
        return {'error': 'Missing required data'}
    
    try:
        # Ensure date column is datetime type
        df[date_col] = pd.to_datetime(df[date_col])
        
        # Count sales by day
        df['day'] = df[date_col].dt.date
        daily_sales = df.groupby('day').size().reset_index(name='count')
        daily_sales['day'] = pd.to_datetime(daily_sales['day'])
        
        # Monthly aggregation
        monthly_sales = analyze_time_series(
            daily_sales, 'day', 'count', aggregation='sum', freq='M'
        )
        
        # Calculate metrics for the last 3 months vs previous 3 months (if data allows)
        if len(monthly_sales) >= 6:
            recent_3m = monthly_sales['count'].tail(3).sum()
            previous_3m = monthly_sales['count'].iloc[-6:-3].sum()
            quarter_over_quarter = ((recent_3m - previous_3m) / previous_3m) * 100 if previous_3m > 0 else None
        else:
            quarter_over_quarter = None
        
        # Calculate year-over-year if possible
        today = datetime.now()
        one_year_ago = today - timedelta(days=365)
        recent_data = df[df[date_col] >= one_year_ago]
        
        if len(recent_data) > 0:
            # Split into current year and previous year
            six_months_ago = today - timedelta(days=180)
            current_period = df[(df[date_col] >= six_months_ago) & (df[date_col] <= today)]
            previous_period = df[(df[date_col] >= one_year_ago) & (df[date_col] < six_months_ago)]
            
            current_count = len(current_period)
            previous_count = len(previous_period)
            
            if previous_count > 0:
                year_over_year = ((current_count - previous_count) / previous_count) * 100
            else:
                year_over_year = None
        else:
            year_over_year = None
        
        # Calculate overall trend metrics
        if not monthly_sales.empty and len(monthly_sales) > 1:
            trend_metrics = calculate_change_metrics(monthly_sales['count'])
        else:
            trend_metrics = {'trend_direction': 'insufficient_data'}
        
        # Detect seasonality
        seasonality = detect_seasonality(df, date_col, 'dummy_value', freq='M')
        if not seasonality['has_seasonality']:
            # Try with day of week
            df['dummy_value'] = 1  # Add dummy value for counting
            seasonality = detect_seasonality(df, date_col, 'dummy_value', freq='D')
        
        # Format for charting
        chart_data = {
            'type': 'line',
            'data': {
                'x': monthly_sales[date_col].dt.strftime('%b %Y').tolist() if not monthly_sales.empty else [],
                'y': monthly_sales['count'].tolist() if not monthly_sales.empty else []
            },
            'title': 'Monthly Sales Trend'
        }
        
        return {
            'trend_direction': trend_metrics.get('trend_direction', 'unknown'),
            'total_sales': len(df),
            'average_monthly': float(monthly_sales['count'].mean()) if not monthly_sales.empty else None,
            'quarter_over_quarter_change': quarter_over_quarter,
            'year_over_year_change': year_over_year,
            'peak_periods': seasonality['peak_periods'],
            'has_seasonality': seasonality['has_seasonality'],
            'chart_data': chart_data
        }
        
    except Exception as e:
        print(f"Error analyzing sales trend: {e}")
        return {'error': str(e)}

def analyze_gross_profit(df: pd.DataFrame, gross_col: str = 'Gross_Profit', 
                        date_col: str = 'Sale_Date') -> Dict[str, Any]:
    """
    Comprehensive gross profit analysis.
    
    Args:
        df: Input DataFrame with sales data
        gross_col: Name of the gross profit column
        date_col: Name of the date column
        
    Returns:
        Dictionary with gross profit analysis results
    """
    if df is None or gross_col not in df.columns:
        return {'error': 'Missing required data'}
    
    try:
        # Ensure gross column is numeric
        df[gross_col] = pd.to_numeric(df[gross_col], errors='coerce')
        
        # Basic metrics
        total_gross = df[gross_col].sum()
        average_gross = df[gross_col].mean()
        median_gross = df[gross_col].median()
        min_gross = df[gross_col].min()
        max_gross = df[gross_col].max()
        
        # Count of negative gross
        negative_gross_count = (df[gross_col] < 0).sum()
        negative_gross_percentage = (negative_gross_count / len(df)) * 100 if len(df) > 0 else 0
        
        # Time trend analysis if date column exists
        time_trend = None
        chart_data = None
        
        if date_col in df.columns:
            # Ensure date column is datetime type
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            df = df.dropna(subset=[date_col, gross_col])
            
            # Monthly average gross
            monthly_gross = analyze_time_series(
                df, date_col, gross_col, aggregation='mean', freq='M'
            )
            
            if not monthly_gross.empty and len(monthly_gross) > 1:
                time_trend = calculate_change_metrics(monthly_gross[gross_col])
                
                # Format for charting
                chart_data = {
                    'type': 'line',
                    'data': {
                        'x': monthly_gross[date_col].dt.strftime('%b %Y').tolist(),
                        'y': monthly_gross[gross_col].tolist()
                    },
                    'title': 'Monthly Average Gross Profit Trend'
                }
        
        # Analyze by vehicle type if available
        vehicle_breakdown = None
        
        if 'VehicleType' in df.columns:
            vehicle_breakdown = df.groupby('VehicleType')[gross_col].agg(['mean', 'count']).reset_index()
            vehicle_breakdown.columns = ['VehicleType', 'AvgGross', 'Count']
            vehicle_breakdown = vehicle_breakdown.to_dict(orient='records')
        elif all(col in df.columns for col in ['VehicleMake', 'VehicleModel']):
            # Create a simplified vehicle type
            df['VehicleType'] = df['VehicleMake'] 
            vehicle_breakdown = df.groupby('VehicleType')[gross_col].agg(['mean', 'count']).reset_index()
            vehicle_breakdown.columns = ['VehicleType', 'AvgGross', 'Count']
            vehicle_breakdown = vehicle_breakdown.to_dict(orient='records')
        
        # Create bar chart for vehicle type breakdown if available
        vehicle_chart = None
        if vehicle_breakdown:
            # Sort by average gross descending
            sorted_breakdown = sorted(vehicle_breakdown, key=lambda x: x['AvgGross'], reverse=True)
            top_types = sorted_breakdown[:8]  # Limit to top 8 for readability
            
            vehicle_chart = {
                'type': 'bar',
                'data': {
                    'x': [item['VehicleType'] for item in top_types],
                    'y': [item['AvgGross'] for item in top_types]
                },
                'title': 'Average Gross Profit by Vehicle Type'
            }
        
        return {
            'total_gross': total_gross,
            'average_gross': average_gross,
            'median_gross': median_gross,
            'min_gross': min_gross,
            'max_gross': max_gross,
            'negative_gross_count': negative_gross_count,
            'negative_gross_percentage': negative_gross_percentage,
            'time_trend': time_trend,
            'vehicle_breakdown': vehicle_breakdown,
            'chart_data': chart_data or vehicle_chart  # Prefer time trend chart if available
        }
        
    except Exception as e:
        print(f"Error analyzing gross profit: {e}")
        return {'error': str(e)}

def analyze_lead_sources(df: pd.DataFrame, lead_col: str = 'LeadSource', 
                        date_col: str = 'Sale_Date', 
                        gross_col: str = None) -> Dict[str, Any]:
    """
    Analyze lead source effectiveness.
    
    Args:
        df: Input DataFrame with sales data
        lead_col: Name of the lead source column
        date_col: Name of the date column (optional)
        gross_col: Name of the gross profit column (optional)
        
    Returns:
        Dictionary with lead source analysis results
    """
    if df is None or lead_col not in df.columns:
        return {'error': 'Missing required data'}
    
    try:
        # Clean lead sources (replace nulls with 'Unknown')
        df[lead_col] = df[lead_col].fillna('Unknown')
        df.loc[df[lead_col].str.strip() == '', lead_col] = 'Unknown'
        
        # Count by lead source
        lead_counts = df[lead_col].value_counts()
        total_count = lead_counts.sum()
        lead_percentages = (lead_counts / total_count) * 100
        
        lead_summary = []
        for lead, count in lead_counts.items():
            lead_summary.append({
                'source': lead,
                'count': int(count),
                'percentage': float(lead_percentages[lead])
            })
        
        # Sort by count (descending)
        lead_summary = sorted(lead_summary, key=lambda x: x['count'], reverse=True)
        
        # If gross profit column exists, calculate average gross by lead source
        if gross_col and gross_col in df.columns:
            df[gross_col] = pd.to_numeric(df[gross_col], errors='coerce')
            gross_by_lead = df.groupby(lead_col)[gross_col].mean().reset_index()
            gross_by_lead.columns = [lead_col, 'avg_gross']
            
            # Update lead summary with gross profit data
            gross_dict = gross_by_lead.set_index(lead_col)['avg_gross'].to_dict()
            for item in lead_summary:
                item['avg_gross'] = float(gross_dict.get(item['source'], 0))
        
        # Create pie chart for lead source distribution
        pie_chart = {
            'type': 'pie',
            'data': {
                'labels': [item['source'] for item in lead_summary[:8]],  # Top 8 sources
                'values': [item['percentage'] for item in lead_summary[:8]]
            },
            'title': 'Sales by Lead Source (%)'
        }
        
        # Analyze trend over time if date column exists
        time_trend = None
        
        if date_col in df.columns:
            try:
                df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                df = df.dropna(subset=[date_col])
                
                # Group by month and lead source
                df['month'] = df[date_col].dt.to_period('M')
                monthly_leads = df.groupby(['month', lead_col]).size().unstack(fill_value=0)
                
                # Get the top 3 lead sources for trend analysis
                top_sources = [item['source'] for item in lead_summary[:3]]
                
                # Analyze growth in top sources
                if len(monthly_leads) >= 2:
                    growth_rates = {}
                    for source in top_sources:
                        if source in monthly_leads.columns:
                            first_month = monthly_leads[source].iloc[0]
                            last_month = monthly_leads[source].iloc[-1]
                            
                            if first_month > 0:
                                growth = ((last_month - first_month) / first_month) * 100
                                growth_rates[source] = growth
                    
                    time_trend = {
                        'top_sources': top_sources,
                        'growth_rates': growth_rates
                    }
            except Exception as e:
                print(f"Error analyzing lead source time trend: {e}")
        
        return {
            'lead_summary': lead_summary,
            'top_source': lead_summary[0]['source'] if lead_summary else None,
            'unknown_percentage': next((item['percentage'] for item in lead_summary if item['source'] == 'Unknown'), 0),
            'time_trend': time_trend,
            'chart_data': pie_chart
        }
        
    except Exception as e:
        print(f"Error analyzing lead sources: {e}")
        return {'error': str(e)}

def analyze_inventory_health(df: pd.DataFrame, days_col: str = 'DaysInInventory', 
                           status_col: str = None) -> Dict[str, Any]:
    """
    Analyze inventory health.
    
    Args:
        df: Input DataFrame with inventory data
        days_col: Name of the days in inventory column
        status_col: Name of the inventory status column (optional)
        
    Returns:
        Dictionary with inventory health analysis results
    """
    if df is None or days_col not in df.columns:
        return {'error': 'Missing required data'}
    
    try:
        # Ensure days column is numeric
        df[days_col] = pd.to_numeric(df[days_col], errors='coerce')
        df = df.dropna(subset=[days_col])
        
        # Basic metrics
        avg_days = df[days_col].mean()
        median_days = df[days_col].median()
        
        # Age buckets
        age_buckets = {
            '<30 days': len(df[df[days_col] < 30]),
            '30-60 days': len(df[(df[days_col] >= 30) & (df[days_col] < 60)]),
            '61-90 days': len(df[(df[days_col] >= 60) & (df[days_col] < 90)]),
            '>90 days': len(df[df[days_col] >= 90])
        }
        
        total_units = sum(age_buckets.values())
        
        age_percentages = {}
        for bucket, count in age_buckets.items():
            age_percentages[bucket] = (count / total_units) * 100 if total_units > 0 else 0
        
        # Aged inventory (>90 days)
        aged_inventory = age_buckets['>90 days']
        aged_percentage = age_percentages['>90 days']
        
        # Calculate turn rate (approximate)
        fresh_inventory = age_buckets['<30 days']
        turn_rate = (fresh_inventory / total_units) * (365 / 30) if total_units > 0 else 0
        
        # Create pie chart for age distribution
        pie_chart = {
            'type': 'pie',
            'data': {
                'labels': list(age_buckets.keys()),
                'values': list(age_percentages.values())
            },
            'title': 'Inventory Age Distribution (%)'
        }
        
        # Analyze by vehicle type if available
        vehicle_breakdown = None
        
        if 'VehicleType' in df.columns:
            vehicle_breakdown = df.groupby('VehicleType')[days_col].mean().reset_index()
            vehicle_breakdown.columns = ['VehicleType', 'AvgDays']
            vehicle_breakdown = vehicle_breakdown.to_dict(orient='records')
        elif all(col in df.columns for col in ['VehicleMake', 'VehicleModel']):
            # Create a simplified vehicle type
            df['VehicleType'] = df['VehicleMake'] 
            vehicle_breakdown = df.groupby('VehicleType')[days_col].mean().reset_index()
            vehicle_breakdown.columns = ['VehicleType', 'AvgDays']
            vehicle_breakdown = vehicle_breakdown.to_dict(orient='records')
        
        return {
            'total_units': total_units,
            'average_days': avg_days,
            'median_days': median_days,
            'age_buckets': age_buckets,
            'age_percentages': age_percentages,
            'aged_inventory': aged_inventory,
            'aged_percentage': aged_percentage,
            'turn_rate': turn_rate,
            'vehicle_breakdown': vehicle_breakdown,
            'chart_data': pie_chart
        }
        
    except Exception as e:
        print(f"Error analyzing inventory health: {e}")
        return {'error': str(e)}

if __name__ == "__main__":
    # Example usage
    import pandas as pd
    import numpy as np
    
    # Sample data
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    np.random.seed(42)  # For reproducibility
    
    # Create sample sales data
    n_sales = 500
    sales_indices = np.random.choice(len(dates), size=n_sales, replace=True)
    
    sales_data = pd.DataFrame({
        'Sale_Date': dates[sales_indices],
        'VIN': [f'VIN{i:06d}' for i in range(n_sales)],
        'Gross_Profit': np.random.normal(2000, 800, n_sales),
        'LeadSource': np.random.choice(['Website', 'Walk-in', 'Referral', 'Third-party', 'Unknown'], n_sales),
        'VehicleMake': np.random.choice(['Toyota', 'Honda', 'Ford', 'Chevrolet', 'BMW', 'Mercedes'], n_sales),
        'VehicleModel': np.random.choice(['Sedan', 'SUV', 'Truck', 'Coupe'], n_sales)
    })
    
    # Create some pattern - more sales in certain months
    summer_indices = np.where((sales_data['Sale_Date'].dt.month >= 6) & 
                             (sales_data['Sale_Date'].dt.month <= 8))[0]
    winter_indices = np.where((sales_data['Sale_Date'].dt.month == 12) | 
                             (sales_data['Sale_Date'].dt.month == 1))[0]
    
    # Add 100 more summer sales
    summer_boost = 100
    summer_dates = dates[(dates.month >= 6) & (dates.month <= 8)]
    summer_sales_indices = np.random.choice(len(summer_dates), size=summer_boost, replace=True)
    
    summer_sales = pd.DataFrame({
        'Sale_Date': summer_dates[summer_sales_indices],
        'VIN': [f'VIN_SUMMER{i:06d}' for i in range(summer_boost)],
        'Gross_Profit': np.random.normal(2200, 700, summer_boost),  # Slightly higher gross in summer
        'LeadSource': np.random.choice(['Website', 'Walk-in', 'Referral', 'Third-party', 'Unknown'], summer_boost),
        'VehicleMake': np.random.choice(['Toyota', 'Honda', 'Ford', 'Chevrolet', 'BMW', 'Mercedes'], summer_boost),
        'VehicleModel': np.random.choice(['Sedan', 'SUV', 'Truck', 'Coupe'], summer_boost)
    })
    
    # Combine the datasets
    sales_data = pd.concat([sales_data, summer_sales], ignore_index=True)
    
    # Sort by date
    sales_data = sales_data.sort_values('Sale_Date').reset_index(drop=True)
    
    # Create inventory data
    inventory_data = pd.DataFrame({
        'VIN': [f'INV{i:06d}' for i in range(200)],
        'DaysInInventory': np.random.exponential(scale=45, size=200),
        'VehicleMake': np.random.choice(['Toyota', 'Honda', 'Ford', 'Chevrolet', 'BMW', 'Mercedes'], 200),
        'VehicleModel': np.random.choice(['Sedan', 'SUV', 'Truck', 'Coupe'], 200)
    })
    
    # Run analysis
    sales_trend = analyze_sales_trend(sales_data)
    gross_analysis = analyze_gross_profit(sales_data)
    lead_analysis = analyze_lead_sources(sales_data)
    inventory_analysis = analyze_inventory_health(inventory_data)
    
    # Print results
    print("Sales Trend Analysis:")
    print(json.dumps(sales_trend, indent=2))
    
    print("\nGross Profit Analysis:")
    print(json.dumps(gross_analysis, indent=2))
    
    print("\nLead Source Analysis:")
    print(json.dumps(lead_analysis, indent=2))
    
    print("\nInventory Health Analysis:")
    print(json.dumps(inventory_analysis, indent=2))