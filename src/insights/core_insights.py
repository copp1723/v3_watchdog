"""
Core insights module for Watchdog AI.

Provides high-value, actionable insights for automotive dealership data.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime, timedelta
import logging
import sentry_sdk
from ..utils.term_normalizer import TermNormalizer

# Create a normalizer instance
normalizer = TermNormalizer()

# Configure logger
logger = logging.getLogger(__name__)

def compute_sales_performance(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate comprehensive sales performance metrics by sales representative.
    
    This insight provides a detailed breakdown of sales performance metrics,
    including total units sold, average gross profit, and performance ranking
    relative to dealership averages. It identifies top performers and highlights
    opportunities for improvement.
    
    Args:
        df: DataFrame containing sales data with at least the following columns:
           - SalesRepName or similar (sales representative identifier)
           - TotalGross or similar (profit amount)
           - SaleDate or similar (date of sale)
           - Any column representing a unique vehicle identifier (VIN, StockNumber, etc.)
           
    Returns:
        Dict containing:
            - overall_metrics: Dealership-wide aggregates
            - rep_metrics: Per-representative metrics
            - top_performers: Details on highest-performing reps
            - ranking: Percentile ranking of all reps
            - insights: Key observations and actionable insights
            - time_based: Time-based trends if available
    
    Raises:
        ValueError: If required columns are missing or data is insufficient
    """
    try:
        # Track insight generation in Sentry
        sentry_sdk.set_tag("insight_type", "sales_performance")
        sentry_sdk.set_tag("data_rows", len(df))
        
        # Identify necessary columns based on common naming patterns
        # Sales rep column
        rep_cols = [col for col in df.columns if any(term in col.lower() 
                                                   for term in ['salesrep', 'rep', 'salesperson', 'sales_rep'])]
        if not rep_cols:
            error_msg = "No sales representative column found in data"
            logger.error(error_msg)
            sentry_sdk.capture_message(error_msg, level="error")
            raise ValueError(error_msg)
        rep_col = rep_cols[0]
        
        # Gross profit column
        gross_cols = [col for col in df.columns if any(term in col.lower() 
                                                     for term in ['gross', 'profit', 'totalgross', 'total_gross'])]
        if not gross_cols:
            error_msg = "No gross profit column found in data"
            logger.error(error_msg)
            sentry_sdk.capture_message(error_msg, level="error")
            raise ValueError(error_msg)
        gross_col = gross_cols[0]
        
        # Date column
        date_cols = [col for col in df.columns if any(term in col.lower() 
                                                    for term in ['date', 'saledate', 'sale_date'])]
        if not date_cols:
            logger.warning("No date column found in data. Time-based analysis will be skipped.")
            date_col = None
        else:
            date_col = date_cols[0]
            # Ensure date column is datetime type
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        
        # Clean and prepare data
        # Convert gross to numeric, handling currency symbols and commas
        df[gross_col] = pd.to_numeric(df[gross_col].astype(str).str.replace(r'[$,]', '', regex=True), 
                                      errors='coerce')
        
        # Drop rows with missing critical values
        required_cols = [rep_col, gross_col] + ([date_col] if date_col else [])
        df_clean = df.dropna(subset=required_cols)
        
        # Log data cleaning results
        cleaning_results = {
            "original_rows": len(df),
            "cleaned_rows": len(df_clean),
            "dropped_rows": len(df) - len(df_clean),
            "gross_nan_count": df[gross_col].isna().sum(),
            "rep_unique_count": df[rep_col].nunique()
        }
        logger.info(f"Data cleaning results: {cleaning_results}")
        
        if len(df_clean) < 5:  # Arbitrary threshold for meaningful analysis
            error_msg = f"Insufficient data for analysis after cleaning. Only {len(df_clean)} valid rows."
            logger.error(error_msg)
            sentry_sdk.capture_message(error_msg, level="error")
            raise ValueError(error_msg)
        
        # Calculate overall dealership metrics
        total_sales = len(df_clean)
        total_gross = df_clean[gross_col].sum()
        avg_gross_per_deal = total_gross / total_sales if total_sales > 0 else 0
        
        overall_metrics = {
            "total_sales": total_sales,
            "total_gross": total_gross,
            "avg_gross_per_deal": avg_gross_per_deal,
            "gross_std_dev": float(df_clean[gross_col].std()),
            "negative_gross_deals": int((df_clean[gross_col] < 0).sum()),
            "negative_gross_percentage": float((df_clean[gross_col] < 0).sum() / total_sales * 100) if total_sales > 0 else 0
        }
        
        # Calculate per-rep metrics
        rep_data = []
        for rep, group in df_clean.groupby(rep_col):
            rep_sales = len(group)
            rep_gross = group[gross_col].sum()
            rep_avg_gross = rep_gross / rep_sales if rep_sales > 0 else 0
            
            # Calculate percentiles for ranking
            sales_percentile = (rep_sales / overall_metrics["total_sales"]) * 100 if overall_metrics["total_sales"] > 0 else 0
            
            # Calculate efficiency metrics
            efficiency_ratio = rep_avg_gross / overall_metrics["avg_gross_per_deal"] if overall_metrics["avg_gross_per_deal"] > 0 else 0
            
            # Add negative gross percentage
            neg_gross_count = (group[gross_col] < 0).sum()
            neg_gross_pct = (neg_gross_count / rep_sales * 100) if rep_sales > 0 else 0
            
            rep_data.append({
                "rep_name": rep,
                "sales_count": rep_sales,
                "total_gross": rep_gross,
                "avg_gross_per_deal": rep_avg_gross,
                "sales_percentile": sales_percentile,
                "efficiency_ratio": efficiency_ratio,
                "negative_gross_count": neg_gross_count,
                "negative_gross_percentage": neg_gross_pct
            })
        
        # Sort by total gross (descending)
        rep_metrics = sorted(rep_data, key=lambda x: x["total_gross"], reverse=True)
        
        # Identify top performers (top 20% or at least 1)
        top_count = max(1, round(len(rep_metrics) * 0.2))
        top_performers = rep_metrics[:top_count]
        
        # Calculate rankings and add percentile rank
        if len(rep_metrics) > 1:
            avg_grosses = [rep["avg_gross_per_deal"] for rep in rep_metrics]
            for rep in rep_metrics:
                rep["gross_percentile"] = (
                    sum(1 for g in avg_grosses if g <= rep["avg_gross_per_deal"]) / len(avg_grosses) * 100
                )
        else:
            for rep in rep_metrics:
                rep["gross_percentile"] = 100.0
        
        # Generate time-based analysis if date column exists
        time_based = {}
        if date_col and len(df_clean) >= 10:  # Need minimum data for time analysis
            try:
                # Group by month and calculate totals
                df_clean['month'] = df_clean[date_col].dt.to_period('M')
                monthly_data = df_clean.groupby('month').agg({
                    gross_col: ['sum', 'mean', 'count']
                })
                
                # Flatten the column names
                monthly_data.columns = ['total_gross', 'avg_gross', 'sales_count']
                monthly_data = monthly_data.reset_index()
                
                # Convert period to string for JSON serialization
                monthly_data['month'] = monthly_data['month'].astype(str)
                
                # Calculate trend direction over time
                if len(monthly_data) >= 2:
                    first_month = monthly_data.iloc[0]
                    last_month = monthly_data.iloc[-1]
                    
                    gross_change = (last_month['total_gross'] - first_month['total_gross']) / first_month['total_gross']
                    sales_change = (last_month['sales_count'] - first_month['sales_count']) / first_month['sales_count']
                    
                    trends = {
                        "gross_change_pct": float(gross_change * 100) if not pd.isna(gross_change) else 0,
                        "sales_change_pct": float(sales_change * 100) if not pd.isna(sales_change) else 0,
                        "gross_trend": "increasing" if gross_change > 0.05 else "decreasing" if gross_change < -0.05 else "stable",
                        "sales_trend": "increasing" if sales_change > 0.05 else "decreasing" if sales_change < -0.05 else "stable",
                    }
                    
                    time_based = {
                        "monthly_data": monthly_data.to_dict('records'),
                        "trends": trends
                    }
            except Exception as e:
                logger.warning(f"Error in time-based analysis: {str(e)}")
                # Continue without time-based analysis
        
        # Generate insights
        insights = []
        
        # Top performer insight
        if top_performers:
            top_rep = top_performers[0]
            insights.append({
                "type": "top_performer",
                "title": "Top Sales Rep Performance",
                "description": (
                    f"{top_rep['rep_name']} is the top performer with ${top_rep['total_gross']:,.2f} in total gross profit "
                    f"from {top_rep['sales_count']} deals, averaging ${top_rep['avg_gross_per_deal']:,.2f} per deal."
                )
            })
        
        # Performance distribution
        if len(rep_metrics) > 1:
            # Calculate what percentage of total gross comes from top performer
            top_gross_pct = top_performers[0]['total_gross'] / overall_metrics['total_gross'] * 100
            
            insights.append({
                "type": "distribution",
                "title": "Sales Performance Distribution",
                "description": (
                    f"The top performer generates {top_gross_pct:.1f}% of total gross profit. "
                    f"There is a {max(rep['avg_gross_per_deal'] for rep in rep_metrics) / min(rep['avg_gross_per_deal'] for rep in rep_metrics):.1f}x "
                    f"difference between highest and lowest average gross per deal among sales reps."
                )
            })
        
        # Negative gross insight if significant
        if overall_metrics["negative_gross_percentage"] > 5:
            # Find rep with highest negative gross percentage
            highest_neg_pct_rep = max(rep_metrics, key=lambda x: x["negative_gross_percentage"])
            
            insights.append({
                "type": "negative_gross",
                "title": "Negative Gross Profit Analysis",
                "description": (
                    f"{overall_metrics['negative_gross_percentage']:.1f}% of all deals have negative gross profit. "
                    f"{highest_neg_pct_rep['rep_name']} has the highest rate at {highest_neg_pct_rep['negative_gross_percentage']:.1f}% of deals."
                )
            })
        
        # Time-based insight if available
        if time_based and "trends" in time_based:
            trends = time_based["trends"]
            insights.append({
                "type": "trend",
                "title": "Sales Performance Trend",
                "description": (
                    f"Total gross profit is {trends['gross_trend']} "
                    f"({trends['gross_change_pct']:.1f}%) while sales volume is {trends['sales_trend']} "
                    f"({trends['sales_change_pct']:.1f}%) over the analyzed period."
                )
            })
        
        # Log successful generation
        logger.info(f"Successfully generated sales performance insight for {len(rep_metrics)} sales reps")
        sentry_sdk.capture_message("Insight: sales_performance generated successfully", level="info")
        
        # Return compiled insight data
        return {
            "insight_type": "sales_performance",
            "generated_at": datetime.now().isoformat(),
            "overall_metrics": overall_metrics,
            "rep_metrics": rep_metrics,
            "top_performers": top_performers,
            "insights": insights,
            "time_based": time_based
        }
        
    except Exception as e:
        error_msg = f"Error generating sales performance insight: {str(e)}"
        logger.error(error_msg)
        sentry_sdk.capture_exception(e)
        
        # Return error information
        return {
            "insight_type": "sales_performance",
            "generated_at": datetime.now().isoformat(),
            "error": str(e),
            "success": False
        }

def compute_inventory_anomalies(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Identify anomalies in inventory data, particularly vehicles with unusually 
    long time in inventory compared to similar models.
    
    This insight analyzes inventory aging patterns, identifies outliers, and
    provides recommendations for inventory management. It highlights specific
    vehicles that are taking significantly longer to sell than similar models,
    helping dealerships optimize their inventory turnover.
    
    Args:
        df: DataFrame containing inventory data with at least the following columns:
           - VIN or similar (vehicle identifier)
           - DaysInInventory or similar (days vehicle has been in inventory)
           - Make and Model (vehicle make and model)
           - Any pricing columns (cost, list price, etc.)
           
    Returns:
        Dict containing:
            - overall_metrics: Inventory-wide statistics
            - model_metrics: Per-model/category metrics
            - outliers: Top outlier vehicles with details
            - recommendations: Actionable recommendations
            - insights: Key observations from analysis
    
    Raises:
        ValueError: If required columns are missing or data is insufficient
    """
    try:
        # Track insight generation in Sentry
        sentry_sdk.set_tag("insight_type", "inventory_anomalies")
        sentry_sdk.set_tag("data_rows", len(df))
        
        # Identify necessary columns based on common naming patterns
        # VIN or Stock Number
        id_cols = [col for col in df.columns if any(term in col.lower() 
                                                  for term in ['vin', 'stock', 'stocknumber', 'stock_number'])]
        if not id_cols:
            error_msg = "No vehicle identifier column found in data"
            logger.error(error_msg)
            sentry_sdk.capture_message(error_msg, level="error")
            raise ValueError(error_msg)
        id_col = id_cols[0]
        
        # Days in inventory column
        days_cols = [col for col in df.columns if any(term in col.lower() 
                                                    for term in ['days', 'daysinventory', 'days_in_inventory', 'age'])]
        if not days_cols:
            error_msg = "No inventory age/days column found in data"
            logger.error(error_msg)
            sentry_sdk.capture_message(error_msg, level="error")
            raise ValueError(error_msg)
        days_col = days_cols[0]
        
        # Make column
        make_cols = [col for col in df.columns if any(term == col.lower() 
                                                    for term in ['make', 'vehiclemake', 'vehicle_make'])]
        if not make_cols:
            logger.warning("No vehicle make column found. Using alternative grouping.")
            make_col = None
        else:
            make_col = make_cols[0]
        
        # Model column
        model_cols = [col for col in df.columns if any(term == col.lower() 
                                                     for term in ['model', 'vehiclemodel', 'vehicle_model'])]
        if not model_cols:
            logger.warning("No vehicle model column found. Using alternative grouping.")
            model_col = None
        else:
            model_col = model_cols[0]
        
        # Price or cost column
        price_cols = [col for col in df.columns if any(term in col.lower() 
                                                     for term in ['price', 'cost', 'value', 'listprice', 'list_price'])]
        price_col = price_cols[0] if price_cols else None
        
        # Clean and prepare data
        # Convert days to numeric
        df[days_col] = pd.to_numeric(df[days_col], errors='coerce')
        
        if price_col:
            # Convert price to numeric, handling currency symbols and commas
            df[price_col] = pd.to_numeric(df[price_col].astype(str).str.replace(r'[$,]', '', regex=True), 
                                       errors='coerce')
        
        # Drop rows with missing critical values
        required_cols = [id_col, days_col] + ([make_col] if make_col else []) + ([model_col] if model_col else [])
        df_clean = df.dropna(subset=required_cols)
        
        # Log data cleaning results
        cleaning_results = {
            "original_rows": len(df),
            "cleaned_rows": len(df_clean),
            "dropped_rows": len(df) - len(df_clean)
        }
        logger.info(f"Data cleaning results: {cleaning_results}")
        
        if len(df_clean) < 5:  # Arbitrary threshold for meaningful analysis
            error_msg = f"Insufficient data for analysis after cleaning. Only {len(df_clean)} valid rows."
            logger.error(error_msg)
            sentry_sdk.capture_message(error_msg, level="error")
            raise ValueError(error_msg)
        
        # Create a model/category grouping field
        if make_col and model_col:
            # Use make + model as grouping
            df_clean['vehicle_category'] = df_clean[make_col] + ' ' + df_clean[model_col]
        elif make_col:
            # Use make only
            df_clean['vehicle_category'] = df_clean[make_col]
        elif model_col:
            # Use model only
            df_clean['vehicle_category'] = df_clean[model_col]
        else:
            # Create arbitrary categories based on VIN or other available data
            # This is a fallback if no make/model columns are available
            df_clean['vehicle_category'] = 'Unknown'
            logger.warning("No make or model columns available. Using 'Unknown' as category.")
        
        # Calculate overall inventory metrics
        total_vehicles = len(df_clean)
        avg_days = df_clean[days_col].mean()
        median_days = df_clean[days_col].median()
        
        # Define age buckets
        age_buckets = {
            '< 30 days': len(df_clean[df_clean[days_col] < 30]),
            '30-60 days': len(df_clean[(df_clean[days_col] >= 30) & (df_clean[days_col] < 60)]),
            '61-90 days': len(df_clean[(df_clean[days_col] >= 60) & (df_clean[days_col] < 90)]),
            '> 90 days': len(df_clean[df_clean[days_col] >= 90])
        }
        
        # Calculate percentages
        age_percentages = {bucket: count / total_vehicles * 100 for bucket, count in age_buckets.items()}
        
        overall_metrics = {
            "total_vehicles": total_vehicles,
            "avg_days_in_inventory": float(avg_days),
            "median_days_in_inventory": float(median_days),
            "max_days_in_inventory": float(df_clean[days_col].max()),
            "age_buckets": age_buckets,
            "age_percentages": age_percentages,
            "vehicles_over_90_days": age_buckets['> 90 days'],
            "percent_over_90_days": age_percentages['> 90 days']
        }
        
        # Calculate metrics by vehicle category
        model_data = []
        for category, group in df_clean.groupby('vehicle_category'):
            if len(group) < 2:  # Skip categories with too few vehicles
                continue
                
            category_avg = group[days_col].mean()
            category_median = group[days_col].median()
            category_std = group[days_col].std()
            
            # Calculate outlier threshold (mean + 1.5*std or 90 days, whichever is lower)
            outlier_threshold = min(category_avg + 1.5 * category_std, 90)
            
            # Count outliers
            outliers = group[group[days_col] > outlier_threshold]
            
            if price_col:
                avg_price = group[price_col].mean()
            else:
                avg_price = None
            
            model_data.append({
                "category": category,
                "count": len(group),
                "avg_days": float(category_avg),
                "median_days": float(category_median),
                "std_dev_days": float(category_std),
                "outlier_threshold": float(outlier_threshold),
                "outlier_count": len(outliers),
                "outlier_percentage": (len(outliers) / len(group) * 100) if len(group) > 0 else 0,
                "avg_price": float(avg_price) if avg_price is not None else None
            })
        
        # Sort by average days (descending)
        model_metrics = sorted(model_data, key=lambda x: x["avg_days"], reverse=True)
        
        # Identify specific outlier vehicles
        outlier_vehicles = []
        for category, group in df_clean.groupby('vehicle_category'):
            if len(group) < 3:  # Skip categories with too few vehicles
                continue
                
            # Get model metrics for this category
            try:
                model_metric = next(m for m in model_metrics if m["category"] == category)
                threshold = model_metric["outlier_threshold"]
            except StopIteration:
                # Skip if we don't have metrics for this category
                continue
            
            # Find outliers for this category
            category_outliers = group[group[days_col] > threshold].copy()
            
            # Skip if no outliers
            if len(category_outliers) == 0:
                continue
            
            # Calculate Z-score (how many standard deviations from the mean)
            if model_metric["std_dev_days"] > 0:
                category_outliers['z_score'] = (
                    (category_outliers[days_col] - model_metric["avg_days"]) / model_metric["std_dev_days"]
                )
            else:
                category_outliers['z_score'] = 0
            
            # Add category average to each record
            category_outliers['category_avg_days'] = model_metric["avg_days"]
            
            # Add variance from average
            category_outliers['days_above_avg'] = category_outliers[days_col] - model_metric["avg_days"]
            category_outliers['percent_above_avg'] = (
                (category_outliers[days_col] / model_metric["avg_days"] - 1) * 100
            )
            
            # Convert to records and add to list
            for _, vehicle in category_outliers.iterrows():
                vehicle_data = {
                    "id": str(vehicle[id_col]),
                    "category": category,
                    "days_in_inventory": float(vehicle[days_col]),
                    "category_avg_days": float(vehicle['category_avg_days']),
                    "days_above_avg": float(vehicle['days_above_avg']),
                    "percent_above_avg": float(vehicle['percent_above_avg']),
                    "z_score": float(vehicle['z_score'])
                }
                
                # Add make and model if available
                if make_col:
                    vehicle_data["make"] = str(vehicle[make_col])
                if model_col:
                    vehicle_data["model"] = str(vehicle[model_col])
                
                # Add price if available
                if price_col:
                    vehicle_data["price"] = float(vehicle[price_col])
                
                outlier_vehicles.append(vehicle_data)
        
        # Sort outliers by days above average (descending)
        outlier_vehicles = sorted(outlier_vehicles, key=lambda x: x["days_above_avg"], reverse=True)
        
        # Get top 3 outliers
        top_outliers = outlier_vehicles[:3] if len(outlier_vehicles) >= 3 else outlier_vehicles
        
        # Generate insights
        insights = []
        
        # Overall inventory health
        insights.append({
            "type": "inventory_health",
            "title": "Inventory Aging Overview",
            "description": (
                f"Your inventory averages {avg_days:.1f} days on lot with {age_percentages['> 90 days']:.1f}% "
                f"of vehicles ({age_buckets['> 90 days']} units) aging beyond 90 days."
            )
        })
        
        # Problematic categories
        if model_metrics:
            # Identify categories with highest average days
            high_avg_models = [m for m in model_metrics if m["avg_days"] > avg_days * 1.2][:3]
            
            if high_avg_models:
                model_list = ", ".join(m["category"] for m in high_avg_models)
                insights.append({
                    "type": "slow_moving_categories",
                    "title": "Slow-Moving Vehicle Categories",
                    "description": (
                        f"The following categories are moving slower than average: {model_list}, "
                        f"with average days in inventory {high_avg_models[0]['avg_days']:.1f} days "
                        f"({(high_avg_models[0]['avg_days']/avg_days - 1)*100:.1f}% above dealership average)."
                    )
                })
        
        # Specific outlier insight
        if top_outliers:
            top = top_outliers[0]
            insights.append({
                "type": "top_outlier",
                "title": "Most Overaged Vehicle",
                "description": (
                    f"A {top.get('make', '')} {top.get('model', top['category'])} has been in inventory for "
                    f"{top['days_in_inventory']:.0f} days, which is {top['percent_above_avg']:.1f}% longer than "
                    f"the average for similar vehicles ({top['category_avg_days']:.1f} days)."
                )
            })
        
        # Generate recommendations
        recommendations = []
        
        # Add recommendation for aged inventory
        if age_percentages['> 90 days'] > 10:
            recommendations.append(
                "Consider a targeted promotion for vehicles over 90 days old, which represent "
                f"{age_percentages['> 90 days']:.1f}% of your inventory."
            )
        
        # Add recommendation for specific outliers
        if top_outliers:
            id_list = ", ".join(o["id"] for o in top_outliers)
            recommendations.append(
                f"Review pricing strategy for your top outliers (IDs: {id_list}), which have been in inventory "
                f"{(sum(o['days_in_inventory'] for o in top_outliers) / len(top_outliers)):.1f} days on average."
            )
        
        # Add recommendation for slow-moving categories
        if model_metrics and len(model_metrics) >= 3:
            slow_categories = sorted(model_metrics, key=lambda x: x["avg_days"], reverse=True)[:3]
            category_list = ", ".join(c["category"] for c in slow_categories)
            recommendations.append(
                f"Evaluate your stocking strategy for slow-moving categories: {category_list}."
            )
        
        # Log successful generation
        logger.info(f"Successfully generated inventory anomalies insight with {len(outlier_vehicles)} outliers identified")
        sentry_sdk.capture_message("Insight: inventory_anomalies generated successfully", level="info")
        
        # Return compiled insight data
        return {
            "insight_type": "inventory_anomalies",
            "generated_at": datetime.now().isoformat(),
            "overall_metrics": overall_metrics,
            "model_metrics": model_metrics,
            "outliers": top_outliers,
            "all_outliers": outlier_vehicles,
            "insights": insights,
            "recommendations": recommendations
        }
        
    except Exception as e:
        error_msg = f"Error generating inventory anomalies insight: {str(e)}"
        logger.error(error_msg)
        sentry_sdk.capture_exception(e)
        
        # Return error information
        return {
            "insight_type": "inventory_anomalies",
            "generated_at": datetime.now().isoformat(),
            "error": str(e),
            "success": False
        }
        
def get_sales_rep_performance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate key sales performance metrics for each sales representative.
    
    This function returns a simplified DataFrame containing essential 
    sales performance metrics for each sales rep, including total cars sold,
    average gross profit, and delta from dealership average.
    
    Args:
        df: DataFrame containing sales data with at least the following columns:
           - SalesRepName or similar (sales representative identifier)
           - TotalGross or similar (profit amount)
           - Any column representing a unique vehicle identifier (VIN, StockNumber, etc.)
           
    Returns:
        DataFrame with columns:
            - rep_name: Normalized sales representative name
            - total_cars_sold: Total number of vehicles sold by the rep
            - average_gross: Average gross profit per deal
            - delta_from_dealership_avg: Difference from the dealership average gross
    
    Raises:
        ValueError: If required columns are missing or data is insufficient
    """
    try:
        # Track insight generation in Sentry
        sentry_sdk.set_tag("insight_type", "sales_performance")
        sentry_sdk.set_tag("data_rows", len(df))
        
        # Identify necessary columns based on common naming patterns
        # Sales rep column
        rep_cols = [col for col in df.columns if any(term in col.lower() 
                                               for term in ['salesrep', 'rep', 'salesperson', 'sales_rep'])]
        if not rep_cols:
            error_msg = "No sales representative column found in data"
            logger.error(error_msg)
            sentry_sdk.capture_message(error_msg, level="error")
            raise ValueError(error_msg)
        rep_col = rep_cols[0]
        
        # Normalize sales rep names
        df = normalizer.normalize_dataframe(df, [rep_col])
        
        # Gross profit column
        gross_cols = [col for col in df.columns if any(term in col.lower() 
                                                 for term in ['gross', 'profit', 'totalgross', 'total_gross'])]
        if not gross_cols:
            error_msg = "No gross profit column found in data"
            logger.error(error_msg)
            sentry_sdk.capture_message(error_msg, level="error")
            raise ValueError(error_msg)
        gross_col = gross_cols[0]
        
        # Clean and prepare data
        # Convert gross to numeric, handling currency symbols and commas
        df[gross_col] = pd.to_numeric(df[gross_col].astype(str).str.replace(r'[$,]', '', regex=True), 
                                  errors='coerce')
        
        # Drop rows with missing critical values
        required_cols = [rep_col, gross_col]
        df_clean = df.dropna(subset=required_cols)
        
        # Log data cleaning results
        cleaning_results = {
            "original_rows": len(df),
            "cleaned_rows": len(df_clean),
            "dropped_rows": len(df) - len(df_clean),
            "gross_nan_count": df[gross_col].isna().sum(),
            "rep_unique_count": df_clean[rep_col].nunique()
        }
        logger.info(f"Data cleaning results: {cleaning_results}")
        
        if len(df_clean) < 2:  # Minimum threshold for meaningful analysis
            error_msg = f"Insufficient data for analysis after cleaning. Only {len(df_clean)} valid rows."
            logger.error(error_msg)
            sentry_sdk.capture_message(error_msg, level="error")
            raise ValueError(error_msg)
        
        # Calculate dealership average
        dealership_avg_gross = df_clean[gross_col].mean()
        
        # Group by sales rep and calculate metrics
        rep_metrics = df_clean.groupby(rep_col).agg({
            gross_col: ['mean', 'sum', 'count']
        }).reset_index()
        
        # Flatten multi-level columns
        rep_metrics.columns = [rep_col, 'average_gross', 'total_gross', 'total_cars_sold']
        
        # Add delta from dealership average
        rep_metrics['delta_from_dealership_avg'] = rep_metrics['average_gross'] - dealership_avg_gross
        
        # Rename rep column to standardized name
        result = rep_metrics.rename(columns={rep_col: 'rep_name'})
        
        # Select and order only the required columns
        result = result[['rep_name', 'total_cars_sold', 'average_gross', 'delta_from_dealership_avg']]
        
        # Sort by total cars sold (descending)
        result = result.sort_values('total_cars_sold', ascending=False)
        
        # Log successful completion
        logger.info(f"Successfully generated sales rep performance metrics for {len(result)} reps")
        
        return result
    
    except Exception as e:
        error_msg = f"Error generating sales rep performance insight: {str(e)}"
        logger.error(error_msg)
        sentry_sdk.capture_exception(e)
        raise ValueError(error_msg)

def get_inventory_aging_alerts(df: pd.DataFrame, threshold_days: int = 60) -> pd.DataFrame:
    """
    Identify vehicles with unusually long time in inventory compared to their model average.
    
    This function flags vehicles whose days on lot exceed the model average
    plus a threshold value, highlighting inventory that may need attention.
    
    Args:
        df: DataFrame containing inventory data with at least the following columns:
           - VIN or StockNumber (vehicle identifier)
           - DaysInInventory or similar (days vehicle has been in inventory)
           - Make and Model (vehicle make and model)
        threshold_days: Additional days beyond model average to flag as anomaly (default: 60)
           
    Returns:
        DataFrame with columns:
            - vin: Vehicle identifier (VIN or stock number)
            - model: Vehicle model (make + model if available)
            - days_on_lot: Total days in inventory
            - model_avg_days: Average days for this model
            - excess_days: Days exceeding model average
    
    Raises:
        ValueError: If required columns are missing or data is insufficient
    """
    try:
        # Track insight generation in Sentry
        sentry_sdk.set_tag("insight_type", "inventory_aging")
        sentry_sdk.set_tag("data_rows", len(df))
        
        # Identify necessary columns based on common naming patterns
        # VIN or Stock Number
        id_cols = [col for col in df.columns if any(term in col.lower() 
                                              for term in ['vin', 'stock', 'stocknumber', 'stock_number'])]
        if not id_cols:
            error_msg = "No vehicle identifier column found in data"
            logger.error(error_msg)
            sentry_sdk.capture_message(error_msg, level="error")
            raise ValueError(error_msg)
        id_col = id_cols[0]
        
        # Days in inventory column
        days_cols = [col for col in df.columns if any(term in col.lower() 
                                                for term in ['days', 'daysinventory', 'days_in_inventory', 'age'])]
        if not days_cols:
            error_msg = "No inventory age/days column found in data"
            logger.error(error_msg)
            sentry_sdk.capture_message(error_msg, level="error")
            raise ValueError(error_msg)
        days_col = days_cols[0]
        
        # Make column
        make_cols = [col for col in df.columns if any(term == col.lower() 
                                                for term in ['make', 'vehiclemake', 'vehicle_make'])]
        if not make_cols:
            logger.warning("No vehicle make column found. Using model only for grouping.")
            make_col = None
        else:
            make_col = make_cols[0]
        
        # Model column
        model_cols = [col for col in df.columns if any(term == col.lower() 
                                                 for term in ['model', 'vehiclemodel', 'vehicle_model'])]
        if not model_cols:
            logger.warning("No vehicle model column found. Using make only for grouping.")
            model_col = None
        else:
            model_col = model_cols[0]
        
        # Clean and prepare data
        # Convert days to numeric
        df[days_col] = pd.to_numeric(df[days_col], errors='coerce')
        
        # Drop rows with missing critical values
        required_cols = [id_col, days_col] + ([make_col] if make_col else []) + ([model_col] if model_col else [])
        df_clean = df.dropna(subset=required_cols)
        
        # Create model column (make + model if both available)
        if make_col and model_col:
            df_clean['full_model'] = df_clean[make_col] + ' ' + df_clean[model_col]
            model_group_col = 'full_model'
        elif make_col:
            df_clean['full_model'] = df_clean[make_col]
            model_group_col = 'full_model'
        elif model_col:
            model_group_col = model_col
        else:
            error_msg = "Neither make nor model columns were found. Cannot group vehicles."
            logger.error(error_msg)
            sentry_sdk.capture_message(error_msg, level="error")
            raise ValueError(error_msg)
        
        # Calculate average days by model
        model_avg = df_clean.groupby(model_group_col)[days_col].mean().reset_index()
        model_avg = model_avg.rename(columns={days_col: 'model_avg_days', model_group_col: 'model'})
        
        # Merge averages back to main DataFrame
        if model_group_col == 'full_model':
            df_with_avg = df_clean.merge(model_avg, left_on='full_model', right_on='model')
        else:
            df_with_avg = df_clean.merge(model_avg, left_on=model_col, right_on='model')
        
        # Calculate excess days
        df_with_avg['excess_days'] = df_with_avg[days_col] - df_with_avg['model_avg_days']
        
        # Filter vehicles that exceed threshold
        alerts = df_with_avg[df_with_avg['excess_days'] > threshold_days].copy()
        
        # If no vehicles exceed threshold, return empty DataFrame with correct columns
        if len(alerts) == 0:
            logger.info("No vehicles found exceeding the model average + threshold")
            return pd.DataFrame(columns=['vin', 'model', 'days_on_lot', 'model_avg_days', 'excess_days'])
        
        # Select and rename columns for final output
        result = alerts[[id_col, 'model', days_col, 'model_avg_days', 'excess_days']]
        result = result.rename(columns={id_col: 'vin', days_col: 'days_on_lot'})
        
        # Sort by excess days (descending)
        result = result.sort_values('excess_days', ascending=False)
        
        # Log successful completion
        logger.info(f"Successfully identified {len(result)} vehicles with inventory aging alerts")
        
        return result
    
    except Exception as e:
        error_msg = f"Error generating inventory aging alerts: {str(e)}"
        logger.error(error_msg)
        sentry_sdk.capture_exception(e)
        raise ValueError(error_msg)