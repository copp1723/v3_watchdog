"""
Lead Flow Optimization Module for V3 Watchdog AI.

Provides functionality for analyzing lead lifecycle, tracking time from 
creation to closing, identifying bottlenecks, and flagging outliers.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime, timedelta
import logging
import json
import os
from pathlib import Path
import time

# Local imports
from .lead_source_roi import LeadSourceNormalizer

logger = logging.getLogger(__name__)

# Constants
DEFAULT_BOTTLENECK_THRESHOLDS = {
    "time_to_contact": 1.0,  # days
    "time_to_appointment": 3.0,  # days
    "time_to_test_drive": 5.0,  # days
    "time_to_offer": 7.0,  # days
    "time_to_sold": 14.0,  # days
    "time_to_delivery": 21.0,  # days
    "time_to_closed": 30.0,  # days
    "aging_threshold": 30.0,  # days (leads older than this are flagged)
}

DEFAULT_STAGES = [
    "created",
    "contacted",
    "appointment",
    "test_drive",
    "offer",
    "sold",
    "delivered",
    "closed"
]

class LeadFlowMetrics:
    """Calculate and track lead flow metrics through the sales pipeline."""
    
    def __init__(self, 
                 thresholds: Optional[Dict[str, float]] = None,
                 custom_stages: Optional[List[str]] = None,
                 storage_path: Optional[str] = None):
        """
        Initialize the lead flow metrics tracker.
        
        Args:
            thresholds: Optional custom thresholds for bottleneck detection
            custom_stages: Optional custom lead stages
            storage_path: Optional path to store metrics data
        """
        # Set thresholds (use defaults and update with custom)
        self.thresholds = DEFAULT_BOTTLENECK_THRESHOLDS.copy()
        if thresholds:
            self.thresholds.update(thresholds)
        
        # Set stages (use defaults or custom)
        self.stages = custom_stages or DEFAULT_STAGES.copy()
        
        # Set storage path
        self.storage_path = storage_path or os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "data", 
            "lead_flow_metrics.json"
        )
        
        # Initialize data storage
        self._load_metrics_data()
    
    def _load_metrics_data(self) -> None:
        """Load stored metrics data or initialize if not found."""
        try:
            if os.path.exists(self.storage_path):
                with open(self.storage_path, 'r') as f:
                    self.metrics_data = json.load(f)
            else:
                # Initialize with empty structure
                self.metrics_data = {
                    "bottlenecks": {},
                    "outliers": {},
                    "stage_metrics": {},
                    "rep_metrics": {},
                    "source_metrics": {},
                    "model_metrics": {},
                    "updated_at": datetime.now().isoformat()
                }
                # Save the initialized structure
                self._save_metrics_data()
        except Exception as e:
            logger.error(f"Error loading metrics data: {e}")
            # Fall back to empty structure
            self.metrics_data = {
                "bottlenecks": {},
                "outliers": {},
                "stage_metrics": {},
                "rep_metrics": {},
                "source_metrics": {},
                "model_metrics": {},
                "updated_at": datetime.now().isoformat()
            }
    
    def _save_metrics_data(self) -> None:
        """Save metrics data to storage."""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
            
            # Update timestamp
            self.metrics_data["updated_at"] = datetime.now().isoformat()
            
            # Save to file
            with open(self.storage_path, 'w') as f:
                json.dump(self.metrics_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving metrics data: {e}")
    
    def process_lead_data(self, df: pd.DataFrame, 
                         id_col: str = "LeadID",
                         stage_date_format: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Process lead flow data to identify bottlenecks and outliers.
        
        Args:
            df: DataFrame with lead data
            id_col: Column with lead IDs
            stage_date_format: Optional mapping of stage columns to date formats
            
        Returns:
            Dictionary with processed metrics
        """
        if df is None or df.empty or id_col not in df.columns:
            return {"error": "Invalid or empty data"}
        
        # Create a working copy
        work_df = df.copy()
        
        # Ensure all stage columns exist (create if missing)
        for stage in self.stages:
            stage_col = f"{stage}_date"
            if stage_col not in work_df.columns:
                work_df[stage_col] = None
        
        # Convert date strings to datetime objects
        date_format = stage_date_format or {}
        for stage in self.stages:
            stage_col = f"{stage}_date"
            if stage_col in work_df.columns:
                default_format = "%Y-%m-%d %H:%M:%S"
                fmt = date_format.get(stage_col, default_format)
                
                # Convert to datetime, accounting for various formats
                work_df[stage_col] = pd.to_datetime(
                    work_df[stage_col], 
                    format=fmt, 
                    errors='coerce'
                )
        
        # Calculate time between stages
        for i in range(1, len(self.stages)):
            prev_stage = self.stages[i-1]
            curr_stage = self.stages[i]
            
            time_col = f"time_{prev_stage}_to_{curr_stage}"
            prev_col = f"{prev_stage}_date"
            curr_col = f"{curr_stage}_date"
            
            # Calculate time difference in days
            work_df[time_col] = (work_df[curr_col] - work_df[prev_col]).dt.total_seconds() / (24 * 3600)
            
            # Handle negative values (data entry errors)
            work_df.loc[work_df[time_col] < 0, time_col] = None
        
        # Calculate time to closed from created
        if "created_date" in work_df.columns and "closed_date" in work_df.columns:
            work_df["time_to_closed"] = (work_df["closed_date"] - work_df["created_date"]).dt.total_seconds() / (24 * 3600)
            work_df.loc[work_df["time_to_closed"] < 0, "time_to_closed"] = None
        
        # Add lead age (current date - created date)
        if "created_date" in work_df.columns:
            current_date = pd.Timestamp.now()
            work_df["lead_age"] = (current_date - work_df["created_date"]).dt.total_seconds() / (24 * 3600)
            work_df.loc[work_df["lead_age"] < 0, "lead_age"] = None
        
        # Process metrics
        results = self._calculate_metrics(work_df)
        
        # Update stored metrics
        self._update_stored_metrics(results)
        
        return results
    
    def _calculate_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate various lead flow metrics.
        
        Args:
            df: Processed DataFrame with lead data
            
        Returns:
            Dictionary with calculated metrics
        """
        results = {
            "bottlenecks": self._identify_bottlenecks(df),
            "outliers": self._identify_outliers(df),
            "stage_metrics": self._calculate_stage_metrics(df),
            "rep_metrics": self._calculate_rep_metrics(df),
            "source_metrics": self._calculate_source_metrics(df),
            "model_metrics": self._calculate_model_metrics(df),
            "timestamp": datetime.now().isoformat()
        }
        
        return results
    
    def _identify_bottlenecks(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Identify bottlenecks in the lead flow.
        
        Args:
            df: Processed DataFrame with lead data
            
        Returns:
            Dictionary with identified bottlenecks
        """
        bottlenecks = {}
        
        # Check each stage transition
        for i in range(1, len(self.stages)):
            prev_stage = self.stages[i-1]
            curr_stage = self.stages[i]
            
            time_col = f"time_{prev_stage}_to_{curr_stage}"
            
            if time_col in df.columns:
                # Calculate metrics for this transition
                avg_time = df[time_col].mean()
                median_time = df[time_col].median()
                max_time = df[time_col].max()
                
                # Check if average exceeds threshold
                threshold = self.thresholds.get(time_col, self.thresholds.get(f"time_to_{curr_stage}", 7.0))
                is_bottleneck = avg_time > threshold if avg_time is not None else False
                
                bottlenecks[time_col] = {
                    "average_days": avg_time,
                    "median_days": median_time,
                    "max_days": max_time,
                    "threshold": threshold,
                    "is_bottleneck": is_bottleneck
                }
        
        # Add overall time to closed
        if "time_to_closed" in df.columns:
            avg_time = df["time_to_closed"].mean()
            median_time = df["time_to_closed"].median()
            max_time = df["time_to_closed"].max()
            
            threshold = self.thresholds.get("time_to_closed", 30.0)
            is_bottleneck = avg_time > threshold if avg_time is not None else False
            
            bottlenecks["time_to_closed"] = {
                "average_days": avg_time,
                "median_days": median_time,
                "max_days": max_time,
                "threshold": threshold,
                "is_bottleneck": is_bottleneck
            }
        
        return bottlenecks
    
    def _identify_outliers(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Identify outlier leads based on age and processing time.
        
        Args:
            df: Processed DataFrame with lead data
            
        Returns:
            Dictionary with identified outliers
        """
        outliers = {}
        
        # Check for aged leads
        if "lead_age" in df.columns:
            age_threshold = self.thresholds.get("aging_threshold", 30.0)
            aged_leads = df[df["lead_age"] > age_threshold]
            
            outliers["aged_leads"] = {
                "count": len(aged_leads),
                "percentage": (len(aged_leads) / len(df)) * 100 if len(df) > 0 else 0,
                "threshold_days": age_threshold,
                "leads": aged_leads.head(20)[["LeadID", "lead_age"]].to_dict(orient="records") if "LeadID" in aged_leads.columns else []
            }
        
        # Check for outliers in each transition
        for i in range(1, len(self.stages)):
            prev_stage = self.stages[i-1]
            curr_stage = self.stages[i]
            
            time_col = f"time_{prev_stage}_to_{curr_stage}"
            
            if time_col in df.columns:
                # Get timeframe data with valid values
                timeframe_data = df[df[time_col].notna()]
                
                if len(timeframe_data) > 5:  # Need enough data for percentiles
                    # Calculate the 95th percentile
                    p95 = np.percentile(timeframe_data[time_col], 95)
                    
                    # Find extreme outliers (beyond 95th percentile)
                    extreme_outliers = timeframe_data[timeframe_data[time_col] > p95]
                    
                    outliers[time_col] = {
                        "count": len(extreme_outliers),
                        "percentage": (len(extreme_outliers) / len(timeframe_data)) * 100,
                        "p95_threshold": p95,
                        "leads": extreme_outliers.head(10)[["LeadID", time_col]].to_dict(orient="records") if "LeadID" in extreme_outliers.columns else []
                    }
        
        return outliers
    
    def _calculate_stage_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate metrics for each stage in the lead flow.
        
        Args:
            df: Processed DataFrame with lead data
            
        Returns:
            Dictionary with stage metrics
        """
        stage_metrics = {}
        
        # Calculate counts and conversion rates for each stage
        for stage in self.stages:
            stage_col = f"{stage}_date"
            
            if stage_col in df.columns:
                # Count leads with this stage completed
                completed = df[df[stage_col].notna()]
                count = len(completed)
                
                # Calculate percentage of total
                total = len(df)
                percentage = (count / total) * 100 if total > 0 else 0
                
                stage_metrics[stage] = {
                    "count": count,
                    "percentage": percentage
                }
                
                # Add conversion rates between stages
                if stage != self.stages[0]:  # Not the first stage
                    prev_stage = self.stages[self.stages.index(stage) - 1]
                    prev_stage_col = f"{prev_stage}_date"
                    
                    if prev_stage_col in df.columns:
                        # Count leads that completed previous stage
                        prev_count = len(df[df[prev_stage_col].notna()])
                        
                        # Calculate conversion rate
                        if prev_count > 0:
                            conversion_rate = (count / prev_count) * 100
                        else:
                            conversion_rate = 0
                        
                        stage_metrics[stage]["conversion_from_previous"] = conversion_rate
        
        return stage_metrics
    
    def _calculate_rep_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate metrics grouped by sales rep.
        
        Args:
            df: Processed DataFrame with lead data
            
        Returns:
            Dictionary with rep metrics
        """
        rep_metrics = {}
        
        # Check if sales rep column exists
        rep_col = self._find_rep_column(df)
        
        if rep_col:
            # Group by sales rep
            rep_groups = df.groupby(rep_col)
            
            for rep, group in rep_groups:
                if rep and str(rep).strip():  # Ensure rep name is valid
                    # Calculate basic metrics
                    total_leads = len(group)
                    
                    # Calculate conversion rates
                    if "sold_date" in group.columns and total_leads > 0:
                        sold_leads = len(group[group["sold_date"].notna()])
                        conversion_rate = (sold_leads / total_leads) * 100
                    else:
                        sold_leads = 0
                        conversion_rate = 0
                    
                    # Calculate average time to close
                    if "time_to_closed" in group.columns:
                        avg_close_time = group["time_to_closed"].mean()
                    else:
                        avg_close_time = None
                    
                    # Store metrics
                    rep_metrics[str(rep)] = {
                        "total_leads": total_leads,
                        "closed_leads": sold_leads,
                        "conversion_rate": conversion_rate,
                        "avg_days_to_close": avg_close_time
                    }
                    
                    # Add bottlenecks specific to this rep
                    bottlenecks = {}
                    for i in range(1, len(self.stages)):
                        prev_stage = self.stages[i-1]
                        curr_stage = self.stages[i]
                        
                        time_col = f"time_{prev_stage}_to_{curr_stage}"
                        
                        if time_col in group.columns:
                            avg_time = group[time_col].mean()
                            if avg_time is not None:
                                threshold = self.thresholds.get(time_col, self.thresholds.get(f"time_to_{curr_stage}", 7.0))
                                is_bottleneck = avg_time > threshold
                                
                                bottlenecks[time_col] = {
                                    "average_days": avg_time,
                                    "threshold": threshold,
                                    "is_bottleneck": is_bottleneck
                                }
                    
                    rep_metrics[str(rep)]["bottlenecks"] = bottlenecks
        
        return rep_metrics
    
    def _calculate_source_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate metrics grouped by lead source.
        
        Args:
            df: Processed DataFrame with lead data
            
        Returns:
            Dictionary with source metrics
        """
        source_metrics = {}
        
        # Check if lead source column exists
        source_col = self._find_source_column(df)
        
        if source_col:
            # Normalize sources if LeadSourceNormalizer is available
            try:
                normalizer = LeadSourceNormalizer()
                work_df = normalizer.normalize_df(df, source_col)
            except Exception:
                # If normalization fails, use original data
                work_df = df
            
            # Group by source
            source_groups = work_df.groupby(source_col)
            
            for source, group in source_groups:
                if source and str(source).strip():  # Ensure source name is valid
                    # Calculate basic metrics
                    total_leads = len(group)
                    
                    # Calculate conversion rates
                    if "sold_date" in group.columns and total_leads > 0:
                        sold_leads = len(group[group["sold_date"].notna()])
                        conversion_rate = (sold_leads / total_leads) * 100
                    else:
                        sold_leads = 0
                        conversion_rate = 0
                    
                    # Calculate average time to close
                    if "time_to_closed" in group.columns:
                        avg_close_time = group["time_to_closed"].mean()
                    else:
                        avg_close_time = None
                    
                    # Store metrics
                    source_metrics[str(source)] = {
                        "total_leads": total_leads,
                        "closed_leads": sold_leads,
                        "conversion_rate": conversion_rate,
                        "avg_days_to_close": avg_close_time
                    }
        
        return source_metrics
    
    def _calculate_model_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate metrics grouped by vehicle model.
        
        Args:
            df: Processed DataFrame with lead data
            
        Returns:
            Dictionary with model metrics
        """
        model_metrics = {}
        
        # Check if model column exists
        model_col = self._find_model_column(df)
        
        if model_col:
            # Group by model
            model_groups = df.groupby(model_col)
            
            for model, group in model_groups:
                if model and str(model).strip():  # Ensure model name is valid
                    # Calculate basic metrics
                    total_leads = len(group)
                    
                    # Calculate conversion rates
                    if "sold_date" in group.columns and total_leads > 0:
                        sold_leads = len(group[group["sold_date"].notna()])
                        conversion_rate = (sold_leads / total_leads) * 100
                    else:
                        sold_leads = 0
                        conversion_rate = 0
                    
                    # Calculate average time to close
                    if "time_to_closed" in group.columns:
                        avg_close_time = group["time_to_closed"].mean()
                    else:
                        avg_close_time = None
                    
                    # Store metrics
                    model_metrics[str(model)] = {
                        "total_leads": total_leads,
                        "closed_leads": sold_leads,
                        "conversion_rate": conversion_rate,
                        "avg_days_to_close": avg_close_time
                    }
                    
                    # Calculate bottlenecks specific to this model
                    bottlenecks = {}
                    for i in range(1, len(self.stages)):
                        prev_stage = self.stages[i-1]
                        curr_stage = self.stages[i]
                        
                        time_col = f"time_{prev_stage}_to_{curr_stage}"
                        
                        if time_col in group.columns:
                            avg_time = group[time_col].mean()
                            if avg_time is not None:
                                threshold = self.thresholds.get(time_col, self.thresholds.get(f"time_to_{curr_stage}", 7.0))
                                is_bottleneck = avg_time > threshold
                                
                                bottlenecks[time_col] = {
                                    "average_days": avg_time,
                                    "threshold": threshold,
                                    "is_bottleneck": is_bottleneck
                                }
                    
                    model_metrics[str(model)]["bottlenecks"] = bottlenecks
        
        return model_metrics
    
    def _update_stored_metrics(self, results: Dict[str, Any]) -> None:
        """
        Update the stored metrics with latest results.
        
        Args:
            results: Latest calculated metrics
        """
        # Update each section
        for section in ["bottlenecks", "outliers", "stage_metrics", "rep_metrics", "source_metrics", "model_metrics"]:
            if section in results:
                self.metrics_data[section] = results[section]
        
        # Save to storage
        self._save_metrics_data()
    
    def _find_rep_column(self, df: pd.DataFrame) -> Optional[str]:
        """
        Find the sales rep column in the DataFrame.
        
        Args:
            df: DataFrame to search
            
        Returns:
            Column name if found, None otherwise
        """
        possible_names = ["SalesRep", "Rep", "RepName", "AssignedTo", "Owner", "LeadOwner", "AgentName"]
        
        for name in possible_names:
            if name in df.columns:
                return name
        
        return None
    
    def _find_source_column(self, df: pd.DataFrame) -> Optional[str]:
        """
        Find the lead source column in the DataFrame.
        
        Args:
            df: DataFrame to search
            
        Returns:
            Column name if found, None otherwise
        """
        possible_names = ["LeadSource", "Source", "TrafficSource", "Channel", "LeadOrigin"]
        
        for name in possible_names:
            if name in df.columns:
                return name
        
        return None
    
    def _find_model_column(self, df: pd.DataFrame) -> Optional[str]:
        """
        Find the vehicle model column in the DataFrame.
        
        Args:
            df: DataFrame to search
            
        Returns:
            Column name if found, None otherwise
        """
        possible_names = ["Model", "VehicleModel", "CarModel", "AutoModel", "VehModel"]
        
        for name in possible_names:
            if name in df.columns:
                return name
        
        return None
    
    def get_bottlenecks(self) -> Dict[str, Any]:
        """
        Get the current bottlenecks.
        
        Returns:
            Dictionary with bottleneck data
        """
        return self.metrics_data.get("bottlenecks", {})
    
    def get_outliers(self) -> Dict[str, Any]:
        """
        Get the current outliers.
        
        Returns:
            Dictionary with outlier data
        """
        return self.metrics_data.get("outliers", {})
    
    def get_aged_leads(self, threshold: Optional[float] = None) -> Dict[str, Any]:
        """
        Get data on aged leads.
        
        Args:
            threshold: Optional custom age threshold in days
            
        Returns:
            Dictionary with aged lead data
        """
        outliers = self.metrics_data.get("outliers", {})
        aged_leads = outliers.get("aged_leads", {})
        
        # If a custom threshold is specified and aged_leads has lead data,
        # filter the leads based on the custom threshold
        if threshold is not None and "leads" in aged_leads:
            leads = aged_leads.get("leads", [])
            filtered_leads = [lead for lead in leads if lead.get("lead_age", 0) > threshold]
            
            # Return a copy with filtered leads
            result = aged_leads.copy()
            result["leads"] = filtered_leads
            result["threshold_days"] = threshold
            result["count"] = len(filtered_leads)
            
            return result
        
        return aged_leads
    
    def get_rep_performance(self) -> Dict[str, Any]:
        """
        Get performance metrics by sales rep.
        
        Returns:
            Dictionary with rep performance data
        """
        return self.metrics_data.get("rep_metrics", {})
    
    def get_source_performance(self) -> Dict[str, Any]:
        """
        Get performance metrics by lead source.
        
        Returns:
            Dictionary with source performance data
        """
        return self.metrics_data.get("source_metrics", {})
    
    def get_model_performance(self) -> Dict[str, Any]:
        """
        Get performance metrics by vehicle model.
        
        Returns:
            Dictionary with model performance data
        """
        return self.metrics_data.get("model_metrics", {})
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of lead flow metrics.
        
        Returns:
            Dictionary with summary metrics
        """
        # Get components
        bottlenecks = self.get_bottlenecks()
        outliers = self.get_outliers()
        reps = self.get_rep_performance()
        sources = self.get_source_performance()
        
        # Identify top bottlenecks
        top_bottlenecks = []
        for name, data in bottlenecks.items():
            if data.get("is_bottleneck", False):
                top_bottlenecks.append({
                    "stage": name,
                    "average_days": data.get("average_days"),
                    "threshold": data.get("threshold")
                })
        
        # Sort bottlenecks by severity (difference from threshold)
        top_bottlenecks.sort(key=lambda x: (x.get("average_days", 0) / x.get("threshold", 1)) if x.get("threshold", 0) > 0 else 0, reverse=True)
        
        # Get aged lead count
        aged_leads_count = outliers.get("aged_leads", {}).get("count", 0)
        
        # Calculate overall conversion rate
        stage_metrics = self.metrics_data.get("stage_metrics", {})
        created_count = stage_metrics.get("created", {}).get("count", 0)
        closed_count = stage_metrics.get("closed", {}).get("count", 0)
        
        if created_count > 0:
            overall_conversion = (closed_count / created_count) * 100
        else:
            overall_conversion = 0
        
        # Find best and worst performing reps
        rep_performance = []
        for rep, data in reps.items():
            if data.get("total_leads", 0) >= 10:  # Minimum threshold to be considered
                rep_performance.append({
                    "name": rep,
                    "conversion_rate": data.get("conversion_rate", 0),
                    "total_leads": data.get("total_leads", 0),
                    "closed_leads": data.get("closed_leads", 0)
                })
        
        # Sort by conversion rate
        rep_performance.sort(key=lambda x: x.get("conversion_rate", 0), reverse=True)
        best_reps = rep_performance[:3] if len(rep_performance) >= 3 else rep_performance
        worst_reps = rep_performance[-3:] if len(rep_performance) >= 3 else rep_performance
        worst_reps.reverse()  # Ascending order
        
        # Find best and worst lead sources
        source_performance = []
        for source, data in sources.items():
            if data.get("total_leads", 0) >= 5:  # Minimum threshold to be considered
                source_performance.append({
                    "name": source,
                    "conversion_rate": data.get("conversion_rate", 0),
                    "total_leads": data.get("total_leads", 0),
                    "closed_leads": data.get("closed_leads", 0)
                })
        
        # Sort by conversion rate
        source_performance.sort(key=lambda x: x.get("conversion_rate", 0), reverse=True)
        best_sources = source_performance[:3] if len(source_performance) >= 3 else source_performance
        worst_sources = source_performance[-3:] if len(source_performance) >= 3 else source_performance
        worst_sources.reverse()  # Ascending order
        
        return {
            "overall_conversion_rate": overall_conversion,
            "aged_leads_count": aged_leads_count,
            "top_bottlenecks": top_bottlenecks[:3],
            "best_performing_reps": best_reps,
            "worst_performing_reps": worst_reps,
            "best_performing_sources": best_sources,
            "worst_performing_sources": worst_sources,
            "last_updated": self.metrics_data.get("updated_at", datetime.now().isoformat())
        }


class LeadFlowOptimizer:
    """Lead flow optimization and analysis."""
    
    def __init__(self, 
                 metrics: Optional[LeadFlowMetrics] = None,
                 thresholds: Optional[Dict[str, float]] = None,
                 custom_stages: Optional[List[str]] = None,
                 storage_path: Optional[str] = None):
        """
        Initialize the lead flow optimizer.
        
        Args:
            metrics: Optional pre-initialized metrics object
            thresholds: Optional custom thresholds for bottleneck detection
            custom_stages: Optional custom lead stages
            storage_path: Optional path to store data
        """
        # Set up metrics tracker
        self.metrics = metrics or LeadFlowMetrics(
            thresholds=thresholds,
            custom_stages=custom_stages,
            storage_path=storage_path
        )
        
        # Store configuration
        self.thresholds = thresholds or DEFAULT_BOTTLENECK_THRESHOLDS.copy()
        self.stages = custom_stages or DEFAULT_STAGES.copy()
        self.storage_path = storage_path
    
    def process_lead_data(self, 
                         df: pd.DataFrame, 
                         id_col: str = "LeadID",
                         stage_date_format: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Process lead data and update metrics.
        
        Args:
            df: DataFrame with lead data
            id_col: Column with lead IDs
            stage_date_format: Optional mapping of stage columns to date formats
            
        Returns:
            Dictionary with processed metrics
        """
        return self.metrics.process_lead_data(df, id_col, stage_date_format)
    
    def identify_bottlenecks(self) -> Dict[str, Any]:
        """
        Identify current bottlenecks in the lead flow.
        
        Returns:
            Dictionary with identified bottlenecks
        """
        return self.metrics.get_bottlenecks()
    
    def flag_aged_leads(self, threshold: Optional[float] = None) -> Dict[str, Any]:
        """
        Flag leads that have been in the pipeline too long.
        
        Args:
            threshold: Optional custom age threshold in days
            
        Returns:
            Dictionary with aged lead data
        """
        return self.metrics.get_aged_leads(threshold)
    
    def get_rep_performance(self) -> Dict[str, Dict[str, Any]]:
        """
        Get performance metrics by sales rep.
        
        Returns:
            Dictionary with rep performance data
        """
        return self.metrics.get_rep_performance()
    
    def get_source_performance(self) -> Dict[str, Dict[str, Any]]:
        """
        Get performance metrics by lead source.
        
        Returns:
            Dictionary with source performance data
        """
        return self.metrics.get_source_performance()
    
    def get_model_performance(self) -> Dict[str, Dict[str, Any]]:
        """
        Get performance metrics by vehicle model.
        
        Returns:
            Dictionary with model performance data
        """
        return self.metrics.get_model_performance()
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of lead flow optimization insights.
        
        Returns:
            Dictionary with summary metrics and insights
        """
        return self.metrics.get_summary()
    
    def generate_recommendations(self) -> List[Dict[str, Any]]:
        """
        Generate actionable recommendations based on current metrics.
        
        Returns:
            List of recommendation dictionaries
        """
        recommendations = []
        
        # Get current metrics
        bottlenecks = self.metrics.get_bottlenecks()
        outliers = self.metrics.get_outliers()
        reps = self.metrics.get_rep_performance()
        sources = self.metrics.get_source_performance()
        summary = self.metrics.get_summary()
        
        # Check for bottlenecks
        for name, data in bottlenecks.items():
            if data.get("is_bottleneck", False):
                stage_name = name.replace("time_", "").replace("_to_", " to ").title()
                recommendations.append({
                    "type": "bottleneck",
                    "priority": "high",
                    "title": f"Bottleneck in {stage_name} Stage",
                    "description": f"Average time of {data.get('average_days', 0):.1f} days exceeds threshold of {data.get('threshold', 0):.1f} days",
                    "action": f"Review the {stage_name} process to identify and address causes of delay"
                })
        
        # Check for aged leads
        aged_leads = outliers.get("aged_leads", {})
        if aged_leads.get("count", 0) > 0:
            recommendations.append({
                "type": "aged_leads",
                "priority": "high",
                "title": f"Aged Leads Requiring Attention",
                "description": f"{aged_leads.get('count', 0)} leads older than {aged_leads.get('threshold_days', 30)} days",
                "action": "Review aged leads and determine appropriate action (reactivate or close)"
            })
        
        # Check for underperforming reps
        if "worst_performing_reps" in summary:
            for rep in summary["worst_performing_reps"]:
                if rep.get("conversion_rate", 0) < 10 and rep.get("total_leads", 0) >= 10:
                    recommendations.append({
                        "type": "rep_performance",
                        "priority": "medium",
                        "title": f"Low Conversion Rate for {rep.get('name', 'Sales Rep')}",
                        "description": f"Conversion rate of {rep.get('conversion_rate', 0):.1f}% across {rep.get('total_leads', 0)} leads",
                        "action": f"Provide coaching or training to {rep.get('name', 'Sales Rep')} on lead management"
                    })
        
        # Check for underperforming sources
        if "worst_performing_sources" in summary:
            for source in summary["worst_performing_sources"]:
                if source.get("conversion_rate", 0) < 5 and source.get("total_leads", 0) >= 10:
                    recommendations.append({
                        "type": "source_performance",
                        "priority": "medium",
                        "title": f"Low Conversion Rate for {source.get('name', 'Lead Source')}",
                        "description": f"Conversion rate of {source.get('conversion_rate', 0):.1f}% across {source.get('total_leads', 0)} leads",
                        "action": f"Review lead quality and follow-up process for {source.get('name', 'Lead Source')}"
                    })
        
        # Add general recommendations if few specific ones
        if len(recommendations) < 2:
            recommendations.append({
                "type": "general",
                "priority": "low",
                "title": "Regular Lead Flow Review",
                "description": "Implement regular review of lead flow metrics",
                "action": "Schedule weekly review of lead aging and conversion rates"
            })
        
        return recommendations


# Utility functions for working with lead flow data

def prepare_lead_data(df: pd.DataFrame, 
                     date_mappings: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    """
    Prepare raw lead data for lead flow analysis.
    
    Args:
        df: Raw DataFrame with lead data
        date_mappings: Optional mapping of raw column names to stage date columns
        
    Returns:
        Prepared DataFrame ready for lead flow analysis
    """
    # Create a working copy
    work_df = df.copy()
    
    # Set up default date mappings if not provided
    if date_mappings is None:
        date_mappings = {
            "CreatedDate": "created_date",
            "FirstContactDate": "contacted_date",
            "AppointmentDate": "appointment_date",
            "TestDriveDate": "test_drive_date",
            "OfferDate": "offer_date",
            "SoldDate": "sold_date",
            "DeliveryDate": "delivered_date",
            "ClosedDate": "closed_date"
        }
    
    # Rename columns based on mappings
    for source_col, target_col in date_mappings.items():
        if source_col in work_df.columns:
            work_df[target_col] = work_df[source_col]
    
    # Ensure all required columns exist
    required_columns = ["LeadID", "created_date"]
    for col in required_columns:
        if col not in work_df.columns:
            # Try to create LeadID if missing
            if col == "LeadID" and "ID" in work_df.columns:
                work_df["LeadID"] = work_df["ID"]
            # Use default for created_date if missing
            elif col == "created_date" and not any(c.lower().endswith("created_date") for c in work_df.columns):
                work_df["created_date"] = datetime.now()
    
    return work_df


def load_test_data() -> pd.DataFrame:
    """
    Load sample test data for lead flow analysis.
    
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
        
        # Determine if lead was contacted
        contacted = created + timedelta(hours=np.random.randint(1, 48)) if np.random.random() < 0.9 else None
        
        # Determine if appointment was scheduled
        appointment = contacted + timedelta(days=np.random.randint(1, 5)) if contacted and np.random.random() < 0.7 else None
        
        # Determine if test drive occurred
        test_drive = appointment + timedelta(days=np.random.randint(0, 3)) if appointment and np.random.random() < 0.8 else None
        
        # Determine if offer was made
        offer = test_drive + timedelta(days=np.random.randint(0, 5)) if test_drive and np.random.random() < 0.6 else None
        
        # Determine if sold
        sold = offer + timedelta(days=np.random.randint(1, 7)) if offer and np.random.random() < 0.7 else None
        
        # Determine if delivered
        delivered = sold + timedelta(days=np.random.randint(1, 14)) if sold and np.random.random() < 0.9 else None
        
        # Determine if closed
        closed = delivered + timedelta(days=np.random.randint(1, 10)) if delivered and np.random.random() < 0.95 else None
        
        # Add to data
        data.append({
            "LeadID": lead_id,
            "LeadSource": source,
            "SalesRep": rep,
            "Model": model,
            "created_date": created,
            "contacted_date": contacted,
            "appointment_date": appointment,
            "test_drive_date": test_drive,
            "offer_date": offer,
            "sold_date": sold,
            "delivered_date": delivered,
            "closed_date": closed
        })
    
    return pd.DataFrame(data)


def run_test():
    """Run a test of the lead flow optimizer with sample data."""
    # Load sample data
    df = load_test_data()
    
    # Create optimizer
    optimizer = LeadFlowOptimizer()
    
    # Process data
    results = optimizer.process_lead_data(df)
    
    # Get summary
    summary = optimizer.get_summary()
    
    # Generate recommendations
    recommendations = optimizer.generate_recommendations()
    
    # Print results
    print("Lead Flow Optimization Test")
    print("==========================")
    
    print(f"\nProcessed {len(df)} leads")
    print(f"Overall conversion rate: {summary.get('overall_conversion_rate', 0):.1f}%")
    print(f"Aged leads: {summary.get('aged_leads_count', 0)}")
    
    print("\nTop Bottlenecks:")
    for bottleneck in summary.get('top_bottlenecks', []):
        print(f"- {bottleneck.get('stage')}: {bottleneck.get('average_days', 0):.1f} days " +
              f"(threshold: {bottleneck.get('threshold', 0):.1f} days)")
    
    print("\nRecommendations:")
    for rec in recommendations:
        print(f"- [{rec.get('priority', 'low')}] {rec.get('title')}: {rec.get('action')}")
    
    return results


if __name__ == "__main__":
    run_test()