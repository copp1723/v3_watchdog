"""
Feedback Analyzer and Heatmap Generator for Watchdog AI.

This module provides functionality for analyzing feedback data and generating
heatmaps to visualize feedback patterns.
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import plotly.express as px

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class FeedbackAnalysis:
    """Represents analyzed feedback data."""
    insight_id: str
    persona: str
    feedback_type: str
    metrics: Dict[str, float]
    clusters: List[Dict[str, Any]]
    timestamp: str

class FeedbackAnalyzer:
    """
    Analyzes feedback data and generates heatmaps.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the feedback analyzer.
        
        Args:
            model_path: Path to pre-trained model (optional)
        """
        self.model_path = model_path
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=3, random_state=42)
        
        if model_path and os.path.exists(model_path):
            self._load_model()
    
    def _load_model(self) -> None:
        """Load pre-trained model from file."""
        try:
            model_data = np.load(self.model_path, allow_pickle=True)
            self.scaler = model_data['scaler'].item()
            self.kmeans = model_data['kmeans'].item()
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def analyze_feedback(
        self,
        feedback_data: List[Dict[str, Any]],
        insight_id: str,
        persona: str
    ) -> FeedbackAnalysis:
        """
        Analyze feedback data for an insight.
        
        Args:
            feedback_data: List of feedback entries
            insight_id: ID of the insight
            persona: User persona
            
        Returns:
            FeedbackAnalysis object
        """
        # Convert feedback data to DataFrame
        df = pd.DataFrame(feedback_data)
        
        # Calculate basic metrics
        metrics = self._calculate_metrics(df)
        
        # Perform clustering analysis
        clusters = self._cluster_feedback(df)
        
        return FeedbackAnalysis(
            insight_id=insight_id,
            persona=persona,
            feedback_type=df['feedback_type'].iloc[0] if not df.empty else "unknown",
            metrics=metrics,
            clusters=clusters,
            timestamp=datetime.now().isoformat()
        )
    
    def _calculate_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate feedback metrics."""
        metrics = {}
        
        if df.empty:
            return metrics
        
        # Calculate average rating
        if 'rating' in df.columns:
            metrics['avg_rating'] = df['rating'].mean()
            metrics['rating_std'] = df['rating'].std()
        
        # Calculate thumbs up percentage
        if 'thumbs_up' in df.columns:
            metrics['thumbs_up_pct'] = (df['thumbs_up'].sum() / len(df)) * 100
        
        # Calculate sentiment scores for comments
        if 'comment' in df.columns:
            # In a real implementation, this would use a sentiment analysis model
            metrics['sentiment_score'] = 0.0
        
        return metrics
    
    def _cluster_feedback(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Cluster feedback data to identify patterns."""
        if df.empty:
            return []
        
        # Prepare features for clustering
        features = []
        for _, row in df.iterrows():
            feature_vector = []
            
            # Add rating if available
            if 'rating' in row:
                feature_vector.append(row['rating'])
            
            # Add thumbs up/down if available
            if 'thumbs_up' in row:
                feature_vector.append(1 if row['thumbs_up'] else 0)
            
            # Add sentiment score if available
            if 'sentiment_score' in row:
                feature_vector.append(row['sentiment_score'])
            
            features.append(feature_vector)
        
        if not features:
            return []
        
        # Scale features
        features_scaled = self.scaler.fit_transform(features)
        
        # Perform clustering
        clusters = self.kmeans.fit_predict(features_scaled)
        
        # Analyze clusters
        cluster_analysis = []
        for i in range(self.kmeans.n_clusters):
            cluster_data = df[clusters == i]
            cluster_analysis.append({
                "cluster_id": i,
                "size": len(cluster_data),
                "centroid": self.kmeans.cluster_centers_[i].tolist(),
                "metrics": self._calculate_metrics(cluster_data)
            })
        
        return cluster_analysis
    
    def generate_heatmap(
        self,
        analysis: FeedbackAnalysis,
        output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate a heatmap visualization of feedback patterns.
        
        Args:
            analysis: FeedbackAnalysis object
            output_path: Path to save the heatmap (optional)
            
        Returns:
            Dictionary containing heatmap data
        """
        # Create heatmap data
        heatmap_data = {
            "insight_id": analysis.insight_id,
            "persona": analysis.persona,
            "timestamp": analysis.timestamp,
            "metrics": analysis.metrics,
            "clusters": analysis.clusters,
            "visualization": {
                "type": "heatmap",
                "data": self._prepare_heatmap_data(analysis),
                "layout": self._prepare_heatmap_layout(analysis)
            }
        }
        
        # Save heatmap if output path is provided
        if output_path:
            self._save_heatmap(heatmap_data, output_path)
        
        return heatmap_data
    
    def _prepare_heatmap_data(self, analysis: FeedbackAnalysis) -> List[List[float]]:
        """Prepare data for heatmap visualization."""
        # Create a 2D grid of feedback intensity
        grid_size = 10
        heatmap_data = np.zeros((grid_size, grid_size))
        
        # Fill grid based on cluster centroids
        for cluster in analysis.clusters:
            centroid = cluster["centroid"]
            x = int(centroid[0] * (grid_size - 1))
            y = int(centroid[1] * (grid_size - 1))
            intensity = cluster["size"] / sum(c["size"] for c in analysis.clusters)
            heatmap_data[y, x] = intensity
        
        return heatmap_data.tolist()
    
    def _prepare_heatmap_layout(self, analysis: FeedbackAnalysis) -> Dict[str, Any]:
        """Prepare layout for heatmap visualization."""
        return {
            "title": f"Feedback Heatmap - {analysis.insight_id}",
            "xaxis": {"title": "Rating"},
            "yaxis": {"title": "Sentiment"},
            "annotations": [
                {
                    "text": f"Persona: {analysis.persona}",
                    "x": 0.5,
                    "y": 1.1,
                    "showarrow": False
                }
            ]
        }
    
    def _save_heatmap(self, heatmap_data: Dict[str, Any], output_path: str) -> None:
        """Save heatmap visualization to file."""
        try:
            # Create heatmap figure
            fig = go.Figure(data=go.Heatmap(
                z=heatmap_data["visualization"]["data"],
                colorscale="RdYlBu"
            ))
            
            # Update layout
            fig.update_layout(**heatmap_data["visualization"]["layout"])
            
            # Save figure
            fig.write_html(output_path)
        except Exception as e:
            logger.error(f"Error saving heatmap: {e}")
            raise
    
    def get_analysis_by_insight(self, insight_id: str) -> Optional[FeedbackAnalysis]:
        """
        Get feedback analysis for an insight.
        
        Args:
            insight_id: ID of the insight
            
        Returns:
            FeedbackAnalysis object or None if not found
        """
        # In a real implementation, this would query a database
        # For now, we'll return None as analyses are not persisted
        return None 