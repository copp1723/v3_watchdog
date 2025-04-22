"""
Feedback collection and analysis engine for Watchdog AI.
"""

import json
import os
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import logging
from collections import defaultdict, Counter
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class FeedbackEngine:
    """Handles collection and analysis of user feedback."""
    
    def __init__(self, feedback_dir: Optional[str] = None):
        """
        Initialize FeedbackEngine.
        
        Args:
            feedback_dir: Optional directory for feedback storage
        """
        self.feedback_dir = feedback_dir or os.path.join(
            os.path.dirname(__file__),
            "..",
            "data",
            "feedback"
        )
        self.feedback_history = []
        self.analysis_cache = {}
        self._ensure_feedback_dir()
        
    def _ensure_feedback_dir(self) -> None:
        """Ensure feedback directory exists."""
        try:
            os.makedirs(self.feedback_dir, exist_ok=True)
        except Exception as e:
            logger.error(f"Error creating feedback directory: {e}")
            
    def add_feedback(self,
                    query: str,
                    rating: int,
                    comment: Optional[str] = None,
                    metadata: Optional[Dict] = None) -> bool:
        """
        Add user feedback.
        
        Args:
            query: The query that was executed
            rating: User rating (1-5)
            comment: Optional user comment
            metadata: Optional metadata about the query/response
            
        Returns:
            bool: Success status
        """
        try:
            # Validate rating
            if not isinstance(rating, int) or rating < 1 or rating > 5:
                logger.error(f"Invalid rating: {rating}")
                return False
                
            # Create feedback entry
            feedback = {
                "query": query,
                "rating": rating,
                "comment": comment,
                "metadata": metadata or {},
                "timestamp": datetime.now().isoformat()
            }
            
            # Add to history
            self.feedback_history.append(feedback)
            
            # Save to file
            self._save_feedback(feedback)
            
            # Clear analysis cache
            self.analysis_cache = {}
            
            return True
            
        except Exception as e:
            logger.error(f"Error adding feedback: {e}")
            return False
            
    def _save_feedback(self, feedback: Dict[str, Any]) -> bool:
        """Save feedback to file system."""
        try:
            # Generate filename based on timestamp
            timestamp = datetime.fromisoformat(feedback["timestamp"])
            filename = f"feedback_{timestamp.strftime('%Y%m%d_%H%M%S')}.json"
            filepath = os.path.join(self.feedback_dir, filename)
            
            with open(filepath, 'w') as f:
                json.dump(feedback, f, indent=2)
                
            return True
            
        except Exception as e:
            logger.error(f"Error saving feedback: {e}")
            return False
            
    def load_feedback_history(self) -> bool:
        """Load all feedback from directory."""
        try:
            self.feedback_history = []
            
            # List all feedback files
            for filename in os.listdir(self.feedback_dir):
                if not filename.endswith('.json'):
                    continue
                    
                filepath = os.path.join(self.feedback_dir, filename)
                try:
                    with open(filepath, 'r') as f:
                        feedback = json.load(f)
                        self.feedback_history.append(feedback)
                except Exception as e:
                    logger.error(f"Error loading feedback file {filename}: {e}")
                    
            # Sort by timestamp
            self.feedback_history.sort(
                key=lambda x: x["timestamp"]
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading feedback history: {e}")
            return False
            
    def get_feedback_stats(self) -> Dict[str, Any]:
        """Get statistical analysis of feedback."""
        try:
            if not self.feedback_history:
                return {}
                
            # Convert to DataFrame for analysis
            df = pd.DataFrame(self.feedback_history)
            
            stats = {
                "total_feedback": len(df),
                "average_rating": float(df["rating"].mean()),
                "rating_distribution": df["rating"].value_counts().to_dict(),
                "feedback_trend": self._calculate_rating_trend(df),
                "common_themes": self._analyze_comments(df),
                "timestamp": datetime.now().isoformat()
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error calculating feedback stats: {e}")
            return {}
            
    def _calculate_rating_trend(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Calculate rating trend over time."""
        try:
            df['date'] = pd.to_datetime(df['timestamp'])
            daily_avg = df.groupby(df['date'].dt.date)['rating'].agg(['mean', 'count'])
            
            return [
                {
                    "date": date.isoformat(),
                    "average_rating": float(row["mean"]),
                    "feedback_count": int(row["count"])
                }
                for date, row in daily_avg.iterrows()
            ]
            
        except Exception as e:
            logger.error(f"Error calculating rating trend: {e}")
            return []
            
    def _analyze_comments(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze feedback comments for common themes."""
        try:
            comments = df[df["comment"].notna()]
            
            # Simple word frequency analysis
            word_freq = defaultdict(int)
            for comment in comments["comment"]:
                words = comment.lower().split()
                for word in words:
                    if len(word) > 3:  # Skip short words
                        word_freq[word] += 1
                        
            # Get top themes
            top_themes = sorted(
                word_freq.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]
            
            return {
                "word_frequency": dict(top_themes),
                "total_comments": len(comments),
                "comment_rate": len(comments) / len(df) * 100
            }
            
        except Exception as e:
            logger.error(f"Error analyzing comments: {e}")
            return {}
            
    def get_low_rating_queries(self, 
                             threshold: int = 3,
                             min_feedback: int = 2) -> List[Dict[str, Any]]:
        """
        Get queries with consistently low ratings.
        
        Args:
            threshold: Rating threshold (below this is considered low)
            min_feedback: Minimum number of feedback entries required
            
        Returns:
            List of problematic queries with stats
        """
        try:
            if not self.feedback_history:
                return []
                
            # Group by query
            query_stats = defaultdict(list)
            for feedback in self.feedback_history:
                query_stats[feedback["query"]].append(feedback["rating"])
                
            # Find problematic queries
            problems = []
            for query, ratings in query_stats.items():
                if len(ratings) >= min_feedback:
                    avg_rating = sum(ratings) / len(ratings)
                    if avg_rating < threshold:
                        problems.append({
                            "query": query,
                            "average_rating": avg_rating,
                            "feedback_count": len(ratings),
                            "rating_distribution": Counter(ratings)
                        })
                        
            return sorted(problems, key=lambda x: x["average_rating"])
            
        except Exception as e:
            logger.error(f"Error finding problematic queries: {e}")
            return []
            
    def export_feedback_report(self, filepath: str) -> bool:
        """
        Export comprehensive feedback report.
        
        Args:
            filepath: Path to save report
            
        Returns:
            bool: Success status
        """
        try:
            report = {
                "statistics": self.get_feedback_stats(),
                "low_rating_queries": self.get_low_rating_queries(),
                "raw_feedback": self.feedback_history,
                "generated_at": datetime.now().isoformat()
            }
            
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2)
                
            return True
            
        except Exception as e:
            logger.error(f"Error exporting feedback report: {e}")
            return False 