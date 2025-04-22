"""
Data pipeline with event integration.
"""

import logging
from typing import Dict, Any, Optional
import pandas as pd
from datetime import datetime

from .event_emitter import EventEmitter, Event, EventType
from ..utils.data_validation import validate_data
from ..utils.data_normalization import normalize_data
from .insight_generator import generate_insights

logger = logging.getLogger(__name__)

class DataPipeline:
    """Data processing pipeline with event integration."""
    
    def __init__(self):
        """Initialize the pipeline."""
        self.event_emitter = EventEmitter()
    
    def process_data(self, data: pd.DataFrame, dealer_id: str,
                    metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process data through the pipeline.
        
        Args:
            data: DataFrame to process
            dealer_id: Dealer ID
            metadata: Optional metadata
            
        Returns:
            Processing results
        """
        try:
            logger.info(f"Processing data for dealer {dealer_id}")
            
            # Validate data
            validation_result = validate_data(data)
            if not validation_result['valid']:
                logger.error(f"Data validation failed: {validation_result['errors']}")
                return {
                    'success': False,
                    'errors': validation_result['errors']
                }
            
            # Normalize data
            normalized_data = normalize_data(data)
            
            # Generate insights
            insights = generate_insights(normalized_data)
            
            # Emit data normalized event
            self.event_emitter.emit(Event(
                event_type=EventType.DATA_NORMALIZED,
                data={
                    'dealer_id': dealer_id,
                    'insights': insights,
                    'metadata': metadata or {}
                },
                source='data_pipeline'
            ))
            
            # Check for alerts
            self._check_alerts(normalized_data, dealer_id)
            
            return {
                'success': True,
                'data': normalized_data,
                'insights': insights
            }
            
        except Exception as e:
            logger.error(f"Error processing data: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _check_alerts(self, data: pd.DataFrame, dealer_id: str) -> None:
        """
        Check for alert conditions.
        
        Args:
            data: Normalized DataFrame
            dealer_id: Dealer ID
        """
        try:
            # Check for negative gross deals
            if 'Gross_Profit' in data.columns:
                negative_gross = data[data['Gross_Profit'] < 0]
                if len(negative_gross) > 0:
                    self._emit_alert(
                        dealer_id=dealer_id,
                        title="Negative Gross Deals Detected",
                        description=f"Found {len(negative_gross)} deals with negative gross profit.",
                        severity="high" if len(negative_gross) > 5 else "medium",
                        metrics={
                            'count': len(negative_gross),
                            'total_impact': negative_gross['Gross_Profit'].sum()
                        }
                    )
            
            # Check for aging inventory
            if 'DaysInInventory' in data.columns:
                aged_units = data[data['DaysInInventory'] > 90]
                if len(aged_units) > 0:
                    self._emit_alert(
                        dealer_id=dealer_id,
                        title="Aging Inventory Alert",
                        description=f"Found {len(aged_units)} units over 90 days old.",
                        severity="high" if len(aged_units) > 10 else "medium",
                        metrics={
                            'count': len(aged_units),
                            'avg_age': aged_units['DaysInInventory'].mean()
                        }
                    )
            
        except Exception as e:
            logger.error(f"Error checking alerts: {e}")
    
    def _emit_alert(self, dealer_id: str, title: str, description: str,
                   severity: str, metrics: Dict[str, Any]) -> None:
        """
        Emit an alert event.
        
        Args:
            dealer_id: Dealer ID
            title: Alert title
            description: Alert description
            severity: Alert severity
            metrics: Alert metrics
        """
        self.event_emitter.emit(Event(
            event_type=EventType.ALERT_TRIGGERED,
            data={
                'dealer_id': dealer_id,
                'alert': {
                    'title': title,
                    'description': description,
                    'severity': severity,
                    'metrics': metrics,
                    'timestamp': datetime.now().isoformat()
                }
            },
            source='data_pipeline'
        ))