import pandas as pd
from typing import Dict, Any, Optional
from ..models.query_models import IntentSchema, QueryContext, QueryResult

class InsightProcessor:
    def __init__(self):
        self.df = None

    def set_data(self, df: pd.DataFrame) -> None:
        """Set the data to be processed."""
        self.df = df

    def process_query(self, context: QueryContext) -> QueryResult:
        """Process a query with intent and parameters."""
        try:
            if self.df is None:
                return QueryResult(
                    status="error",
                    message="No data available for processing",
                    data=None
                )

            if context.intent is None:
                return QueryResult(
                    status="error",
                    message="No intent specified in query context",
                    data=None
                )

            if context.intent.intent == "performance_analysis":
                return self._process_performance_analysis(context)
            
            return QueryResult(
                status="error",
                message=f"Unsupported intent: {context.intent.intent}",
                data=None
            )

        except Exception as e:
            return QueryResult(
                status="error",
                message=f"Error processing query: {str(e)}",
                data=None
            )

    def _process_performance_analysis(self, context: QueryContext) -> QueryResult:
        """Process performance analysis intent."""
        try:
            if not context.intent.category or not context.intent.metric:
                return QueryResult(
                    status="error",
                    message="Category and metric are required for performance analysis",
                    data=None
                )

            # Group by category and aggregate metric
            agg_func = context.intent.aggregation or 'sum'
            grouped = self.df.groupby(context.intent.category)[context.intent.metric].agg(agg_func)

            # Sort if specified
            if context.intent.sort_order:
                ascending = context.intent.sort_order.lower() == 'asc'
                grouped = grouped.sort_values(ascending=ascending)

            # Apply limit if specified
            if context.intent.limit:
                grouped = grouped.head(context.intent.limit)

            return QueryResult(
                status="success",
                message="Performance analysis completed successfully",
                data={
                    "results": grouped.to_dict(),
                    "metadata": {
                        "category": context.intent.category,
                        "metric": context.intent.metric,
                        "aggregation": agg_func
                    }
                }
            )

        except Exception as e:
            return QueryResult(
                status="error",
                message=f"Error in performance analysis: {str(e)}",
                data=None
            )

    def _process_groupby_summary(self, context: QueryContext) -> QueryResult:
        # ... existing code ...
        pass
        
    def _process_total_summary(self, context: QueryContext) -> QueryResult:
        # ... existing code ...
        pass 