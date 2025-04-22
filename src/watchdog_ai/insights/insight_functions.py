import pandas as pd
import logging
from watchdog_ai.ui.utils.status_formatter import StatusType

# Configure logging
logger = logging.getLogger(__name__)

class InsightFunctions:
    """
    Collection of functions for generating insights from data.
    """
    
    def __init__(self):
        """Initialize the insight functions."""
        self.logger = logging.getLogger(__name__)
    
    def find_column(self, df, candidates):
        """
        Find a column in df matching any of the candidates (case/underscore/space insensitive).
        Returns the actual column name or None.
        """
        def normalize(col):
            return col.lower().replace("_", "").replace(" ", "")
        norm_cols = {normalize(col): col for col in df.columns}
        for candidate in candidates:
            key = normalize(candidate)
            if key in norm_cols:
                return norm_cols[key]
        return None

    def compute_lead_conversion_rate(self, df):
        lead_source_col = self.find_column(df, ["LeadSource", "lead_source", "Lead Source"])
        is_sale_col = self.find_column(df, ["IsSale", "is_sale", "Sold"])
        if not lead_source_col or not is_sale_col:
            self.logger.warning("Required columns not found for lead conversion rate calculation")
            return pd.DataFrame(columns=['LeadSource', 'total_leads', 'total_sales', 'conversion_rate'])
        
        try:
            df = df.copy()
            df[lead_source_col] = df[lead_source_col].fillna('Unknown')
            def to_sale(val):
                if pd.isna(val):
                    return 0
                if isinstance(val, (int, float)):
                    return int(val != 0)
                if isinstance(val, str):
                    return 1 if val.strip().lower() in ['1', 'true', 'yes', 'sold'] else 0
                if isinstance(val, bool):
                    return int(val)
                return 0
            df['_is_sale_norm'] = df[is_sale_col].apply(to_sale)
            grouped = df.groupby(lead_source_col)['_is_sale_norm'].agg(['count', 'sum']).reset_index()
            grouped.columns = ['LeadSource', 'total_leads', 'total_sales']
            grouped['conversion_rate'] = grouped.apply(
                lambda row: 100.0 * row['total_sales'] / row['total_leads'] if row['total_leads'] > 0 else 0.0,
                axis=1
            )
            grouped = grouped.sort_values(by='conversion_rate', ascending=False).reset_index(drop=True)
            return grouped
        except Exception as e:
            self.logger.error(f"Error in compute_lead_conversion_rate: {e}")
            return pd.DataFrame(columns=['LeadSource', 'total_leads', 'total_sales', 'conversion_rate'])

    def compute_gross_profit_summary(self, df):
        gross_profit_col = self.find_column(df, ["GrossProfit", "gross_profit", "Gross Profit"])
        if not gross_profit_col:
            return {"total_gross_profit": None, "average_gross_profit": None}
        total = df[gross_profit_col].sum()
        avg = df[gross_profit_col].mean()
        return {"total_gross_profit": total, "average_gross_profit": avg}

    def compute_salesperson_performance(self, df):
        salesperson_col = self.find_column(df, ["SalesPerson", "sales_person", "Sales Person"])
        is_sale_col = self.find_column(df, ["IsSale", "is_sale", "Sold"])
        gross_profit_col = self.find_column(df, ["GrossProfit", "gross_profit", "Gross Profit"])
        if not salesperson_col or not is_sale_col or not gross_profit_col:
            return pd.DataFrame(columns=['SalesPerson', 'total_sales', 'total_gross_profit'])
        df = df.copy()
        def to_sale(val):
            if pd.isna(val):
                return 0
            if isinstance(val, (int, float)):
                return int(val != 0)
            if isinstance(val, str):
                return 1 if val.strip().lower() in ['1', 'true', 'yes', 'sold'] else 0
            if isinstance(val, bool):
                return int(val)
            return 0
        df['_is_sale_norm'] = df[is_sale_col].apply(to_sale)
        grouped = df.groupby(salesperson_col).agg(
            total_sales=('_is_sale_norm', 'sum'),
            total_gross_profit=(gross_profit_col, 'sum')
        ).reset_index()
        grouped = grouped.sort_values(by='total_sales', ascending=False).reset_index(drop=True)
        grouped.rename(columns={salesperson_col: 'SalesPerson'}, inplace=True)
        return grouped

    def compute_total_sales(self, df):
        is_sale_col = self.find_column(df, ["IsSale", "is_sale", "Sold"])
        if not is_sale_col:
            return 0
        def to_sale(val):
            if pd.isna(val):
                return 0
            if isinstance(val, (int, float)):
                return int(val != 0)
            if isinstance(val, str):
                return 1 if val.strip().lower() in ['1', 'true', 'yes', 'sold'] else 0
            if isinstance(val, bool):
                return int(val)
            return 0
        return df[is_sale_col].apply(to_sale).sum()

    def compute_average_gross_profit_per_sale(self, df):
        gross_profit_col = self.find_column(df, ["GrossProfit", "gross_profit", "Gross Profit"])
        is_sale_col = self.find_column(df, ["IsSale", "is_sale", "Sold"])
        if not gross_profit_col or not is_sale_col:
            return None
        def to_sale(val):
            if pd.isna(val):
                return 0
            if isinstance(val, (int, float)):
                return int(val != 0)
            if isinstance(val, str):
                return 1 if val.strip().lower() in ['1', 'true', 'yes', 'sold'] else 0
            if isinstance(val, bool):
                return int(val)
            return 0
        sales_mask = df[is_sale_col].apply(to_sale) == 1
        if sales_mask.sum() == 0:
            return None
        return df.loc[sales_mask, gross_profit_col].mean()
        
    def groupby_summary(self, df, metric, category, aggregation="sum"):
        """
        Generate a summary by grouping data by a category and aggregating a metric.
        
        Args:
            df: The DataFrame to analyze
            metric: The metric column to aggregate
            category: The category column to group by
            aggregation: The aggregation method (sum, count, mean)
            
        Returns:
            A dictionary with summary information
        """
        if df.empty:
            self.logger.warning("Empty DataFrame provided for groupby_summary")
            return {
                "summary": "No data available for analysis.",
                "metrics": {},
                "breakdown": [],
                "recommendations": [],
                "confidence": "low",
                "error_type": "NO_DATA",
                "status_type": "WARNING"  # For use with StatusFormatter
            }

        # Find the actual column names
        category_col = self.find_column(df, [category])
        if not category_col:
            self.logger.warning(f"Could not find category column: {category}")
            return {
                "summary": f"Could not find category column: {category}",
                "metrics": {},
                "breakdown": [],
                "recommendations": [],
                "confidence": "low",
                "error_type": "COLUMN_NOT_FOUND",
                "status_type": "WARNING"  # For use with StatusFormatter
            }

        try:
            # Handle special case for IsSale metric
            if metric.lower() in ["issale", "is_sale", "sold", "sale", "numberofsales", "sales"]:
                # For sales analysis, count rows by category
                grouped = df.groupby(category_col).size().reset_index(name='value')
                total = len(df)
                
                # Sort by value descending
                grouped = grouped.sort_values('value', ascending=False)
                
                # Calculate percentages
                grouped['percentage'] = grouped['value'] / total * 100
                grouped['percentage'] = grouped['percentage'].apply(lambda x: f"{x:.1f}%")
                
                # Convert to list of dictionaries for JSON serialization
                breakdown = grouped.to_dict('records')
                
                # Find the top lead source
                top_source = breakdown[0] if breakdown else None
                summary = f"Analysis of sales by lead source shows that {top_source[category_col] if top_source else 'no'} leads produced the most sales with {top_source['value'] if top_source else 0} sales ({top_source['percentage'] if top_source else '0%'})."
                
                return {
                    "summary": summary,
                    "metrics": {"total_sales": total},
                    "breakdown": breakdown,
                    "recommendations": [],
                    "confidence": "high"
                }
            else:
                # Normal aggregation for other metrics
                metric_col = self.find_column(df, [metric])
                if not metric_col:
                    self.logger.warning(f"Could not find metric column: {metric}")
                    return {
                        "summary": f"Could not find metric column: {metric}",
                        "metrics": {},
                        "breakdown": [],
                        "recommendations": [],
                        "confidence": "low",
                        "error_type": "COLUMN_NOT_FOUND",
                        "status_type": "WARNING"  # For use with StatusFormatter
                    }

                # Clean numeric data and handle NaN values
                df = df.copy()
                df[metric_col] = pd.to_numeric(df[metric_col].astype(str).str.replace(r'[\$,]', '', regex=True), errors='coerce')
                
                # Count NaN values before dropping
                total_rows = len(df)
                nan_count = df[metric_col].isna().sum()
                nan_percentage = (nan_count / total_rows * 100) if total_rows > 0 else 0
                
                # Drop NaN values for aggregation
                df_clean = df.dropna(subset=[metric_col])
                
                if df_clean.empty:
                    return {
                        "summary": f"No valid data found for {metric} after removing missing values.",
                        "metrics": {},
                        "breakdown": [],
                        "recommendations": [],
                        "confidence": "low",
                        "error_type": "NO_VALID_DATA",
                        "status_type": "WARNING"  # For use with StatusFormatter
                    }

                # Perform aggregation
                if aggregation.lower() in ["sum", "count"]:
                    grouped = df_clean.groupby(category_col)[metric_col].agg(aggregation).reset_index()
                    grouped.columns = [category_col, 'value']
                    total = df_clean[metric_col].agg(aggregation)
                elif aggregation.lower() in ["mean", "avg", "average"]:
                    grouped = df_clean.groupby(category_col)[metric_col].mean().reset_index()
                    grouped.columns = [category_col, 'value']
                    total = df_clean[metric_col].mean()
                else:
                    # Default to sum
                    grouped = df_clean.groupby(category_col)[metric_col].sum().reset_index()
                    grouped.columns = [category_col, 'value']
                    total = df_clean[metric_col].sum()
                
                # Calculate percentages
                grouped['percentage'] = grouped['value'] / total * 100
                grouped['percentage'] = grouped['percentage'].apply(lambda x: f"{x:.1f}%")
                
                # Sort by value descending
                grouped = grouped.sort_values('value', ascending=False)
                
                # Convert to list of dictionaries for JSON serialization
                breakdown = grouped.to_dict('records')
                
                # Create summary with warning about missing data if significant
                summary_parts = [f"Breakdown of {metric} by {category}."]
                if nan_percentage > 10:  # Warning if more than 10% of data is missing
                    summary_parts.append(f"Note: {nan_count:,} rows ({nan_percentage:.1f}%) had missing or invalid values and were excluded.")
                
                return {
                    "summary": " ".join(summary_parts),
                    "metrics": {
                        f"total_{metric.lower()}": total,
                        "processed_rows": len(df_clean),
                        "excluded_rows": nan_count
                    },
                    "breakdown": breakdown,
                    "recommendations": [
                        "Review data quality and completeness" if nan_percentage > 10 else None,
                        "Consider data validation rules for required fields" if nan_percentage > 20 else None
                    ],
                    "confidence": "high" if nan_percentage < 10 else "medium"
                }
        except Exception as e:
            self.logger.error(f"Error in groupby_summary: {e}")
            return {
                "summary": f"An error occurred during analysis: {str(e)}",
                "metrics": {},
                "breakdown": [],
                "recommendations": [],
                "confidence": "low",
                "error_type": "PROCESSING_ERROR",
                "status_type": "ERROR"  # For use with StatusFormatter
            }
    
    def total_summary(self, df, metric):
        """
        Generate a summary of a metric across the entire dataset.
        
        Args:
            df: The DataFrame to analyze
            metric: The metric column to summarize
            
        Returns:
            A dictionary with summary information
        """
        if df.empty:
            self.logger.warning(f"Cannot summarize {metric}. Empty dataset.")
            return {
                "summary": f"Cannot summarize {metric}. Empty dataset.",
                "metrics": {},
                "breakdown": [],
                "recommendations": [],
                "confidence": "low",
                "error_type": "DATA_ERROR",
                "status_type": "WARNING"  # For use with StatusFormatter
            }
            
        # Check if metric exists in DataFrame
        if metric not in df.columns:
            # Try to find a similar column
            found_col = self.find_column(df, [metric])
            if found_col:
                metric = found_col
                self.logger.info(f"Using alternative column '{found_col}' for metric '{metric}'")
            else:
                self.logger.warning(f"Cannot summarize {metric}. Column not found.")
                return {
                    "summary": f"Cannot summarize {metric}. Column not found.",
                    "metrics": {},
                    "breakdown": [],
                    "recommendations": [],
                    "confidence": "low",
                    "error_type": "DATA_ERROR",
                    "status_type": "WARNING"  # For use with StatusFormatter
                }
        
        try:
            # Handle special case for IsSale metric
            if metric.lower() in ["issale", "is_sale", "sold", "sale", "numberofsales", "sales"]:
                # Use row count as the metric
                total = len(df)
            else:
                # Sum the metric
                total = df[metric].sum()
            
            # Format the summary based on the metric type
            if "price" in metric.lower() or "cost" in metric.lower() or "profit" in metric.lower() or "revenue" in metric.lower():
                formatted_total = f"${total:,.2f}"
            elif "percent" in metric.lower() or "percentage" in metric.lower() or "rate" in metric.lower():
                formatted_total = f"{total:.1f}%"
            else:
                formatted_total = f"{total:,}"
            
            return {
                "summary": f"Total {metric}: {formatted_total}.",
                "metrics": {f"total_{metric.lower()}": total},
                "breakdown": [],
                "recommendations": [],
                "confidence": "high"
            }
        except Exception as e:
            self.logger.error(f"Error in total_summary: {e}")
            return {
                "summary": f"{str(e)}",
                "metrics": {},
                "breakdown": [],
                "recommendations": [],
                "confidence": "low",
                "error_type": "DATA_ERROR",
                "status_type": "ERROR"  # For use with StatusFormatter
            }

# For backward compatibility
find_column = InsightFunctions().find_column
compute_lead_conversion_rate = InsightFunctions().compute_lead_conversion_rate
compute_gross_profit_summary = InsightFunctions().compute_gross_profit_summary
compute_salesperson_performance = InsightFunctions().compute_salesperson_performance
compute_total_sales = InsightFunctions().compute_total_sales
compute_average_gross_profit_per_sale = InsightFunctions().compute_average_gross_profit_per_sale

# Helper function to format status response using the StatusFormatter
def format_insight_status(response):
    """
    Format an insight response for display using StatusFormatter.
    
    Args:
        response: The insight response dictionary
        
    Returns:
        Updated response with formatted status text
    """
    if "status_type" in response:
        try:
            from watchdog_ai.ui.utils.status_formatter import format_status_text, StatusType
            status_type = getattr(StatusType, response["status_type"])
            response["formatted_summary"] = format_status_text(
                status_type, 
                custom_text=response["summary"]
            )
        except (ImportError, AttributeError) as e:
            # If StatusFormatter is not available, keep original summary
            pass
    return response
