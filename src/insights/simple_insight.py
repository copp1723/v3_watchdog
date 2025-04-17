"""
Simple insight generation module based on pandas operations.
"""

from typing import Dict, Optional, List, Any
import pandas as pd
from dataclasses import dataclass
import re
from datetime import datetime, timedelta

@dataclass
class InsightResult:
    """Structure for insight analysis results."""
    title: str
    summary: str
    metrics: List[Dict[str, Any]]
    chart_data: Optional[pd.DataFrame] = None
    recommendations: List[str] = None

    def __post_init__(self):
        if self.recommendations is None:
            self.recommendations = []

def analyze_sales_reps_by_deals(df: pd.DataFrame) -> InsightResult:
    """Analyze sales reps by number of deals."""
    # Ensure gross is numeric
    df = df.copy()  # Make a copy to avoid modifying original
    df['gross'] = pd.to_numeric(df['gross'].astype(str).str.replace(r'[\$,]', '', regex=True), errors='coerce')
    
    # Group by sales rep and count deals
    rep_stats = df.groupby('sales_rep').agg({
        'gross': {
            'deals': 'count',
            'total_gross': 'sum',
            'avg_gross': 'mean'
        }
    }).reset_index()
    
    # Flatten column names
    rep_stats.columns = ['sales_rep', 'deals', 'total_gross', 'avg_gross']
    
    # Sort by number of deals
    rep_stats = rep_stats.sort_values('deals', ascending=False)
    
    # Get top performer by deals
    top_rep = rep_stats.iloc[0]
    
    return InsightResult(
        title="Sales Representatives by Deal Count",
        summary=f"{top_rep['sales_rep']} leads with {int(top_rep['deals'])} deals totaling ${top_rep['total_gross']:,.2f}.",
        metrics=[{
            "rep": row['sales_rep'],
            "deals": int(row['deals']),
            "total_gross": f"${row['total_gross']:,.2f}",
            "avg_gross": f"${row['avg_gross']:,.2f}"
        } for _, row in rep_stats.iterrows()],
        chart_data=rep_stats,
        recommendations=[
            f"Study {top_rep['sales_rep']}'s lead conversion techniques",
            "Analyze deal velocity and closing ratios",
            "Share best practices for maintaining high deal volume"
        ]
    )

def query_insight(df_dict: Dict[str, pd.DataFrame], question: str) -> InsightResult:
    """
    Generate insights based on the question and available data.
    
    Args:
        df_dict: Dictionary mapping sheet names to DataFrames
        question: User's question
        
    Returns:
        InsightResult containing the analysis
    """
    question = question.lower().strip()
    
    # Ensure we have the sales sheet for most analyses
    if 'sales' not in df_dict:
        return InsightResult(
            title="Missing Data",
            summary="Sales data is required for this analysis.",
            metrics=[],
            recommendations=["Please upload a file containing sales data."]
        )
    
    sales_df = df_dict['sales']
    
    # Lead source analysis
    if "lead source" in question or "leads from" in question:
        source_pattern = r"from\s+(\w+)"
        match = re.search(source_pattern, question)
        if match:
            source = match.group(1).lower()
            return analyze_lead_source(sales_df, source)
        else:
            return analyze_all_lead_sources(sales_df)
    
    # Sales rep performance
    if "sales rep" in question or "salesperson" in question or "rep" in question:
        if "most deals" in question or "most sales" in question:
            return analyze_sales_reps_by_deals(sales_df)
        elif "most" in question or "top" in question:
            return analyze_top_sales_reps(sales_df)
        elif "worst" in question or "bottom" in question or "least" in question:
            return analyze_bottom_sales_reps(sales_df)
        else:
            return analyze_all_sales_reps(sales_df)
    
    # Deal count specific questions
    if any(term in question for term in ["most deals", "highest sales", "most sales"]):
        return analyze_sales_reps_by_deals(sales_df)
    
    # Gross profit analysis
    if "gross" in question or "profit" in question:
        if "negative" in question:
            return analyze_negative_gross(sales_df)
        elif "average" in question or "mean" in question:
            return analyze_average_gross(sales_df)
        else:
            return analyze_gross_trends(sales_df)
    
    # Time-based analysis
    if any(term in question for term in ["today", "yesterday", "this week", "last week", "month"]):
        return analyze_time_period(sales_df, question)
    
    # Fallback for unrecognized questions
    return InsightResult(
        title="Question Not Understood",
        summary="I'm not sure how to analyze that specific question.",
        metrics=[],
        recommendations=[
            "Try asking about:",
            "- Lead sources (e.g., 'How many sales from CarGurus?')",
            "- Sales rep performance (e.g., 'Who are the top sales reps?')",
            "- Gross profit (e.g., 'Show me negative gross deals')",
            "- Time periods (e.g., 'How many sales this month?')"
        ]
    )

def analyze_top_sales_reps(df: pd.DataFrame) -> InsightResult:
    """Analyze top performing sales reps."""
    # Ensure gross is numeric
    df['gross'] = pd.to_numeric(df['gross'].astype(str).str.replace(r'[\$,]', '', regex=True), errors='coerce')
    
    # Group by sales rep
    rep_stats = df.groupby('sales_rep').agg({
        'gross': ['count', 'sum', 'mean']
    }).reset_index()
    
    rep_stats.columns = ['sales_rep', 'deals', 'total_gross', 'avg_gross']
    rep_stats = rep_stats.sort_values('total_gross', ascending=False)
    
    # Get top 5 reps
    top_reps = rep_stats.head(5)
    top_rep = top_reps.iloc[0]
    
    return InsightResult(
        title="Top Sales Representatives",
        summary=f"{top_rep['sales_rep']} leads with ${top_rep['total_gross']:,.2f} from {int(top_rep['deals'])} deals.",
        metrics=[{
            "rep": row['sales_rep'],
            "deals": int(row['deals']),
            "total_gross": f"${row['total_gross']:,.2f}",
            "avg_gross": f"${row['avg_gross']:,.2f}"
        } for _, row in top_reps.iterrows()],
        chart_data=top_reps,
        recommendations=[
            f"Study {top_rep['sales_rep']}'s techniques for team training",
            "Analyze deal types and lead sources of top performers",
            "Consider implementing mentorship program"
        ]
    )

def analyze_bottom_sales_reps(df: pd.DataFrame) -> InsightResult:
    """Analyze underperforming sales reps."""
    # Ensure gross is numeric
    df['gross'] = pd.to_numeric(df['gross'].astype(str).str.replace(r'[\$,]', '', regex=True), errors='coerce')
    
    # Group by sales rep
    rep_stats = df.groupby('sales_rep').agg({
        'gross': ['count', 'sum', 'mean']
    }).reset_index()
    
    rep_stats.columns = ['sales_rep', 'deals', 'total_gross', 'avg_gross']
    rep_stats = rep_stats.sort_values('total_gross', ascending=True)
    
    # Get bottom 5 reps
    bottom_reps = rep_stats.head(5)
    bottom_rep = bottom_reps.iloc[0]
    
    return InsightResult(
        title="Sales Representatives Needing Support",
        summary=f"{bottom_rep['sales_rep']} shows opportunity for improvement with ${bottom_rep['total_gross']:,.2f} from {int(bottom_rep['deals'])} deals.",
        metrics=[{
            "rep": row['sales_rep'],
            "deals": int(row['deals']),
            "total_gross": f"${row['total_gross']:,.2f}",
            "avg_gross": f"${row['avg_gross']:,.2f}"
        } for _, row in bottom_reps.iterrows()],
        chart_data=bottom_reps,
        recommendations=[
            "Review training needs for lower performing reps",
            "Analyze deal types and conversion rates",
            "Consider additional support or mentorship"
        ]
    )

def analyze_all_sales_reps(df: pd.DataFrame) -> InsightResult:
    """Analyze performance of all sales reps."""
    # Ensure gross is numeric
    df['gross'] = pd.to_numeric(df['gross'].astype(str).str.replace(r'[\$,]', '', regex=True), errors='coerce')
    
    # Group by sales rep
    rep_stats = df.groupby('sales_rep').agg({
        'gross': ['count', 'sum', 'mean']
    }).reset_index()
    
    rep_stats.columns = ['sales_rep', 'deals', 'total_gross', 'avg_gross']
    
    # Sort by total gross for ranking
    rep_stats = rep_stats.sort_values('total_gross', ascending=True)
    
    # Get bottom performer for summary
    bottom_rep = rep_stats.iloc[0]
    
    return InsightResult(
        title="Sales Representatives Performance",
        summary=f"{bottom_rep['sales_rep']} had the lowest total gross with ${bottom_rep['total_gross']:,.2f} from {int(bottom_rep['deals'])} deals.",
        metrics=[{
            "rep": row['sales_rep'],
            "deals": int(row['deals']),
            "total_gross": f"${row['total_gross']:,.2f}",
            "avg_gross": f"${row['avg_gross']:,.2f}"
        } for _, row in rep_stats.iterrows()],
        chart_data=rep_stats,
        recommendations=[
            "Review training and support needs for lower performing reps",
            "Analyze successful techniques from top performers",
            "Consider implementing mentorship program"
        ]
    )

def analyze_all_lead_sources(df: pd.DataFrame) -> InsightResult:
    """Analyze performance across all lead sources."""
    # Ensure gross is numeric
    df['gross'] = pd.to_numeric(df['gross'].astype(str).str.replace(r'[\$,]', '', regex=True), errors='coerce')
    
    # Group by lead source
    source_stats = df.groupby('lead_source').agg({
        'gross': ['count', 'sum', 'mean']
    }).reset_index()
    
    source_stats.columns = ['lead_source', 'deals', 'total_gross', 'avg_gross']
    source_stats['share'] = (source_stats['deals'] / len(df)) * 100
    
    # Sort by total gross
    source_stats = source_stats.sort_values('total_gross', ascending=False)
    
    # Get top performer
    top_source = source_stats.iloc[0]
    
    return InsightResult(
        title="Lead Source Performance",
        summary=f"Top source {top_source['lead_source']} generated ${top_source['total_gross']:,.2f} from {int(top_source['deals'])} deals.",
        metrics=[{
            "source": row['lead_source'],
            "deals": int(row['deals']),
            "total_gross": f"${row['total_gross']:,.2f}",
            "avg_gross": f"${row['avg_gross']:,.2f}",
            "share": f"{row['share']:.1f}%"
        } for _, row in source_stats.iterrows()],
        chart_data=source_stats,
        recommendations=[
            f"Focus on {top_source['lead_source']} with highest average gross (${top_source['avg_gross']:,.2f})",
            "Review low-performing sources for optimization",
            "Consider reallocating marketing budget based on ROI"
        ]
    )

def analyze_lead_source(df: pd.DataFrame, source: str) -> InsightResult:
    """Analyze performance of a specific lead source."""
    # Ensure gross is numeric
    df['gross'] = pd.to_numeric(df['gross'].astype(str).str.replace(r'[\$,]', '', regex=True), errors='coerce')
    
    # Clean and standardize lead source values
    df['lead_source'] = df['lead_source'].str.lower()
    
    # Filter for the requested source
    source_df = df[df['lead_source'].str.contains(source)]
    
    if len(source_df) == 0:
        return InsightResult(
            title=f"No Data for {source.title()}",
            summary=f"No sales found from {source.title()}.",
            metrics=[],
            recommendations=[
                f"Verify the lead source name '{source}'",
                "Available sources: " + ", ".join(df['lead_source'].unique())
            ]
        )
    
    # Calculate metrics
    total_deals = len(source_df)
    total_gross = source_df['gross'].sum()
    avg_gross = total_gross / total_deals
    source_pct = (total_deals / len(df)) * 100
    
    return InsightResult(
        title=f"{source.title()} Performance",
        summary=f"{source.title()} generated {total_deals} deals ({source_pct:.1f}% of total) with ${total_gross:,.2f} total gross.",
        metrics=[
            {"metric": "Total Deals", "value": total_deals},
            {"metric": "Total Gross", "value": f"${total_gross:,.2f}"},
            {"metric": "Average Gross", "value": f"${avg_gross:,.2f}"},
            {"metric": "Source Share", "value": f"{source_pct:.1f}%"}
        ],
        chart_data=pd.DataFrame({
            "Metric": ["Deal Count", "Total Gross"],
            "Value": [total_deals, total_gross]
        }),
        recommendations=[
            f"Average gross per deal: ${avg_gross:,.2f}",
            f"This source represents {source_pct:.1f}% of all deals",
            "Monitor conversion rates from this source",
            "Compare ROI with other lead sources"
        ]
    )

def analyze_negative_gross(df: pd.DataFrame) -> InsightResult:
    """Analyze deals with negative gross profit."""
    negative_df = df[df['gross'] < 0]
    
    if len(negative_df) == 0:
        return InsightResult(
            title="No Negative Gross Deals",
            summary="No deals found with negative gross profit.",
            metrics=[],
            recommendations=["Continue monitoring for negative gross deals"]
        )
    
    total_deals = len(negative_df)
    total_loss = negative_df['gross'].sum()
    avg_loss = total_loss / total_deals
    pct_deals = (total_deals / len(df)) * 100
    
    return InsightResult(
        title="Negative Gross Analysis",
        summary=f"Found {total_deals} deals ({pct_deals:.1f}%) with negative gross, totaling ${abs(total_loss):,.2f} in losses.",
        metrics=[
            {"metric": "Number of Deals", "value": total_deals},
            {"metric": "Total Loss", "value": f"${abs(total_loss):,.2f}"},
            {"metric": "Average Loss", "value": f"${abs(avg_loss):,.2f}"},
            {"metric": "Percentage of Deals", "value": f"{pct_deals:.1f}%"}
        ],
        chart_data=pd.DataFrame({
            "Category": ["Negative Gross", "Other"],
            "Deals": [total_deals, len(df) - total_deals]
        }),
        recommendations=[
            "Review pricing and cost structure",
            "Analyze common factors in negative deals",
            "Consider implementing pre-deal profit checks"
        ]
    )

def analyze_average_gross(df: pd.DataFrame) -> InsightResult:
    """Analyze average gross profit across different dimensions."""
    # Ensure gross is numeric
    df['gross'] = pd.to_numeric(df['gross'].astype(str).str.replace(r'[\$,]', '', regex=True), errors='coerce')
    
    # Calculate overall metrics
    overall_avg = df['gross'].mean()
    overall_std = df['gross'].std()
    total_deals = len(df)
    
    # Calculate by sales rep
    rep_stats = df.groupby('sales_rep')['gross'].agg(['mean', 'count']).reset_index()
    rep_stats = rep_stats.sort_values('mean', ascending=False)
    top_rep = rep_stats.iloc[0]
    
    # Calculate by lead source
    source_stats = df.groupby('lead_source')['gross'].agg(['mean', 'count']).reset_index()
    source_stats = source_stats.sort_values('mean', ascending=False)
    top_source = source_stats.iloc[0]
    
    return InsightResult(
        title="Average Gross Profit Analysis",
        summary=f"Overall average gross is ${overall_avg:,.2f} across {total_deals} deals.",
        metrics=[
            {"metric": "Average Gross", "value": f"${overall_avg:,.2f}"},
            {"metric": "Standard Deviation", "value": f"${overall_std:,.2f}"},
            {"metric": "Total Deals", "value": total_deals},
            {"metric": "Top Rep Average", "value": f"${top_rep['mean']:,.2f} ({top_rep['sales_rep']})"},
            {"metric": "Top Source Average", "value": f"${top_source['mean']:,.2f} ({top_source['lead_source']})"}
        ],
        chart_data=pd.DataFrame({
            "Category": ["Overall", f"Top Rep ({top_rep['sales_rep']})", f"Top Source ({top_source['lead_source']})"],
            "Average Gross": [overall_avg, top_rep['mean'], top_source['mean']]
        }),
        recommendations=[
            f"Study {top_rep['sales_rep']}'s techniques for maintaining high averages",
            f"Focus on {top_source['lead_source']} leads which yield higher averages",
            "Consider setting minimum gross targets based on these benchmarks"
        ]
    )

def analyze_gross_trends(df: pd.DataFrame) -> InsightResult:
    """Analyze trends in gross profit over time."""
    # Ensure gross is numeric and date is datetime
    df['gross'] = pd.to_numeric(df['gross'].astype(str).str.replace(r'[\$,]', '', regex=True), errors='coerce')
    df['date'] = pd.to_datetime(df['date'])
    
    # Group by date and calculate daily metrics
    daily_stats = df.groupby(df['date'].dt.date).agg({
        'gross': ['count', 'sum', 'mean']
    }).reset_index()
    
    daily_stats.columns = ['date', 'deals', 'total_gross', 'avg_gross']
    
    # Calculate trend metrics
    first_day = daily_stats.iloc[0]
    last_day = daily_stats.iloc[-1]
    total_change = last_day['avg_gross'] - first_day['avg_gross']
    pct_change = (total_change / first_day['avg_gross']) * 100 if first_day['avg_gross'] != 0 else 0
    
    # Calculate moving averages
    daily_stats['7_day_avg'] = daily_stats['avg_gross'].rolling(7).mean()
    
    trend_direction = "increased" if total_change > 0 else "decreased"
    
    return InsightResult(
        title="Gross Profit Trends",
        summary=f"Average gross has {trend_direction} by ${abs(total_change):,.2f} ({abs(pct_change):.1f}%) over the period.",
        metrics=[
            {"metric": "Starting Average", "value": f"${first_day['avg_gross']:,.2f}"},
            {"metric": "Current Average", "value": f"${last_day['avg_gross']:,.2f}"},
            {"metric": "Total Change", "value": f"${total_change:,.2f}"},
            {"metric": "Percent Change", "value": f"{pct_change:.1f}%"}
        ],
        chart_data=daily_stats,
        recommendations=[
            "Monitor 7-day moving average for short-term trends",
            "Investigate factors behind significant daily variations",
            "Set gross targets based on historical performance"
        ]
    )

def analyze_time_period(df: pd.DataFrame, question: str) -> InsightResult:
    """Analyze sales performance for a specific time period."""
    # Convert date column to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Determine time period from question
    today = pd.Timestamp.now().normalize()
    
    if "today" in question:
        period_df = df[df['date'].dt.date == today.date()]
        period_name = "Today"
    elif "yesterday" in question:
        period_df = df[df['date'].dt.date == (today - timedelta(days=1)).date()]
        period_name = "Yesterday"
    elif "this week" in question:
        start_of_week = today - timedelta(days=today.weekday())
        period_df = df[df['date'] >= start_of_week]
        period_name = "This Week"
    elif "last week" in question:
        end_of_last_week = today - timedelta(days=today.weekday())
        start_of_last_week = end_of_last_week - timedelta(days=7)
        period_df = df[(df['date'] >= start_of_last_week) & (df['date'] < end_of_last_week)]
        period_name = "Last Week"
    elif "this month" in question:
        start_of_month = today.replace(day=1)
        period_df = df[df['date'] >= start_of_month]
        period_name = "This Month"
    elif "last month" in question:
        start_of_this_month = today.replace(day=1)
        end_of_last_month = start_of_this_month - timedelta(days=1)
        start_of_last_month = end_of_last_month.replace(day=1)
        period_df = df[(df['date'] >= start_of_last_month) & (df['date'] < start_of_this_month)]
        period_name = "Last Month"
    else:
        return InsightResult(
            title="Time Period Not Understood",
            summary="Could not determine which time period to analyze.",
            metrics=[],
            recommendations=["Try specifying 'today', 'this week', 'last month', etc."]
        )
    
    if len(period_df) == 0:
        return InsightResult(
            title=f"No Sales for {period_name}",
            summary=f"No deals found for {period_name.lower()}.",
            metrics=[],
            recommendations=["Check if data is up to date"]
        )
    
    total_deals = len(period_df)
    total_gross = period_df['gross'].sum()
    avg_gross = total_gross / total_deals
    
    return InsightResult(
        title=f"{period_name} Performance",
        summary=f"{period_name} had {total_deals} deals totaling ${total_gross:,.2f} in gross profit.",
        metrics=[
            {"metric": "Total Deals", "value": total_deals},
            {"metric": "Total Gross", "value": f"${total_gross:,.2f}"},
            {"metric": "Average Gross", "value": f"${avg_gross:,.2f}"}
        ],
        chart_data=period_df.groupby(period_df['date'].dt.date).agg({
            'gross': ['count', 'sum']
        }).reset_index(),
        recommendations=[
            f"Average gross per deal: ${avg_gross:,.2f}",
            "Monitor daily trends",
            "Compare to previous periods"
        ]
    )