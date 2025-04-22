import streamlit as st
import pandas as pd
import datetime
import json
import os
import random
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import sys
import base64
from watchdog_ai.ui.utils.status_formatter import StatusType, format_status_text
from io import BytesIO
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from src.insight_tagger import InsightStore

# --- Data Loading ---
def load_insights_data() -> List[Dict[str, Any]]:
    """Load insights from InsightStore and parse timestamps."""
    try:
        store = InsightStore()  # Uses default storage path
        insights = store.insights
        
        # Parse ISO timestamps into datetime objects
        for insight in insights:
            if isinstance(insight.get('timestamp'), str):
                try:
                    insight['timestamp'] = datetime.fromisoformat(insight['timestamp'].replace('Z', '+00:00'))
                except ValueError:
                    # If timestamp parsing fails, use current time
                    insight['timestamp'] = datetime.now()
        
        # Sort by timestamp (newest first)
        insights.sort(key=lambda x: x['timestamp'], reverse=True)
        return insights
    except Exception as e:
        st.error(f"Error loading insights: {str(e)}")
        return []

def find_similar_insights(current_insight: Dict[str, Any], all_insights: List[Dict[str, Any]], max_similar: int = 3) -> List[Dict[str, Any]]:
    """Find insights similar to the current one based on tags and content."""
    if not current_insight or not all_insights:
        return []
    
    # Extract keywords from current insight
    keywords = set()
    if 'tags' in current_insight:
        keywords.update(current_insight['tags'])
    
    # Add words from summary
    if 'summary' in current_insight:
        keywords.update(word.lower() for word in current_insight['summary'].split())
    
    # Add KPI names
    if 'kpi_metrics' in current_insight:
        keywords.update(current_insight['kpi_metrics'].keys())
    
    similar_insights = []
    for insight in all_insights:
        if insight['id'] == current_insight['id']:
            continue
            
        # Calculate similarity score
        score = 0
        if 'tags' in insight:
            score += len(set(insight['tags']) & keywords)
        if 'summary' in insight:
            score += len(set(word.lower() for word in insight['summary'].split()) & keywords)
        if 'kpi_metrics' in insight:
            score += len(set(insight['kpi_metrics'].keys()) & keywords)
            
        if score > 0:
            similar_insights.append((score, insight))
    
    # Sort by similarity score and return top matches
    similar_insights.sort(reverse=True)
    return [insight for _, insight in similar_insights[:max_similar]]

def export_insight_as_json(insight: Dict[str, Any]) -> str:
    """Convert insight to JSON string, handling datetime serialization."""
    def datetime_handler(obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
    
    return json.dumps(insight, default=datetime_handler, indent=2)

def export_insights_as_csv(insights: List[Dict[str, Any]]) -> str:
    """Convert insights to CSV format."""
    if not insights:
        return ""
    
    # Flatten the insight structure
    flattened_insights = []
    for insight in insights:
        flat_insight = {
            'id': insight.get('id', ''),
            'timestamp': insight.get('timestamp', '').isoformat() if isinstance(insight.get('timestamp'), datetime) else '',
            'priority': insight.get('priority', ''),
            'dealership_name': insight.get('dealership_name', ''),
            'summary': insight.get('summary', ''),
            'confidence_score': insight.get('raw_data', {}).get('confidence_score', ''),
            'potential_impact': insight.get('raw_data', {}).get('potential_impact', '')
        }
        
        # Add KPI metrics
        for kpi, values in insight.get('kpi_metrics', {}).items():
            flat_insight[f'{kpi}_value'] = values.get('value', '')
            flat_insight[f'{kpi}_delta'] = values.get('delta', '')
        
        flattened_insights.append(flat_insight)
    
    # Convert to DataFrame and then to CSV
    df = pd.DataFrame(flattened_insights)
    return df.to_csv(index=False)

def generate_insight_pdf(insight: Dict[str, Any]) -> bytes:
    """Generate a PDF report for a single insight."""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    # Title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=16,
        spaceAfter=30
    )
    story.append(Paragraph(f"Insight Report - {insight['dealership_name']}", title_style))
    story.append(Spacer(1, 12))

    # Metadata
    meta_data = [
        ['ID:', insight['id']],
        ['Priority:', insight['priority']],
        ['Generated:', insight['timestamp'].strftime('%Y-%m-%d %H:%M')],
    ]
    meta_table = Table(meta_data, colWidths=[100, 300])
    meta_table.setStyle(TableStyle([
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
        ('PADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 12))

    # Summary
    story.append(Paragraph("Summary", styles['Heading2']))
    story.append(Paragraph(insight['summary'], styles['Normal']))
    story.append(Spacer(1, 12))

    # KPI Metrics
    if insight.get('kpi_metrics'):
        story.append(Paragraph("Key Metrics", styles['Heading2']))
        metrics_data = [['Metric', 'Value', 'Change']]
        for kpi, values in insight['kpi_metrics'].items():
            metrics_data.append([
                kpi,
                str(values.get('value', 'N/A')),
                str(values.get('delta', 'N/A'))
            ])
        metrics_table = Table(metrics_data, colWidths=[150, 100, 100])
        metrics_table.setStyle(TableStyle([
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
            ('PADDING', (0, 0), (-1, -1), 6),
        ]))
        story.append(metrics_table)
        story.append(Spacer(1, 12))

    # Recommendations
    if insight.get('recommendations'):
        story.append(Paragraph("Recommendations", styles['Heading2']))
        for rec in insight['recommendations']:
            story.append(Paragraph(f"• {rec}", styles['Normal']))
        story.append(Spacer(1, 12))

    # Generate PDF
    doc.build(story)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes

def display_insight(insight: Dict[str, Any], all_insights: List[Dict[str, Any]]):
    """Renders a single insight using Streamlit elements."""
    
    priority_color_map = {
        "Critical": "error",
        "Recommended": "warning",
        "Optional": "info"
    }
    badge_color = priority_color_map.get(insight['priority'], "secondary")
    
    # Determine status type based on insight content
    insight_type = "INSIGHT"  # Default label
    status_type = StatusType.INFO  # Default status type
    summary_lower = insight['summary'].lower()
    if any(kw in summary_lower for kw in ['sales', 'revenue', 'profit']):
        insight_type = "SALES"
        status_type = StatusType.SUCCESS
    elif any(kw in summary_lower for kw in ['inventory', 'stock']):
        insight_type = "INVENTORY"
        status_type = StatusType.INFO
    elif any(kw in summary_lower for kw in ['service', 'repair']):
        insight_type = "SERVICE"
        status_type = StatusType.INFO
    elif any(kw in summary_lower for kw in ['lead', 'customer']):
        insight_type = "CUSTOMER"
        status_type = StatusType.INFO
        
    # Format status text
    status_label = format_status_text(status_type, custom_text=insight_type)

    with st.container(border=True):
        # --- Header ---
        col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
        with col1:
            st.markdown(f"{status_label} {insight['dealership_name']}", unsafe_allow_html=True)
            st.caption(f"Insight ID: {insight['id']}")
        with col2:
            st.markdown(f"**Priority:** <span class='badge-{badge_color}'>{insight['priority']}</span>", unsafe_allow_html=True)
        with col3:
            st.caption(f"Generated: {insight['timestamp'].strftime('%Y-%m-%d %H:%M')}")
        with col4:
            # Export button for this insight
            # Export button for this insight
            st.download_button(
                "Export JSON",
                export_insight_as_json(insight),
                file_name=f"insight_{insight['id']}.json",
                mime="application/json",
                help="Download this insight as a JSON file"
            )
        st.divider()
        
        # --- Body ---
        st.markdown(f"**Summary:** {insight['summary']}")
        
        # KPI Metrics with tooltips
        if insight.get("kpi_metrics"):
            st.markdown("**Key Metrics:**")
            kpi_cols = st.columns(len(insight["kpi_metrics"]))
            for i, (kpi, values) in enumerate(insight["kpi_metrics"].items()):
                with kpi_cols[i]:
                    st.metric(
                        label=kpi,
                        value=values.get("value", "N/A"),
                        delta=values.get("delta"),
                        help=f"Click to see historical trend for {kpi}"  # Tooltip
                    )

        # Breakdown Chart/Table in expander
        # Breakdown Chart/Table in expander
        if insight.get("breakdown"):
            breakdown_label = format_status_text(StatusType.INFO, custom_text="View Breakdown")
            with st.expander(f"{breakdown_label}", unsafe_allow_html=True, expanded=False):
                st.markdown(f"**{breakdown.get('title', 'Breakdown')}:**")
                if breakdown.get("type") == "bar_chart" and isinstance(breakdown.get("data"), dict):
                    df_breakdown = pd.DataFrame.from_dict(breakdown["data"], orient='index', columns=['Value'])
                    st.bar_chart(df_breakdown)
                elif breakdown.get("type") == "table" and isinstance(breakdown.get("data"), dict):
                    df_breakdown = pd.DataFrame.from_dict(breakdown["data"], orient='index', columns=['Value'])
                    st.table(df_breakdown)
                else:
                    st.write("Breakdown data not available or in unsupported format.")
                
        # Recommendations in expander
        # Recommendations in expander
        if insight.get("recommendations"):
            recommendations_label = format_status_text(StatusType.SUCCESS, custom_text="View Recommendations")
            with st.expander(f"{recommendations_label}", unsafe_allow_html=True, expanded=False):
                    st.markdown(f"- {rec}")

        # Drill-Down Section
        # Drill-Down Section
        details_label = format_status_text(StatusType.INFO, custom_text="Explore Details")
        with st.expander(f"{details_label}", unsafe_allow_html=True, expanded=False):
            
            with tab1:
                if insight.get("raw_data"):
                    st.json(insight["raw_data"])
                else:
                    st.info("No raw metrics available for this insight.")
            
            with tab2:
                if insight.get("data_rows"):
                    st.markdown("**Sample Data Points:**")
                    df = pd.DataFrame(insight["data_rows"])
                    st.dataframe(
                        df.head(),
                        use_container_width=True,
                        hide_index=True,
                        help="First 5 rows of relevant data"
                    )
                    
                    if len(df) > 5:
                        st.caption(f"Showing 5 of {len(df)} total rows")
                else:
                    st.info("No sample data rows available for this insight.")
                
            with tab3:
                similar = find_similar_insights(insight, all_insights)
                if similar:
                    for sim in similar:
                        with st.container(border=True):
                            st.markdown(f"**{sim['dealership_name']}** - {sim['summary']}")
                            st.caption(f"Priority: {sim['priority']} | Generated: {sim['timestamp'].strftime('%Y-%m-%d %H:%M')}")
                            if st.button("View Details", key=f"sim_{sim['id']}"):
                                st.session_state.selected_insight = sim['id']
                else:
                    st.info("No similar insights found.")

        # Export Options
        col1, col2 = st.columns([1, 1])
        with col1:
            # JSON export
            # JSON export
            st.download_button(
                "Export as JSON",
                export_insight_as_json(insight),
                file_name=f"insight_{insight['id']}.json",
                mime="application/json",
                help="Download this insight as a JSON file"
            )
        with col2:
            # PDF export
            pdf_bytes = generate_insight_pdf(insight)
            st.download_button(
                "Export as PDF",
                pdf_bytes,
                file_name=f"insight_{insight['id']}.pdf",
                mime="application/pdf",
                help="Download this insight as a PDF report"
            )
def render_page():
    """Renders the full Insight Feed page."""
    # Apply caching to load_insights_data here
    @st.cache_data(ttl=300)  # Cache for 5 minutes
    def cached_load_insights():
        return load_insights_data()
    
    st.title("Insight Feed")
    
    # Load insights using cached function
    insights = cached_load_insights()
    
    # Initialize session state for pagination
    if 'page_number' not in st.session_state:
        st.session_state.page_number = 1
    
    # --- Filtering ---
    st.sidebar.header("Filters")
    
    # Priority Filter
    priorities = ["All"] + sorted(list(set(i['priority'] for i in insights)))
    selected_priority = st.sidebar.selectbox("Filter by Priority", options=priorities)

    # Date Range Filter
    min_date = min(i['timestamp'].date() for i in insights) if insights else datetime.now().date()
    max_date = max(i['timestamp'].date() for i in insights) if insights else datetime.now().date()
    
    selected_date_range = st.sidebar.date_input(
        "Filter by Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )

    # Text Search
    search_query = st.sidebar.text_input("Search Insights", help="Search in summaries and recommendations")
    
    # --- What-If Simulation ---
    st.sidebar.header("What-If Simulation")
    simulation_label = format_status_text(StatusType.INFO, custom_text="Simulate Scenarios")
    with st.sidebar.expander(f"{simulation_label}", unsafe_allow_html=True):
        st.caption("Adjust variables to simulate potential outcomes")
        inventory_change = st.slider("Inventory Level Change", -50, 50, 0, 5, format="%d%%")
        price_change = st.slider("Price Adjustment", -30, 30, 0, 5, format="%d%%")
        marketing_change = st.slider("Marketing Spend Change", -50, 50, 0, 5, format="%d%%")
        
        if st.button("Simulate Impact"):
            st.info(f"""
                Simulated Impact (Demo):
                - Projected Revenue Change: {random.uniform(-5, 15):.1f}%
                - Expected Lead Volume: {random.uniform(-10, 20):.1f}%
                - Estimated Profit Margin: {random.uniform(-3, 8):.1f}%
                """)

    # --- Apply Filters ---
    filtered_insights = insights

    # Priority
    if selected_priority != "All":
        filtered_insights = [i for i in filtered_insights if i['priority'] == selected_priority]

    # Date Range
    if isinstance(selected_date_range, tuple) and len(selected_date_range) == 2:
        start_date, end_date = selected_date_range
        end_date = datetime.combine(end_date, datetime.max.time())  # Include full end date
        filtered_insights = [
            i for i in filtered_insights 
            if start_date <= i['timestamp'].date() <= end_date.date()
        ]
    
    # Text Search
    if search_query:
        query_lower = search_query.lower()
        filtered_insights = [
            i for i in filtered_insights 
            if query_lower in i['summary'].lower() or 
               any(query_lower in rec.lower() for rec in i.get('recommendations', []))
        ]

    # --- Pagination ---
    items_per_page = 10
    total_pages = max(1, (len(filtered_insights) + items_per_page - 1) // items_per_page)
    
    # Ensure current page is valid
    st.session_state.page_number = min(st.session_state.page_number, total_pages)
    
    # Calculate slice indices
    start_idx = (st.session_state.page_number - 1) * items_per_page
    end_idx = start_idx + items_per_page
    
    # Display header with export button
    col1, col2 = st.columns([3, 1])
    with col1:
        st.header(f"Displaying {len(filtered_insights)} Insights (Page {st.session_state.page_number} of {total_pages})")
    with col2:
        if filtered_insights:
            st.download_button(
                "Export All as CSV",
                export_insights_as_csv(filtered_insights),
                file_name="insights_export.csv",
                mime="text/csv",
                help="Download all filtered insights as a CSV file"
            )

    # Display warning if no insights match filters
    if not filtered_insights:
        st.warning("No insights match the current filters.")
    else:
        # Display paginated insights
        page_insights = filtered_insights[start_idx:end_idx]
        for insight in page_insights:
            display_insight(insight, filtered_insights)
            st.markdown("---")
        
        # Pagination controls
        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            if st.session_state.page_number > 1:
                if st.button("← Previous Page"):
                    st.session_state.page_number -= 1
                    st.rerun()
        with col3:
            if st.session_state.page_number < total_pages:
                if st.button("Next Page →"):
                    st.session_state.page_number += 1
                    st.rerun()
        
        # Scroll to top button (only show if more than one page)
        if total_pages > 1:
            st.markdown("""
                <a href="#top" style="position: fixed; bottom: 20px; right: 20px; 
                background-color: #0E1117; padding: 10px; border-radius: 5px; 
                text-decoration: none; color: white;">↑ Top</a>
                """, unsafe_allow_html=True)

    # Add custom CSS for badges and layout
    st.markdown("""
        <style>
        .badge-error { background-color: #ff4b4b; color: white; padding: 4px 8px; border-radius: 4px; }
        .badge-warning { background-color: #ffa726; color: white; padding: 4px 8px; border-radius: 4px; }
        .badge-info { background-color: #42a5f5; color: white; padding: 4px 8px; border-radius: 4px; }
        .badge-secondary { background-color: #9e9e9e; color: white; padding: 4px 8px; border-radius: 4px; }
        </style>
    """, unsafe_allow_html=True)

# --- Entry Point ---
if __name__ == "__main__":
    render_page()
