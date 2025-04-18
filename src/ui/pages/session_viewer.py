import streamlit as st
import json
import os
from datetime import datetime, date, time as dtime
import pandas as pd
import altair as alt

from src.utils.session import get_all_session_ids


# Function to load audit logs from Redis with fallback to file
def load_audit_logs():
    logs = []
    try:
        import redis
        r = redis.Redis(host='localhost', port=6379, db=0)
        raw_logs = r.lrange("watchdog:audit_logs", 0, -1)
        if raw_logs:
            logs = [json.loads(item.decode('utf-8')) for item in raw_logs]
        else:
            raise Exception("No logs in Redis")
    except Exception as e:
        if os.path.exists("audit.log"):
            with open("audit.log", 'r') as f:
                lines = f.readlines()
            logs = [json.loads(line.strip()) for line in lines if line.strip()]
    return logs


# Sidebar filters setup
st.sidebar.header("Session History Filters")

# Session ID dropdown
session_ids = get_all_session_ids()
if not session_ids:
    session_ids = ["None"]
selected_session = st.sidebar.selectbox("Select Session ID", options=session_ids)

# Event type multi-select
event_options = ["upload_complete", "normalize_complete", "insight_start", "insight_end", "insight_error"]
selected_events = st.sidebar.multiselect("Select Event Types", options=event_options, default=event_options)

# Date range picker
# Default to today
default_start = date.today()
default_end = date.today()
selected_dates = st.sidebar.date_input("Select Date Range", [default_start, default_end])
if isinstance(selected_dates, list) and len(selected_dates) == 2:
    start_date = datetime.combine(selected_dates[0], dtime.min)
    end_date = datetime.combine(selected_dates[1], dtime.max)
else:
    start_date = datetime.combine(default_start, dtime.min)
    end_date = datetime.combine(default_end, dtime.max)

# Load audit logs
logs = load_audit_logs()

# Filter logs
filtered_logs = []
for log in logs:
    try:
        log_time = datetime.fromisoformat(log.get('timestamp'))
    except Exception:
        continue
    if selected_session != "None" and log.get('session_id') != selected_session:
        continue
    if log.get('event') not in selected_events:
        continue
    if log_time < start_date or log_time > end_date:
        continue
    filtered_logs.append(log)

# Main area display
st.header("Session History Viewer")
st.write(f"Displaying {len(filtered_logs)} audit events for session: {selected_session}")

if filtered_logs:
    df = pd.DataFrame(filtered_logs)
    st.dataframe(df)
    
    # Optional: Timeline chart using Altair
    try:
        chart = alt.Chart(df).mark_circle(size=60).encode(
            x=alt.X('timestamp:T', title='Time'),
            y=alt.Y('event:N', title='Event'),
            tooltip=list(df.columns)
        ).interactive()
        st.altair_chart(chart, use_container_width=True)
    except Exception as e:
        st.write("Timeline chart unavailable.")
else:
    st.write("No audit events found for the selected filters.") 