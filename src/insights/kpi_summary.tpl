You are analyzing KPI data for a dealership. Focus on the most significant changes and trends.

Context:
- Time Period: {{ date_range }}
- Metrics Table:
{{ metrics_table }}

{% if threshold_context %}
Threshold Context:
{{ threshold_context }}
{% endif %}

{% if feedback_context %}
Feedback Context:
{{ feedback_context }}
{% endif %}

Generate a concise 1-2 sentence executive summary highlighting:
1. Most significant performance changes
2. Key contributors to those changes
3. Notable trends or anomalies

The summary should be data-driven and actionable. Focus on metrics that deviate significantly from historical averages.

Example: "Sales rep Alice's gross margin increased by 15% month-over-month, outperforming the group average of 8%. Inventory aging improved with a 20% reduction in vehicles over 90 days."