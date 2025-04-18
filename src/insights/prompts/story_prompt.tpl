You are creating an executive story from multiple dealership insights. Weave these insights into a coherent narrative that highlights key trends and opportunities.

Context:
- Time Period: {{ date_range }}
- Selected Insights: {{ selected_insights | join(", ") }}

Insights Data:
{% for insight in insights %}
### {{ insight.title }}
{{ insight.summary }}
{{ insight.metrics_table }}
{% endfor %}

{% if feedback_stats %}
Feedback Context:
{{ feedback_stats }}
{% endif %}

Generate a cohesive narrative that:
1. Summarizes the overall business performance
2. Highlights key trends and relationships between different metrics
3. Identifies potential opportunities or risks
4. Provides clear, actionable recommendations

The narrative should flow naturally between topics and maintain a clear thread connecting the insights. Focus on insights that show significant changes or patterns.

End with a "Next Steps" section listing 2-3 specific, actionable recommendations based on the data.

Example:
"Q3 2023 showed strong performance in gross margins, driven by Alice's exceptional sales team performance (+15% vs. target). However, inventory aging metrics reveal an opportunity: 35% of inventory is over 90 days old, concentrated in luxury vehicles. This aging inventory correlates with lower lead conversion rates from CarGurus (down 10% MoM).

Next Steps:
1. Consider targeted promotions for aged luxury inventory through top-performing channels
2. Evaluate pricing strategy for vehicles approaching 60-day threshold
3. Strengthen CarGurus listing quality based on successful AutoTrader patterns"