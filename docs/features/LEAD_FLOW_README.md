# Lead Flow Optimization & Insight Tagging Engine

This module provides lead flow analysis capabilities and enhanced insight tagging features for Watchdog AI.

## Lead Flow Optimization

The Lead Flow Optimizer (`lead_flow_optimizer.py`) tracks lead progression through the sales pipeline, identifying bottlenecks, analyzing rep and source performance, and generating recommendations for improvement.

### Key Features

- **Lead Pipeline Analytics**: Track time from lead creation → contact → appointment → test drive → offer → sold → delivery → closed
- **Bottleneck Detection**: Identify where leads are getting stuck in the sales process
- **Aged Lead Detection**: Flag leads that have been in the system too long (> 30 days by default)
- **Performance Analysis**: Compare performance by:
  - Sales rep
  - Lead source
  - Vehicle model
- **Recommendation Engine**: Generate actionable recommendations based on analysis
- **Visualization**: Generate charts showing bottlenecks and performance metrics

### Usage

```python
from src.validators.lead_flow_optimizer import LeadFlowOptimizer, prepare_lead_data

# Load your lead data (as a pandas DataFrame)
df = load_your_data()  # Replace with your data loading logic

# Prepare the data (if needed)
prepared_df = prepare_lead_data(df, {
    "CreatedDate": "created_date",
    "FirstContactDate": "contacted_date",
    # Map your date columns to the expected format
})

# Initialize the optimizer
optimizer = LeadFlowOptimizer()

# Process the data
results = optimizer.process_lead_data(prepared_df)

# Get insights
bottlenecks = optimizer.identify_bottlenecks()
aged_leads = optimizer.flag_aged_leads()
rep_performance = optimizer.get_rep_performance()
source_performance = optimizer.get_source_performance()

# Get summary and recommendations
summary = optimizer.get_summary()
recommendations = optimizer.generate_recommendations()
```

See `examples/lead_flow_example.py` for a complete working example.

## Enhanced Insight Tagging

The enhanced Insight Tagger (`insight_tagger.py`) adds multi-tag support, audit metadata, and embedding-based similarity grouping for insights.

### Key Features

- **Multi-Tag Support**: Each insight can have multiple tags (e.g., ["profit", "lead", "alert"])
- **Tag Suggestions**: Tags are suggested based on content analysis and embedding similarity
- **Audit Metadata**: Each insight includes creation time, origin dataset, and tag history
- **Embedding-Based Similarity**: Group similar insights using SentenceTransformers embeddings
- **Audit Logging**: Track all changes to insights in a separate audit log file

### Usage

```python
from src.insight_tagger import InsightTagger, InsightStore, tag_insight

# Create a sample insight
insight = {
    "title": "Sales Performance Alert",
    "summary": "Sales have decreased by 15% over the last month.",
    "metrics": {
        "sales_change_pct": -15,
        "total_sales": 850000,
        "target_sales": 1000000
    },
    "recommendations": [
        "Review sales process",
        "Implement targeted promotions"
    ]
}

# Tag a single insight
tagger = InsightTagger()
tagged = tagger.tag_insight(insight)

# Store tagged insights
store = InsightStore("insights.json")
insight_id = store.add_insight(tagged)

# Find insights by tags
sales_insights = store.get_insights_by_tag("sales")
critical_alerts = store.get_insights_by_tags(["critical", "alert"], match_all=True)

# Find similar insights
similar = tagger.find_similar_insights(tagged, store.insights)

# Group insights by similarity
groups = tagger.group_insights_by_similarity(store.insights)
```

## Connecting Lead Flow Analysis with Insight Tagging

Insights generated from lead flow analysis can be automatically tagged and stored:

```python
from src.validators.lead_flow_optimizer import LeadFlowOptimizer
from src.insight_tagger import InsightTagger, InsightStore, tag_insight

# Initialize components
optimizer = LeadFlowOptimizer()
tagger = InsightTagger()
store = InsightStore()

# Process lead data
optimizer.process_lead_data(your_lead_data)

# Create insights from bottlenecks
bottlenecks = optimizer.identify_bottlenecks()
for stage, data in bottlenecks.items():
    if data.get('is_bottleneck', False):
        insight = {
            "title": f"Bottleneck in {stage}",
            "summary": f"Process bottleneck detected in {stage}...",
            "metrics": {
                "average_days": data.get('average_days', 0),
                "threshold": data.get('threshold', 0),
            },
            "recommendations": [
                f"Review the {stage} process"
            ]
        }
        
        # Tag and store the insight
        tagged_insight = tag_insight(insight)
        store.add_insight(tagged_insight)
```

## Testing

Unit tests are provided for both the Lead Flow Optimizer and Insight Tagger:

- `tests/unit/test_lead_flow_optimizer.py`
- `tests/unit/test_insight_tagger.py`

Integration tests demonstrate end-to-end functionality:

- `tests/integration/test_lead_flow_integration.py`

## Dependencies

- pandas & numpy for data manipulation
- sentence-transformers for embedding-based similarity (optional)
- matplotlib for visualization (in the example)

These dependencies are included in the updated requirements.txt file.