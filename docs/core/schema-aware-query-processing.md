# Schema-Aware Query Processing Implementation

## Overview

The Schema-Aware Query Processing system ensures all queries are semantically correct, schema-safe, and business-rule compliant. It consists of four main components:

1. **Dynamic Executive Schema Profile Tuning**
2. **Pluggable Rule Engine**
3. **Query Rewriting + Schema Compatibility**
4. **Precision Scoring Engine**

## Components Implemented

### 1. SchemaProfile Manager

- `SchemaProfileManager` class with methods:
  - `load_profiles()`: Loads schema profiles from disk
  - `get_profile(profile_id)`: Retrieves a specific profile
  - `get_profile_for_role(role)`: Gets profile for specific role
  - `get_adjusted_profile(profile_id, user_id)`: Gets personalized profile with adjustments

- `SchemaAdjustment` class for tracking user-specific schema adjustments
  - Persists per-user schema adjustments and applies them at runtime
  - Supports different adjustment types: aliases, visibility, relevance, etc.

### 2. Business Rule Engine

- `BusinessRuleEngine` with methods:
  - `load_rules(file_path)`: Loads rules from YAML file
  - `save_rules(file_path)`: Saves rules to YAML file
  - `get_rules_for_column(column_name)`: Gets rules for a specific column
  - `get_rules_for_role(role)`: Gets rules for a specific executive role
  - `evaluate_rule(rule_id, data)`: Evaluates a specific rule against data
  - `evaluate_all_rules(data, role)`: Evaluates all applicable rules

- Support for multiple rule types:
  - Comparison rules (==, !=, >, >=, <, <=, in, not in)
  - Range rules (min_value to max_value)
  - Regular expression rules
  - Custom function rules

### 3. Query Rewriter

- `QueryRewriter` class with methods:
  - `load_model(model_path)`: Loads NLP model for query rewriting
  - `rewrite_query(query)`: Rewrites ambiguous queries, substitutes tokens
  - `_apply_nlp_rewrites(query, metadata)`: Applies NLP-based rewrites

- `QueryRewriteStats` class for tracking rewrite statistics
  - Records term mappings, ambiguity resolutions, success rate
  - Provides historical data to improve future rewrites

### 4. Precision Scoring Engine

- `PrecisionScoringEngine` class with methods:
  - `load_model(model_path)`: Loads trained model for scoring
  - `predict_precision(query, metadata)`: Predicts query precision
  - `train_on_feedback(feedback_data)`: Trains model from feedback

- Multiple model types:
  - `HeuristicPrecisionModel`: Rule-based approach using thresholds
  - `MLPrecisionModel`: Machine learning approach for complex patterns

## Integration Demo

The complete system is demonstrated in `demo_schema_processor.py`, which:

1. Loads schema profiles from disk
2. Initializes business rules from YAML
3. Creates query rewriters and precision scorers
4. Processes sample queries and displays results
5. Provides an interactive demo mode

## Example Usage

```python
# Load components
profile_manager = SchemaProfileManager("profiles")
profiles = profile_manager.load_profiles()
rule_engine = BusinessRuleEngine("BusinessRuleRegistry.yaml")

# Get a profile and personalize it for the user
base_profile = profile_manager.get_profile("general_manager")
user_profile = profile_manager.get_adjusted_profile("general_manager", "user_123")

# Create query processors
rewriter = QueryRewriter(user_profile, rule_engine)
scorer = PrecisionScoringEngine(user_profile)

# Process a query
rewritten_query, metadata = rewriter.rewrite_query("What's our total gross profit?")
precision = scorer.predict_precision(rewritten_query, metadata)

# Check if query is valid based on precision
is_valid = precision["score"] >= 0.4  # Medium or higher confidence
```

## Key Features

- **Role-based schema profiles**: Different views for different executive roles
- **Personalized schema adjustments**: Learn from user feedback
- **Business rule enforcement**: Ensure data integrity and semantic correctness
- **Ambiguity resolution**: Rewrite queries to use canonical column names
- **Precision prediction**: Score queries for reliability and prevent low-quality results
- **Continuous learning**: Systems improve over time from user interactions

## Testing and Usage

The system can be tested by running the demo script:

```bash
./demo_schema_processor.py
```

This will demonstrate:
1. Sample query processing
2. Role-specific schema enforcement
3. Query rewriting
4. Precision scoring
5. Interactive query processing

## Future Enhancements

1. **Advanced NLP Models**: Integrate with more sophisticated language models
2. **Automated Schema Learning**: Discover schema relationships from data
3. **Cross-Profile Reasoning**: Handle queries spanning multiple executive roles
4. **Expanded Business Rules**: Support more complex rule types and relationships