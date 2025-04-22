# Schema Profile Editor & LLM Mapping Integration Plan

## 1. Schema Profile Editor Enhancements

### SchemaProfileEditor Class
```python
class SchemaProfileEditor:
    def __init__(self):
        self.current_profile = None
        self.sample_data = None
        self.preview_cache = {}
    
    def load_profile(self, profile_id: str) -> SchemaProfile:
        # Load profile from disk
        pass
    
    def save_profile(self, profile: SchemaProfile) -> bool:
        # Validate and save profile
        pass
        
    def preview_validation(self, profile: SchemaProfile, sample_data: pd.DataFrame) -> Dict[str, Any]:
        # Run validation on sample data
        pass
        
    def export_profile(self, profile: SchemaProfile, format: str = 'json') -> str:
        # Export to JSON/YAML
        pass
        
    def import_profile(self, data: str, format: str = 'json') -> SchemaProfile:
        # Import and validate profile
        pass
```

### Live Preview Component
```python
def render_preview_panel(profile: SchemaProfile, sample_data: pd.DataFrame):
    # Show first 5 rows
    st.dataframe(sample_data.head())
    
    # Run validation
    results = validate_with_profile(sample_data, profile)
    
    # Show validation results
    st.write("Validation Results:")
    for issue in results.issues:
        st.warning(issue)
```

## 2. LLM-Powered Column Mapper

### LLMColumnMapper Class
```python
class LLMColumnMapper:
    def __init__(self, cache_dir: str = ".cache/column_mapping"):
        self.llm = LLMEngine()
        self.cache = {}
        self.cache_dir = cache_dir
        
    def get_mapping_confidence(self, source_col: str, target_col: str) -> float:
        # Calculate base confidence
        pass
        
    async def get_llm_suggestions(self, source_cols: List[str], 
                                target_schema: SchemaProfile) -> Dict[str, List[str]]:
        # Get suggestions from LLM
        pass
        
    def persist_learned_mapping(self, source: str, target: str, 
                              confidence: float) -> None:
        # Save to cache
        pass
```

### Integration Points
1. Modify `column_mapper.py` to use LLMColumnMapper as fallback
2. Add caching layer for LLM responses
3. Update validation pipeline to track original vs mapped names

## 3. Implementation Steps

1. Schema Profile Editor
   - Create SchemaProfileEditor class
   - Add profile CRUD operations
   - Implement preview functionality
   - Add export/import support

2. LLM Column Mapper
   - Create LLMColumnMapper class
   - Implement confidence scoring
   - Add caching system
   - Create LLM prompt templates

3. Integration
   - Update ColumnMapper to use LLM
   - Modify validation pipeline
   - Add tracking for mapped columns

4. UI Components
   - Create profile editor interface
   - Add live preview panel
   - Implement import/export UI
   - Add validation highlighting

## 4. Testing Strategy

1. Unit Tests
```python
def test_schema_profile_operations():
    editor = SchemaProfileEditor()
    profile = editor.create_profile(...)
    assert editor.validate_profile(profile)
    
def test_llm_mapping_fallback():
    mapper = LLMColumnMapper()
    result = mapper.get_mapping_suggestions(...)
    assert len(result) > 0
```

2. Integration Tests
```python
def test_end_to_end_mapping():
    # Test full pipeline with LLM fallback
    pass
    
def test_profile_import_export():
    # Test profile serialization
    pass
```

## 5. Validation Rules

1. Profile Validation
- Required fields present
- Valid column definitions
- Unique column names
- Valid role permissions

2. Import Validation
- Schema version check
- Required sections
- Data type validation
- Role validation

## 6. Cache Management

1. LLM Response Cache
- Cache key: hash(source_cols + target_schema)
- TTL: 24 hours
- Persistence: JSON files

2. Preview Cache
- Cache key: hash(profile + sample_data)
- TTL: 5 minutes
- Memory-only cache

## 7. Error Handling

1. Profile Errors
- Invalid schema format
- Missing required fields
- Permission conflicts

2. LLM Errors
- API failures
- Invalid responses
- Timeout handling

## 8. Performance Considerations

1. Preview Optimization
- Limit sample size
- Async validation
- Progressive loading

2. LLM Optimization
- Batch requests
- Response caching
- Confidence thresholds