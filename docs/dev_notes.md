# Developer Notes

## Architecture Patterns

### Insight Intent Architecture
- All analysis lives in Intent classes implementing matches() and analyze()
- ConversationManager routes prompts to matching Intents
- Each Intent returns a structured InsightResult
- UI renders InsightResult without knowledge of analysis details
- New analytics = new Intent class, no UI changes needed

### Column Name Normalization
- Centralize column aliases in `src/utils/columns.py`
- Support multiple formats through multi-step normalization
- Store both original and normalized names
- Enhanced alias matching with progressive fallbacks:
  1. Exact matches in aliases list
  2. Normalized canonical names
  3. Normalized aliases
  4. Partial matches

### Supported Aliases
Metrics:
- Gross: total_gross, gross_profit, total gross, etc.
- Price: sale_price, list_price, price, etc.
- Revenue: total_revenue, rev, revenue, etc.

Categories:
- Rep: sales_rep, salesperson, representative, etc.
- Source: lead_source, channel, origin, etc.
- Make: manufacturer, brand, car_make, etc.

### Adding New Intents
1. Create new class inheriting from Intent
2. Implement matches() to detect relevant prompts
3. Implement analyze() to process data
4. Return InsightResult with:
   - title & summary
   - recommendations
   - optional chart_data & encoding
   - optional supporting_data
5. Register in ConversationManager.intents list

### Data I/O
- Use `@st.cache_data` for data loading and processing functions
- Centralize data validation in `data_io.py`
- Define required columns as constants
- Provide clear validation summaries with metrics

### Schema Validation
- Define required columns at module level
- Validate schema before processing data
- Check for missing and invalid values
- Calculate and display quality scores
- Support multiple column name formats through normalization

### Visualization
- Use Altair for interactive charts
- Enable tooltips for detailed metrics
- Make charts responsive with `use_container_width=True`
- Place charts below insights for context

### Theme Management
- Centralize theme tokens in `theme.py`
- Use CSS variables for dynamic theming
- Group tokens by purpose (colors, spacing, etc.)
- Include mobile breakpoints

## UI Components

### Chat Interface
- Use `st.chat_input` and `st.chat_message` for chat-like interactions
- Store example queries in a constant at the module level
- Use `st.spinner` with descriptive messages for loading states
- Add download buttons for data exports with timestamped filenames

### Animation
- Use CSS `animation` property with `fadeIn` keyframes for smooth transitions
- Keep animations subtle and purposeful (0.5s - 1s duration)
- Add hover effects for interactive elements
- Consider mobile performance when adding animations

### Mobile Responsiveness
- Use media queries for screens under 768px
- Adjust padding and margins for touch targets
- Scale down font sizes and icons
- Ensure buttons remain usable on small screens

### State Management
- Initialize all session state variables in one place
- Use descriptive keys for session state
- Clear temporary state after use
- Store data context with insights for downloads

### Branding Integration (New)
- Logo file (`watchdog_logo.png`) added to `/assets/`.
- Header refactored to use Streamlit native layout (`st.columns`, `st.image`) in `src/ui/pages/modern_ui.py` instead of custom HTML/CSS.
    - Logo loaded via `PIL.Image.open`.
    - `st.image` used for rendering (width set to 180px).
    - `st.columns` used for side-by-side layout ([1, 4] ratio).
    - Title and subtitle rendered using `st.markdown` with inline styles.
- Removed old CSS rules (`.header-container`, `.logo`, `.title-block`, `.watchdog-vibe`) from `modern_ui.py`.
- Added CSS rule to align items in header columns to `flex-start`.
- *Note: Responsive logo resizing (80px mobile) specified in ticket is not directly implemented via CSS due to `st.image` fixed width; relies on column stacking.* 
- *Note: `st.image` does not support `alt` text directly; accessibility relies on context.* 

## Best Practices

### Loading States
- Always show loading indicators for async operations
- Use descriptive messages in spinners
- Disable inputs during loading
- Provide clear feedback on completion

### User Guidance
- Offer example queries as clickable buttons
- Use clear, action-oriented button labels
- Provide tooltips for complex features
- Include download options for data exports

### Error Handling
- Show clear error messages with supported formats
- Log errors for debugging
- Gracefully handle missing data
- Provide recovery options when possible
- Include column name suggestions in validation errors

### Performance
- Cache expensive operations with `@st.cache_data`
- Minimize state updates and reruns
- Use lazy loading for large datasets
- Optimize chart rendering for mobile