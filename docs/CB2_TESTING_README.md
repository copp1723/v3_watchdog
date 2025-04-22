# Nova Act Integration Testing Guide for CB 2

## Overview
This guide will help you test the end-to-end flow of the Nova Act integration, focusing on the SalesReportRenderer component and insight generation.

## Mock Data Setup

### Sales Report File
Location: `tests/fixtures/mock_sales_reports_123456789.csv`

This file contains 18 records of sales data with the following structure:
```csv
Lead Source,Sales Count,Avg Profit
NeoIdentity,4,2600
AutoTrader,3,2100
...
```

The data includes various lead sources with realistic sales counts and average profits, perfect for testing the SalesReportRenderer.

## Testing the Integration

### 1. Manual Testing with Mock Data

1. Start the Streamlit app:
   ```bash
   streamlit run app.py
   ```

2. Navigate to the "Connect My Systems" section

3. Use these test credentials:
   - Vendor: dealersocket
   - Dealer ID: test123
   - Username: test_user
   - Password: test_pass

4. Click "Run Sync Now" to trigger a manual sync

5. Verify the following:
   - Sync status shows "Connected"
   - Last sync time updates
   - Sales report data appears in the insights section

### 2. Testing SalesReportRenderer

The SalesReportRenderer should display:
- Sales performance chart
- Key metrics (total sales, average profit)
- Lead source analysis
- Recommendations

Verify that:
- Chart shows all 18 lead sources
- Metrics are calculated correctly
- Recommendations are relevant to the data

### 3. Testing Error Scenarios

1. Test with invalid credentials:
   - Use incorrect password
   - Verify error message appears
   - Check error styling (message-container error class)

2. Test with missing data:
   - Temporarily rename mock file
   - Verify fallback message appears
   - Restore mock file

## Automated Tests

Run the test suite:
```bash
pytest tests/e2e/test_nova_act_integration.py -v
```

Key test cases:
- End-to-end flow
- Error handling
- UI component rendering
- Metric calculations

## Scheduling Tests

1. Test daily scheduling:
   ```bash
   python scripts/start_nova_act.py
   ```

2. Set schedule frequency to "Daily" in UI

3. Verify:
   - Scheduler starts
   - Sync runs at scheduled time
   - UI updates with new data

## Common Issues & Solutions

### Missing Mock Data
If mock data file is missing:
```bash
cp tests/fixtures/mock_sales_reports_123456789.csv /tmp/
```

### Scheduler Not Running
Check logs:
```bash
tail -f logs/nova_act_scheduler.log
```

### UI Not Updating
Try:
1. Clear browser cache
2. Restart Streamlit app
3. Check WebSocket connection

## Need Help?

If you encounter any issues:
1. Check the logs in `logs/`
2. Review error messages in UI
3. Contact Cursor Dev team with:
   - Error message
   - Steps to reproduce
   - Log snippets

## Next Steps

After successful testing:
1. Document any issues found
2. Note any UI/UX improvements needed
3. Prepare for Phase 2 enhancements
4. Share feedback with the team 