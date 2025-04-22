# Lead Prediction System & Escalation Layer

This module implements a predictive system to forecast likely sales outcomes from incoming leads and builds an escalation system for high-risk cases.

## Components

### 1. LeadOutcomePredictor (`ml/lead_model.py`)

The prediction model uses machine learning to forecast sale probabilities for leads.

- **Features Used**:
  - `rep`: Sales representative assigned to the lead
  - `vehicle`: Vehicle/model of interest
  - `source`: Lead source channel
  - `hour_created`: Hour of day lead was created
  - `day_created`: Day of week lead was created
  - `contact_delay`: Hours between lead creation and first contact

- **Model Types**:
  - Random Forest (default)
  - Gradient Boosting
  - Logistic Regression

- **Prediction Timeframes**:
  - Short-term (14-day outcome)
  - Medium-term (30-day outcome)

- **Key Functionality**:
  - Train models on historical lead data
  - Predict sale probability for new leads
  - Optimize probability threshold
  - Calculate feature importance
  - Evaluate model performance

### 2. AlertEscalationRouter (`notifications/escalation.py`)

Routes high-risk leads to appropriate team members with configurable rules.

- **Escalation Levels**:
  - Low
  - Medium
  - High
  - Critical

- **Notification Channels**:
  - Email
  - SMS
  - Slack
  - Webhook
  - In-app notifications

- **Key Features**:
  - Risk-based lead routing
  - Configurable thresholds and delays
  - Support for after-hours routing
  - Fallback recipients
  - Detailed logging and statistics
  - Scheduled vs. immediate escalations

### 3. UI Configuration (`watchdog_ai/ui/pages/notification_settings.py`)

User interface for configuring the escalation system.

- **Configuration Options**:
  - Probability thresholds
  - Recipient management
  - Escalation delays
  - Working hours
  - Webhook integration
  - Advanced routing options

- **Monitoring Features**:
  - Recent escalations view
  - Statistics dashboard
  - Success rate tracking
  - Channel and level distribution

## How It Works

1. **Lead Data Analysis**:
   - Lead data is processed through the LeadFlowOptimizer
   - Historical patterns are identified (bottlenecks, conversion rates)

2. **Outcome Prediction**:
   - The trained machine learning model predicts sale probability
   - Leads are classified by risk level based on probability

3. **Escalation Routing**:
   - High-risk leads trigger escalations
   - Critical leads (very low probability) get immediate attention
   - Other escalations may be delayed based on configuration
   - Recipients are selected based on lead attributes and escalation level

4. **Notification Delivery**:
   - Notifications are sent via configured channels
   - Templates customize messages based on lead data
   - Retry logic ensures delivery

5. **Monitoring & Analytics**:
   - All escalations are tracked and logged
   - Success rates are calculated
   - UI provides visibility into escalation performance

## Integration Points

- Works with the existing lead flow optimizer
- Connects with notification service for delivery
- Compatible with existing user preference system
- UI integrated into notification settings tab

## Testing

- Unit tests for model and escalation components
- Integration tests for end-to-end flow
- Test functions to simulate lead escalation scenarios

## Configuration

Configuration options are available in the UI under "Notification Settings" > "Lead Escalation" tab. The system supports:

- Multiple probability thresholds for different risk levels
- Configurable delay times before escalation
- Recipient management for different escalation levels
- Working hours configuration for after-hours routing
- Webhook integration for external systems