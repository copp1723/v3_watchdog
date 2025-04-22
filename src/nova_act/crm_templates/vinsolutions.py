"""
VinSolutions CRM navigation template.
"""

VINSOLUTIONS_TEMPLATE = {
    "login": {
        "steps": [
            "enter username {email} in #email field",
            "enter password in #password field",
            "click #loginBtn to submit form"
        ],
        "selectors": {
            "username": "#email",
            "password": "#password",
            "submit": "#loginBtn",
            "2fa_code": "#verificationCode",
            "2fa_submit": "#verifyBtn"
        }
    },
    "reports": {
        "steps": [
            "click #reportingDashboard to open dashboard",
            "click #salesOverview to select sales overview",
            "set date range in #reportDateRange",
            "click #exportBtn to download report"
        ],
        "selectors": {
            "dashboard": "#reportingDashboard",
            "sales_overview": "#salesOverview",
            "date_range": "#reportDateRange",
            "export": "#exportBtn"
        },
        "wait_conditions": {
            "dashboard_load": "#reportingDashboard.loaded",
            "export_ready": "#exportBtn:not(.disabled)"
        }
    },
    "error_indicators": {
        "login_failed": "#loginError",
        "session_expired": "#sessionExpired",
        "report_error": "#reportError"
    }
} 