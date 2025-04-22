"""
DealerSocket CRM navigation template.
"""

DEALERSOCKET_TEMPLATE = {
    "login": {
        "steps": [
            "enter username {email} in #username field",
            "enter password in #password field",
            "click #login-button to submit form"
        ],
        "selectors": {
            "username": "#username",
            "password": "#password",
            "submit": "#login-button",
            "2fa_code": "#verification-code",
            "2fa_submit": "#verify-button"
        }
    },
    "reports": {
        "steps": [
            "click #reports-menu to open reports section",
            "click #sales-reports-tab to select sales reports",
            "set date range in #date-range-picker",
            "click #download-csv to export report"
        ],
        "selectors": {
            "menu": "#reports-menu",
            "sales_tab": "#sales-reports-tab",
            "date_picker": "#date-range-picker",
            "download": "#download-csv"
        },
        "wait_conditions": {
            "menu_load": "#reports-menu:not(.loading)",
            "download_ready": "#download-csv:not(.disabled)"
        }
    },
    "error_indicators": {
        "login_failed": ".error-message",
        "session_expired": "#session-timeout-dialog",
        "report_error": "#error-notification"
    }
} 