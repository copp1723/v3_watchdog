"""
Validators package for Watchdog AI.

This package contains validation modules that analyze and flag issues in automotive dealer data.
"""

# Import key validation functionality for easy access
from .insight_validator import (
    flag_all_issues,
    summarize_flags,
    generate_flag_summary,
    highlight_flagged_rows
)

from .validation_profile import (
    ValidationProfile,
    ValidationRule,
    ValidationRuleType,
    get_available_profiles,
    apply_validation_profile,
    render_profile_editor,
    create_default_profile,
    create_minimal_profile,
    create_comprehensive_profile
)

# Export a unified validation service
from .validator_service import (
    ValidatorService,
    process_uploaded_file
)
