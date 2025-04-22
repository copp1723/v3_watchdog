#!/usr/bin/env python3
"""
Test script to verify that all imports are working correctly.
Run this script from the project root to verify import paths.
"""

import sys
import os

print("Python path:", sys.path)
print("Current directory:", os.getcwd())

# Try importing from the validators package
print("\nTesting validators imports...")
try:
    from validators import ValidationProfile, get_available_profiles
    print("✅ Successfully imported ValidationProfile and get_available_profiles")
except ImportError as e:
    print(f"❌ Error importing from validators: {e}")

# Try importing from the ui package
print("\nTesting UI component imports...")
try:
    from ui.components import render_data_upload, render_flag_summary
    print("✅ Successfully imported render_data_upload and render_flag_summary")
except ImportError as e:
    print(f"❌ Error importing from ui.components: {e}")

# Try importing other modules
print("\nTesting other module imports...")
try:
    from insight_conversation import ConversationManager
    print("✅ Successfully imported ConversationManager")
except ImportError as e:
    print(f"❌ Error importing ConversationManager: {e}")

try:
    from insight_card import render_insight_card
    print("✅ Successfully imported render_insight_card")
except ImportError as e:
    print(f"❌ Error importing render_insight_card: {e}")

print("\nImport test complete.")
