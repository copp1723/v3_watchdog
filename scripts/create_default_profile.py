# scripts/create_default_profile.py
import os
import sys

# Ensure the src directory is in the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(project_root, 'src'))

from validators.validation_profile import create_default_profile

# Define the profiles directory relative to the project root
PROFILES_DIR = os.path.join(project_root, "profiles")

print(f"Creating default profile in: {PROFILES_DIR}")

# Create the profile object
profile = create_default_profile()

# Ensure the directory exists
os.makedirs(PROFILES_DIR, exist_ok=True)

# Construct the full path for the default profile
# Using the profile's ID ('default') as the filename
default_profile_path = os.path.join(PROFILES_DIR, f"{profile.id}.json")

# Save the profile
saved_path = profile.save(PROFILES_DIR) # profile.save already joins dir and id.json

print(f"Default profile saved to: {saved_path}") 