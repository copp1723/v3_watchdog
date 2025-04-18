import os
from setuptools import setup, find_packages

# Read version from VERSION file
with open("VERSION", "r") as f:
    version = f.read().strip()

# Read long description from README.md
with open("README.md", "r") as f:
    long_description = f.read()

# Read requirements from requirements.txt
with open("requirements.txt", "r") as f:
    requirements = f.read().splitlines()

setup(
    name="v3watchdog_ai",
    version=version,
    description="Intelligent analytics platform for automotive dealerships, powered by AI.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Watchdog AI Team",
    author_email="support@watchdogai.com",
    url="https://github.com/yourusername/v3watchdog_ai",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    install_requires=requirements,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Office/Business :: Financial",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.9",
    entry_points={
        "console_scripts": [
            "watchdog=src.app:main",
        ],
    },
)