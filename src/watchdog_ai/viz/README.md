# Watchdog AI Visualization Layer

This package provides encapsulated chart builders with mobile-responsive sizing for Streamlit applications.

## Features

- **Mobile-responsive charts**: Automatically adapt to different screen sizes
- **Streamlined API**: Consistent interface across all chart types
- **Auto-detection**: Smart column type detection for date, category, and value fields
- **Tooltips and Interactivity**: Rich data exploration with hover and zoom
- **Error handling**: Graceful fallbacks when data is invalid

## Installation

The visualization layer is included as part of the Watchdog AI package:

```bash
pip install watchdog-ai
```

## Quick Start

```python
import streamlit as st
import pandas as pd
from watchdog_ai.viz import

