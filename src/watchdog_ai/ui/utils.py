import os
from typing import Any, Dict, Optional
import json
from datetime import datetime

FEEDBACK_FILE = os.path.join(os.path.dirname(__file__), '../../../../data/feedback.json')

def save_feedback(insight_id: str, useful: bool, timestamp: Optional[str] = None, extra: Optional[Dict[str, Any]] = None) -> None:
    """
    Save feedback for an insight to feedback.json. Appends a record keyed by insight_id and timestamp.
    """
    feedback = {
        "insight_id": insight_id,
        "useful": useful,
        "timestamp": timestamp or datetime.now().isoformat(),
    }
    if extra:
        feedback.update(extra)
    # Ensure feedback file exists and is a list
    if os.path.exists(FEEDBACK_FILE):
        try:
            with open(FEEDBACK_FILE, 'r') as f:
                data = json.load(f)
            if not isinstance(data, list):
                data = []
        except Exception:
            data = []
    else:
        data = []
    data.append(feedback)
    with open(FEEDBACK_FILE, 'w') as f:
        json.dump(data, f, indent=2)