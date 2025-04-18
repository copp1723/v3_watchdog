"""
E2E test configuration and fixtures.
"""

import pytest
import subprocess
import time
import requests
import os
import signal
from typing import Generator

@pytest.fixture(scope="session")
def streamlit_server() -> Generator[None, None, None]:
    """
    Fixture to start and stop the Streamlit server for E2E tests.
    """
    # Start server
    server = subprocess.Popen(
        ["streamlit", "run", "src/ui/streamlit_app.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        preexec_fn=os.setsid  # Use process group
    )
    
    # Wait for server to be ready
    timeout = 30
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get("http://localhost:8501/_stcore/health")
            if response.status_code == 200:
                break
        except requests.exceptions.ConnectionError:
            time.sleep(1)
    else:
        os.killpg(os.getpgid(server.pid), signal.SIGTERM)
        raise TimeoutError("Streamlit server failed to start")
    
    # Run tests
    yield
    
    # Cleanup
    os.killpg(os.getpgid(server.pid), signal.SIGTERM)