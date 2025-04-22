#!/usr/bin/env python3
"""
Watchdog AI Application Runner
"""

import os
import sys
import signal
import subprocess
import psutil
import time

def kill_streamlit_processes():
    """Kill any existing Streamlit processes on ports 8501-8503."""
    ports = [8501, 8502, 8503]
    for port in ports:
        try:
            # Find process using port
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    cmdline = proc.info['cmdline']
                    if cmdline and '--server.port' in cmdline and str(port) in cmdline:
                        print(f"Killing Streamlit process on port {port} (PID: {proc.pid})")
                        proc.kill()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
        except Exception as e:
            print(f"Error killing process on port {port}: {e}")
    
    # Wait for processes to die
    time.sleep(2)

def setup_environment():
    """Set up the Python environment."""
    # Add src directory to Python path
    src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src')
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    os.environ['PYTHONPATH'] = src_path

def run_app():
    """Run the Streamlit application."""
    try:
        # Kill existing processes
        kill_streamlit_processes()
        
        # Set up environment
        setup_environment()
        
        # Start Streamlit
        print("Starting Streamlit app...")
        cmd = [
            "streamlit",
            "run",
            "src/watchdog_ai/ui/pages/main_app.py",
            "--server.port",
            "8503"
        ]
        
        process = subprocess.Popen(cmd)
        
        # Handle graceful shutdown
        def signal_handler(signum, frame):
            print("\nShutting down gracefully...")
            process.terminate()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Wait for process
        process.wait()
        
    except Exception as e:
        print(f"Error starting app: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_app() 