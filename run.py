"""
run.py — Simple startup script.
Usage: poetry run python run.py
"""

import os
import subprocess
import sys
import webbrowser
import threading
import time


def open_browser():
    """Wait for server to start then open browser automatically."""
    time.sleep(2)
    webbrowser.open("http://127.0.0.1:8000/docs")


if __name__ == "__main__":
    print("\n========================================")
    print("  Legal Contract Entity Extractor")
    print("  Starting server...")
    print("  URL: http://127.0.0.1:8000/docs")
    print("  Press Ctrl+C to stop")
    print("========================================\n")

    # Open browser automatically after 2 seconds
    threading.Thread(target=open_browser, daemon=True).start()

    # Start the server
    subprocess.run([
        sys.executable, "-m", "uvicorn",
        "app.main:app",
        "--reload",
        "--host", "127.0.0.1",
        "--port", "8000",
    ])