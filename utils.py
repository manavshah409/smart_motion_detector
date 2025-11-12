# Save as utils.py
import os
import datetime

# Define central output directories
BASE_DIR = "output"
LOGS_DIR = os.path.join(BASE_DIR, "logs")
RECORDINGS_DIR = os.path.join(BASE_DIR, "recordings")
SNAPSHOTS_DIR = os.path.join(BASE_DIR, "snapshots")

def ensure_dirs():
    """Ensures all necessary output directories exist."""
    os.makedirs(LOGS_DIR, exist_ok=True)
    os.makedirs(RECORDINGS_DIR, exist_ok=True)
    os.makedirs(SNAPSHOTS_DIR, exist_ok=True)
    print("✅ Output directories ensured.")

def make_filename(prefix="rec", ext="avi"):
    """Generates a timestamped filename for recordings."""
    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return f"{prefix}_{ts}.{ext}"

def make_snapshot_name(prefix="snap", ext="jpg"):
    """Generates a timestamped filename for snapshots."""
    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return f"{prefix}_{ts}.{ext}"