import datetime
import os

LOG_FILE = "events.log"

def log_event(message, severity="INFO", risk_score=None, snapshot_path=None):
    """
    Logs structured events to events.log and console.
    """
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{now}] [{severity.upper()}] {message}"
    if risk_score is not None:
        log_entry += f" | Risk: {risk_score * 100:.1f}%"
    if snapshot_path:
        log_entry += f" | Snapshot: {snapshot_path}"
    log_entry += "\n"

    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(log_entry)
    except Exception as e:
        print(f"Log write error: {e}")

def get_recent_logs(limit=10):
    """Returns the most recent log entries."""
    if not os.path.exists(LOG_FILE):
        return []
    try:
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            lines = f.readlines()
            return lines[-limit:]
    except Exception:
        return []