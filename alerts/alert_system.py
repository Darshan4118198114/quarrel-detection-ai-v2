import os
import time
import datetime
import threading
import cv2

class AlertManager:
    """
    Production alert manager supporting:
    - Audio chime / siren (non-blocking thread)
    - Automated incident snapshot capture with bounding boxes and timestamp
    - Cooldown throttling to prevent spamming
    """
    def __init__(self, incidents_dir="incidents", cooldown_seconds=5.0, sound_enabled=True):
        self.incidents_dir = incidents_dir
        self.cooldown_seconds = cooldown_seconds
        self.sound_enabled = sound_enabled
        self.last_alert_time = 0.0

        os.makedirs(self.incidents_dir, exist_ok=True)

    def _play_sound(self):
        try:
            import winsound
            # Two-tone alert beep (Windows)
            winsound.Beep(1200, 250)
            time.sleep(0.05)
            winsound.Beep(1600, 300)
        except Exception:
            # Fallback console bell
            print('\a', end='', flush=True)

    def trigger_alert(self, frame=None, risk_score=1.0, details=None):
        now = time.time()
        if (now - self.last_alert_time) < self.cooldown_seconds:
            return None  # In cooldown period

        self.last_alert_time = now
        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        print(f"\n[ALERT TRIGGERED] Quarrel / Physical Confrontation Detected! (Risk: {risk_score*100:.1f}%)")

        # Play audible alert in background thread
        if self.sound_enabled:
            sound_thread = threading.Thread(target=self._play_sound, daemon=True)
            sound_thread.start()

        snapshot_path = None
        if frame is not None:
            snapshot_filename = f"quarrel_{timestamp_str}.jpg"
            snapshot_path = os.path.join(self.incidents_dir, snapshot_filename)
            cv2.imwrite(snapshot_path, frame)
            print(f"[SNAPSHOT] Incident image saved to: {snapshot_path}")

        return snapshot_path


# Global default manager
_default_alert_manager = AlertManager()

def send_alert(frame=None, risk_score=1.0, details=None):
    """
    Backward-compatible alert trigger function.
    """
    return _default_alert_manager.trigger_alert(frame, risk_score, details)