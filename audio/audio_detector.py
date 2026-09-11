import threading
import time
import numpy as np

class AudioShoutingDetector:
    """
    Lightweight, non-blocking real-time audio shouting / yelling detector.
    Samples microphone input via sounddevice, continuously tracks ambient noise floor,
    and flags sudden shouting/loud vocal aggression.
    Fails gracefully if no microphone is present or disabled.
    """
    def __init__(self, sample_rate=16000, block_size=1024, threshold_factor=2.5, enabled=True):
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.threshold_factor = threshold_factor
        self.enabled = enabled

        self.current_rms = 0.0
        self.baseline_noise = 0.005
        self.shouting_score = 0.0
        self.is_shouting = False

        self.stream = None
        self.running = False
        self.thread = None
        self.available = False

        if self.enabled:
            self.start()

    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            pass
        # Calculate RMS energy of current audio block
        audio_data = indata[:, 0] if indata.ndim > 1 else indata
        rms = np.sqrt(np.mean(audio_data ** 2))
        self.current_rms = float(rms)

        # Adaptively update baseline noise floor (slow EMA)
        if self.current_rms < self.baseline_noise * 1.5:
            self.baseline_noise = 0.98 * self.baseline_noise + 0.02 * self.current_rms
        else:
            self.baseline_noise = 0.999 * self.baseline_noise + 0.001 * self.current_rms

        self.baseline_noise = max(0.002, self.baseline_noise)

        # Compute shouting score (0.0 to 1.0)
        ratio = self.current_rms / self.baseline_noise
        if ratio > self.threshold_factor and self.current_rms > 0.02:
            self.is_shouting = True
            self.shouting_score = min(1.0, (ratio - self.threshold_factor) / 3.0)
        else:
            self.is_shouting = False
            self.shouting_score = max(0.0, self.shouting_score * 0.85)

    def start(self):
        try:
            import sounddevice as sd
            # Test if a default input device exists
            devices = sd.query_devices()
            default_in = sd.default.device[0]
            if default_in is None or default_in < 0:
                # Find any input device
                input_devs = [i for i, d in enumerate(devices) if d.get('max_input_channels', 0) > 0]
                if not input_devs:
                    print("[INFO] [Audio] No microphone input device found. Audio detection disabled.")
                    self.available = False
                    return
                default_in = input_devs[0]

            self.stream = sd.InputStream(
                samplerate=self.sample_rate,
                blocksize=self.block_size,
                channels=1,
                dtype='float32',
                callback=self._audio_callback
            )
            self.stream.start()
            self.running = True
            self.available = True
            print("[AUDIO] Real-time audio aggression & shouting detector ACTIVE.")
        except Exception as e:
            print(f"[INFO] [Audio] Microphone not initialized ({e}). Audio detection disabled.")
            self.available = False

    def get_status(self):
        """
        Returns (is_shouting: bool, shouting_score: float, current_rms: float, available: bool)
        """
        if not self.available:
            return False, 0.0, 0.0, False
        return self.is_shouting, self.shouting_score, self.current_rms, True

    def stop(self):
        if self.stream and self.running:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
            self.running = False
