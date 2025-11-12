# Save as data_logger.py
import os
import datetime
import pandas as pd
import numpy as np
from utils import LOGS_DIR
import json # <-- NEW

class EventLogger:
    def __init__(self, event_bus):
        self.frame_log_path = os.path.join(LOGS_DIR, "frames_log.csv")
        self.event_log_path = os.path.join(LOGS_DIR, "motion_log.csv")
        self.current_motion_scores = []
        
        # --- NEW: Add a reference to the live event bus ---
        self.event_bus = event_bus
        # --- End New ---
        
        if not os.path.exists(self.frame_log_path):
            pd.DataFrame(columns=["timestamp_iso", "motion_score"]).to_csv(self.frame_log_path, index=False)
        
        if not os.path.exists(self.event_log_path):
            pd.DataFrame(columns=["start_time", "end_time", "avg_intensity", "classification"]).to_csv(self.event_log_path, index=False)

    def log_frame_motion(self, ts, score):
        try:
            with open(self.frame_log_path, 'a') as f:
                f.write(f"{ts.isoformat()},{score}\n")
        except Exception as e:
            print(f"⚠️ Error logging frame: {e}")

    def start_event(self):
        self.current_motion_scores = []

    def add_score_to_event(self, score):
        self.current_motion_scores.append(score)

    def end_event(self, start_time, end_time, classification="unknown"):
        avg_int = np.mean(self.current_motion_scores) if self.current_motion_scores else 0.0
        
        # --- UPDATED: Log to file AND publish to live event bus ---
        try:
            # 1. Log to the CSV file as normal
            with open(self.event_log_path, 'a') as f:
                f.write(f"{start_time.isoformat()},{end_time.isoformat()},{avg_int:.2f},{classification}\n")
            
            print(f"✅ Event logged: {start_time.strftime('%H:%M:%S')} to {end_time.strftime('%H:%M:%S')} (Avg Score: {avg_int:.2f}, Class: {classification})")
            
            # 2. Create the data packet for the dashboard
            event_data = {
                "log": {
                    "start": start_time.strftime('%H:%M:%S'),
                    "end": end_time.strftime('%H:%M:%S'),
                    "intensity": f"{avg_int:.0f}",
                    "classification": classification
                },
                "chart": {
                    "label": start_time.strftime('%H:%M:%S'),
                    "data": avg_int,
                }
            }
            
            # 3. Put the event on the live bus for the web server to find
            self.event_bus.put(event_data)

        except Exception as e:
            print(f"⚠️ Error logging or publishing event: {e}")
        # --- End Update ---
        
        self.current_motion_scores = []