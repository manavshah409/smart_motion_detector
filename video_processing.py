# video_processing.py
import cv2
import imutils
import datetime
import os
import numpy as np
import threading
import queue
import time
from ultralytics import YOLO
from data_logger import EventLogger
from utils import RECORDINGS_DIR, SNAPSHOTS_DIR, make_filename, make_snapshot_name

class VideoProcessor:
    def __init__(self, video_source, event_bus, on_ended=None):
        """
        video_source: "0" (string) or an index string, or file path.
        event_bus: queue.Queue for events.
        on_ended: optional callable called when a file source finishes.
        """
        print("[VideoProcessor] init source:", video_source)
        self.video_source = int(video_source) if str(video_source).isdigit() else str(video_source)
        self.event_bus = event_bus
        self.on_ended = on_ended

        # open capture (prefer default backend for camera)
        try:
            if isinstance(self.video_source, str) and self.video_source.lower().endswith((".mp4", ".mov", ".avi", ".mkv")):
                self.cap = cv2.VideoCapture(self.video_source, cv2.CAP_FFMPEG)
            else:
                self.cap = cv2.VideoCapture(self.video_source)
        except Exception as e:
            print("[VideoProcessor] capture open error:", e)
            self.cap = cv2.VideoCapture(self.video_source)

        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        if not self.cap.isOpened():
            raise IOError(f"Cannot open video source: {self.video_source}")

        # load YOLO (optional; errors degrade to simple 'motion' class)
        try:
            self.model = YOLO('yolov8n.pt')
            print("[VideoProcessor] YOLOv8 loaded")
        except Exception as e:
            print("[VideoProcessor] YOLO load failed:", e)
            self.model = None

        self.config_lock = threading.Lock()
        self.MIN_AREA = 800
        self.THRESHOLD_VAL = 25

        self.first_frame = None
        self.logger = EventLogger(event_bus=event_bus)
        self.motion_start = None
        self.out_writer = None
        self.fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.motion_heatmap = None
        self.current_event_class = "unknown"

        self.io_queue = queue.Queue()
        self.worker_thread = threading.Thread(target=self._io_worker, daemon=True)
        self.worker_thread.start()

        self._running = True
        print("[VideoProcessor] initialized")

    def update_config(self, threshold=None, min_area=None):
        with self.config_lock:
            if threshold is not None:
                try:
                    self.THRESHOLD_VAL = int(threshold)
                except Exception:
                    pass
            if min_area is not None:
                try:
                    self.MIN_AREA = int(min_area)
                except Exception:
                    pass
        print(f"[VideoProcessor] updated config threshold={self.THRESHOLD_VAL} min_area={self.MIN_AREA}")

    def get_current_heatmap(self):
        if self.motion_heatmap is None:
            return None
        with self.config_lock:
            return self.motion_heatmap.copy()

    def _classify_frame(self, frame):
        if self.model is None:
            return "motion"
        try:
            results = self.model(frame, verbose=False)
            if len(results) == 0 or len(results[0].boxes) == 0:
                return "motion"
            top_box = results[0].boxes[0]
            class_id = int(top_box.cls)
            class_name = self.model.names.get(class_id, "obj")
            return class_name.capitalize()
        except Exception as e:
            print("[VideoProcessor] classification error:", e)
            return "error"

    def _io_worker(self):
        print("[VideoProcessor] io worker started")
        while True:
            try:
                task, data = self.io_queue.get()
                if task == "LOG_FRAME":
                    ts, score = data
                    self.logger.log_frame_motion(ts, score)
                elif task == "LOG_EVENT":
                    start_time, end_time, classification = data
                    self.logger.end_event(start_time, end_time, classification)
                elif task == "WRITE_FRAME":
                    frame = data
                    if self.out_writer:
                        self.out_writer.write(frame)
                elif task == "SAVE_SNAPSHOT":
                    frame = data
                    snap_name = make_snapshot_name()
                    cv2.imwrite(os.path.join(SNAPSHOTS_DIR, snap_name), frame)
                elif task == "START_WRITER":
                    frame_shape = data
                    fname = make_filename()
                    out_path = os.path.join(RECORDINGS_DIR, fname)
                    self.out_writer = cv2.VideoWriter(out_path, self.fourcc, 20.0, (frame_shape[1], frame_shape[0]))
                elif task == "STOP_WRITER":
                    if self.out_writer:
                        try:
                            self.out_writer.release()
                        except Exception:
                            pass
                        self.out_writer = None
                elif task == "STOP":
                    break
                self.io_queue.task_done()
            except Exception as e:
                print("[VideoProcessor] io worker error:", e)

    def _process_frame(self):
        try:
            ret, frame = self.cap.read()
        except AssertionError as e:
            print("[VideoProcessor] FFmpeg assertion caught:", e)
            time.sleep(0.05)
            try:
                ret, frame = self.cap.read()
            except Exception as e2:
                print("[VideoProcessor] second read failed:", e2)
                return False, None, False, 0.0
        except Exception as e:
            print("[VideoProcessor] capture read error:", e)
            return False, None, False, 0.0

        if not ret:
            # for file sources: call on_ended (not used for camera)
            if isinstance(self.video_source, str) and os.path.exists(self.video_source):
                try:
                    if callable(self.on_ended):
                        try:
                            threading.Thread(target=self.on_ended, daemon=True).start()
                        except Exception:
                            try:
                                self.on_ended()
                            except Exception:
                                pass
                except Exception as e:
                    print("[VideoProcessor] on_ended error:", e)
                return False, None, False, 0.0
            # try reset for camera
            try:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.cap.read()
            except Exception:
                return False, None, False, 0.0
            if not ret:
                return False, None, False, 0.0

        try:
            frame = imutils.resize(frame, width=480)
        except Exception:
            return False, None, False, 0.0

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        if self.first_frame is None:
            self.first_frame = gray.copy().astype("float")
            return True, frame, False, 0.0

        cv2.accumulateWeighted(gray, self.first_frame, 0.1)
        frame_delta = cv2.absdiff(gray, cv2.convertScaleAbs(self.first_frame))

        with self.config_lock:
            current_threshold = self.THRESHOLD_VAL
            current_min_area = self.MIN_AREA

        thresh = cv2.threshold(frame_delta, current_threshold, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        thresh_norm = thresh.astype("float") / 255.0

        with self.config_lock:
            if self.motion_heatmap is None:
                h, w = frame.shape[:2]
                self.motion_heatmap = np.zeros((h, w), dtype="float")
            h_t, w_t = thresh_norm.shape
            self.motion_heatmap[0:h_t, 0:w_t] += thresh_norm

        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        motion_detected = False
        total_motion_score = 0.0
        for c in contours:
            area = cv2.contourArea(c)
            total_motion_score += area
            if area < current_min_area:
                continue
            motion_detected = True
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        return True, frame, motion_detected, total_motion_score

    def _manage_events_and_recording(self, frame, motion_detected, score):
        ts = datetime.datetime.now()
        if motion_detected:
            if self.motion_start is None:
                self.motion_start = ts
                self.current_event_class = self._classify_frame(frame)
                print(f"[VideoProcessor] motion started {self.motion_start} class={self.current_event_class}")
                self.logger.start_event()
                self.io_queue.put(("START_WRITER", frame.shape))
                self.io_queue.put(("SAVE_SNAPSHOT", frame.copy()))
            self.logger.add_score_to_event(score)
            self.io_queue.put(("WRITE_FRAME", frame.copy()))
        else:
            if self.motion_start is not None:
                motion_end = ts
                self.io_queue.put(("LOG_EVENT", (self.motion_start, motion_end, self.current_event_class)))
                self.motion_start = None
                self.current_event_class = "unknown"
                self.io_queue.put(("STOP_WRITER", None))

    def get_frame_bytes(self):
        while self._running:
            try:
                ret, frame, motion_detected, score = self._process_frame()
                if not ret:
                    time.sleep(0.03)
                    continue
                try:
                    self._manage_events_and_recording(frame, motion_detected, score)
                except Exception as e:
                    print("[VideoProcessor] manage events error:", e)

                status_text = 'Motion Detected' if motion_detected else 'No Motion'
                if motion_detected and self.current_event_class != "unknown":
                    status_text = f"Motion: {self.current_event_class}"
                color = (0, 0, 255) if motion_detected else (255, 255, 255)
                cv2.putText(frame, status_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                (flag, encodedImage) = cv2.imencode(".jpg", frame)
                if not flag:
                    continue

                chunk = b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n'
                yield chunk
            except Exception as e:
                print("[VideoProcessor] frame generator exception:", e)
                time.sleep(0.05)
                continue

    def close(self):
        print("[VideoProcessor] closing")
        self._running = False
        try:
            self.io_queue.put(("STOP", None))
        except Exception:
            pass
        try:
            self.worker_thread.join(timeout=2.0)
        except Exception:
            pass
        try:
            self.cap.release()
        except Exception:
            pass

        with self.config_lock:
            heatmap_copy = self.motion_heatmap.copy() if self.motion_heatmap is not None else None

        if heatmap_copy is not None and np.sum(heatmap_copy) > 0:
            try:
                import matplotlib.pyplot as plt
                heatmap_img = cv2.normalize(heatmap_copy, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
                plt.figure(figsize=(10, 8))
                plt.imshow(heatmap_img, cmap='hot')
                plt.colorbar()
                os.makedirs("output", exist_ok=True)
                plt.savefig(os.path.join("output", "motion_heatmap.png"))
                plt.close()
                print("[VideoProcessor] final heatmap saved")
            except Exception as e:
                print("[VideoProcessor] final heatmap save failed:", e)
