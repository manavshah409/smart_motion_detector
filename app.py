# app.py
import os
import time
import json
import queue
import threading
import datetime
from flask import Flask, Response, request, jsonify, send_file, render_template

# project paths
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# import processor
from video_processing import VideoProcessor

# flask app
app = Flask(__name__)

# shared state
state_lock = threading.Lock()
event_bus = queue.Queue()
sse_queue = queue.Queue(maxsize=200)
recent_logs = []
recent_chart = []
MAX_LOGS = 50
MAX_CHART = 50

DEFAULT_SOURCE = "0"  # camera index 0

def format_sse(data: dict):
    return f"data: {json.dumps(data)}\n\n"

def event_consumer_thread_fn():
    while True:
        try:
            ev = event_bus.get()
            if ev is None:
                event_bus.task_done()
                continue

            # Accept dict or tuple (log, chart) shapes
            log = ev.get("log") if isinstance(ev, dict) else None
            chart = ev.get("chart") if isinstance(ev, dict) else None
            if not log and isinstance(ev, tuple) and len(ev) == 2:
                log, chart = ev

            if not log:
                log = {"timestamp": datetime.datetime.utcnow().isoformat(), "msg": str(ev)}

            with state_lock:
                recent_logs.insert(0, log)
                if len(recent_logs) > MAX_LOGS:
                    recent_logs.pop()
                if chart:
                    recent_chart.append(chart)
                    if len(recent_chart) > MAX_CHART:
                        recent_chart.pop(0)

            payload = {"log": log or {}, "chart": chart or {}}
            try:
                sse_queue.put_nowait(payload)
            except queue.Full:
                pass

            event_bus.task_done()
        except Exception as e:
            print("Event consumer error:", e)
            time.sleep(0.1)

threading.Thread(target=event_consumer_thread_fn, daemon=True).start()

# Processor lifecycle
processor_lock = threading.Lock()
processor = None

def on_source_ended_callback():
    # default behaviour: try to restart camera (safe no-op for camera)
    print("[app] source ended callback (no-op for live-only)")

def start_processor(source):
    global processor
    with processor_lock:
        if processor is not None:
            try:
                processor.close()
            except Exception:
                pass
            processor = None
            time.sleep(0.08)
        proc = VideoProcessor(video_source=source, event_bus=event_bus, on_ended=on_source_ended_callback)
        processor = proc
        print("[app] Processor started on:", source)
        return proc

def switch_video_source(new_source):
    return start_processor(new_source)

# start the camera processor
try:
    start_processor(DEFAULT_SOURCE)
except Exception as e:
    print("Warning: could not start default processor:", e)

@app.route("/")
def index():
    # If you have a template index.html in templates/ this will render it.
    # Otherwise return a tiny placeholder.
    try:
        return render_template("index.html")
    except Exception:
        # fallback simple page
        return render_template_string("""
            <h3>Smart Motion Dashboard</h3>
            <p>Open the UI served separately (static index.html). MJPEG: <img src="{{url_for('video_feed')}}"></p>
        """)

# MJPEG generator
def mjpeg_generator():
    global processor
    while True:
        with processor_lock:
            proc = processor
        if proc is None:
            time.sleep(0.1)
            continue
        try:
            for chunk in proc.get_frame_bytes():
                yield chunk
            time.sleep(0.1)
        except GeneratorExit:
            break
        except Exception as e:
            print("MJPEG generator error:", e)
            time.sleep(0.2)

@app.route("/video_feed")
def video_feed():
    return Response(mjpeg_generator(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/api/set_config", methods=["POST"])
def api_set_config():
    data = request.get_json() or {}
    threshold = data.get("threshold")
    min_area = data.get("min_area")
    with processor_lock:
        if processor:
            processor.update_config(threshold=threshold, min_area=min_area)
    return jsonify({"ok": True})

@app.route("/api/get_initial_data")
def api_get_initial_data():
    with state_lock:
        logs = list(recent_logs)[:MAX_LOGS]
        chart = list(recent_chart)[-MAX_CHART:]
    return jsonify({"logs": logs, "chart": chart})

@app.route("/api/event-stream")
def api_event_stream():
    def stream():
        while True:
            try:
                payload = sse_queue.get()
                if payload is None:
                    continue
                yield format_sse(payload)
                sse_queue.task_done()
            except GeneratorExit:
                break
            except Exception as e:
                print("SSE error:", e)
                time.sleep(0.2)
    return Response(stream(), mimetype="text/event-stream")

@app.route("/api/heatmap_image")
def api_heatmap_image():
    with processor_lock:
        proc = processor
    try:
        if proc:
            hm = proc.get_current_heatmap()
            if hm is not None:
                import cv2
                norm = cv2.normalize(hm, None, 0, 255, cv2.NORM_MINMAX)
                heatmap_u8 = norm.astype("uint8")
                colored = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_HOT)
                is_success, buffer = cv2.imencode(".png", colored)
                if is_success:
                    return Response(buffer.tobytes(), mimetype="image/png")
    except Exception as e:
        print("Error creating heatmap:", e)

    saved_path = os.path.join(OUTPUT_DIR, "motion_heatmap.png")
    if os.path.exists(saved_path):
        return send_file(saved_path, mimetype="image/png")
    return ("", 204)

@app.route("/api/current_source")
def api_current_source():
    with processor_lock:
        proc = processor
    if proc is None:
        return jsonify({"current_source": None})
    return jsonify({"current_source": str(getattr(proc, "video_source", "unknown"))})

@app.route("/api/shutdown", methods=["POST"])
def api_shutdown():
    global processor
    with processor_lock:
        if processor:
            try:
                processor.close()
            except Exception:
                pass
            processor = None
    return jsonify({"ok": True, "msg": "Processor stopped"})

if __name__ == "__main__":
    # debug True for development
    app.run(host="0.0.0.0", port=5000, debug=True, threaded=True)
