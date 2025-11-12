# Smart Person-Detection Dashboard

This is a real-time smart surveillance system that uses YOLOv8 for person
detection and tracking, logging all events, and streaming the
results to a live web dashboard.

## Features
- Real-time person detection with YOLOv8.
- Tracks unique individuals with bounding boxes and IDs.
- Streams the live video feed to a web dashboard.
- Generates a live-updating chart of motion events.
- Automatically records video clips and snapshots of detected events.
- Generates a motion heatmap upon exit.

## Tech Stack
- **Backend:** Python, Flask, OpenCV, Ultralytics (YOLOv8)
- **Frontend:** HTML5, Chart.js

## How to Run

1.  **Clone the repository:**
    ```bash
    git clone [your-repo-link]
    cd smart-detector
    ```

2.  **Create a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: .\venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the web server:**
    ```bash
    python app.py
    ```

5.  **Open the dashboard:**
    Open your web browser and go to `http://127.0.0.1:5000`