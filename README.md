# ExamShield AI – Intelligent Smart Proctoring System

ExamShield AI is a real-time AI-powered exam monitoring dashboard built in Python with OpenCV, YOLOv8, Tkinter, typing analytics, evidence logging, and report export.

## Features

- Real-time webcam monitoring with Haar Cascade face detection
- Multiple face detection with evidence capture and alarms
- Gaze direction analysis (left/right/center) with time-based away alerts
- No-face detection warnings and evidence saving
- YOLOv8 object detection for phones, books, laptops, and devices
- Keyboard-based typing speed analysis and suspicious typing detection
- Live cheating score engine with risk thresholds
- Dark-themed Tkinter dashboard with event history and real-time metrics
- Excel and PDF report generation into the `reports/` folder

## Installation

1. Install Python 3.11+.
2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```
4. Start the Chrome-ready web dashboard:
   ```bash
   python chrome_dashboard.py
   ```
5. Open Google Chrome to `http://127.0.0.1:5000/`.

## Runtime Notes

- The first run may download `yolov8n.pt` automatically through Ultralytics.
- Evidence screenshots are stored under `evidence/` subfolders.
- Generated Excel/PDF reports are saved into `reports/`.

## Project Structure

- `chrome_dashboard.py` – Flask web dashboard optimized for Chrome
- `proctor_engine.py` – backend AI monitoring, detection, and evidence engine
- `dashboard.py` – legacy desktop dashboard
- `report_generator.py` – Excel and PDF report export utilities
- `templates/` – browser dashboard HTML
- `static/` – browser dashboard assets
- `requirements.txt` – Python dependencies
- `evidence/` – suspicious behavior screenshots
- `reports/` – generated reports
- `assets/` – assets and support resources
