import base64
import os
import threading
import webbrowser

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request

from proctor_engine import ProctorEngine
from report_generator import generate_reports


app = Flask(__name__)
engine = ProctorEngine()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/status')
def status():
    return jsonify(engine.get_status())


@app.route('/events')
def events():
    return jsonify({"events": engine.state.event_history[-20:]})


@app.route('/process_frame', methods=['POST'])
def process_frame():
    try:
        payload = request.get_json(force=True)
        image_data = payload.get('image', '')
        if not image_data:
            return jsonify({"error": "No image data received"}), 400

        header, encoded = image_data.split(',', 1)
        image_bytes = base64.b64decode(encoded)
        np_arr = np.frombuffer(image_bytes, dtype=np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({"error": "Failed to decode frame"}), 400

        annotated_frame, status = engine.process_frame(frame)
        success, encoded_frame = cv2.imencode('.jpg', annotated_frame)
        if not success:
            return jsonify({"error": "Failed to encode annotated frame"}), 500

        return jsonify({
            **status,
            "annotated_frame": f"data:image/jpeg;base64,{base64.b64encode(encoded_frame.tobytes()).decode('utf-8')}",
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route('/generate_report', methods=['POST'])
def generate_report():
    try:
        excel_path, pdf_path = generate_reports(engine.state.event_history, engine.state.cheating_score, "reports")
        return jsonify({
            "excel_path": excel_path,
            "pdf_path": pdf_path,
            "status": "generated",
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route('/reset', methods=['POST'])
def reset_monitoring():
    engine.reset()
    return jsonify({"status": "reset"})


if __name__ == '__main__':
    port = 5000
    server_thread = threading.Thread(target=lambda: webbrowser.open_new_tab(f"http://127.0.0.1:{port}"), daemon=True)
    server_thread.start()
    app.run(host="0.0.0.0", port=port, debug=False)
