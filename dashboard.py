import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime

import tkinter.ttk as ttk

import cv2
import keyboard
import numpy as np
import winsound
from PIL import Image, ImageTk
from report_generator import generate_reports
from ultralytics import YOLO


@dataclass
class DashboardState:
    cheating_score: int = 0
    face_count: int = 0
    typing_speed: int = 0
    typing_status: str = "Normal Typing"
    typing_probability: int = 5
    alert_status: str = "SAFE"
    gaze_status: str = "Looking Center"
    object_status: str = "No suspicious objects"
    event_history: list = field(default_factory=list)


class ExamShieldDashboard:
    def __init__(self):
        # Initialize core application status and monitor paths.
        self.evidence_base = "evidence"
        self.reports_dir = "reports"
        self.model_path = "yolov8n.pt"
        self.frame_count = 0
        self.monitoring_running = True
        self.running = True
        self.key_timestamps = deque()
        self.alert_cooldowns = {}
        self.current_alert = "SAFE"
        self.no_face_start = None
        self.away_start = None
        self.last_object_seen = None
        self.last_frame_time = time.time()
        self.state = DashboardState()

        # Create processing directories.
        self._ensure_directories()

        # Initialize GUI.
        self.root = None
        self.video_label = None
        self.face_count_label = None
        self.typing_speed_label = None
        self.typing_status_label = None
        self.typing_risk_label = None
        self.cheating_score_label = None
        self.risk_level_label = None
        self.alert_status_label = None
        self.gaze_status_label = None
        self.object_status_label = None
        self.report_status_label = None
        self.event_tree = None
        self.video_status_label = None
        self.photo = None

        # Load computer vision models.
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.yolo_model = None
        self._load_yolo_model()

        # Webcam capture.
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Webcam could not be opened. The application will show a camera error.")

        # Start the keyboard listener for typing analysis.
        self._start_keyboard_listener()

        # Build GUI.
        self._build_gui()

        # Start real-time frame updates.
        self.root.after(10, self.update_frame)

    def _ensure_directories(self):
        os.makedirs(self.evidence_base, exist_ok=True)
        os.makedirs(os.path.join(self.evidence_base, "multiple_faces"), exist_ok=True)
        os.makedirs(os.path.join(self.evidence_base, "phone_detected"), exist_ok=True)
        os.makedirs(os.path.join(self.evidence_base, "looking_away"), exist_ok=True)
        os.makedirs(os.path.join(self.evidence_base, "no_face"), exist_ok=True)
        os.makedirs(self.reports_dir, exist_ok=True)

    def _load_yolo_model(self):
        try:
            self.yolo_model = YOLO(self.model_path)
        except Exception as exc:
            print(f"YOLO model load failed: {exc}")
            self.yolo_model = None

    def _build_gui(self):
        self.root = tk = __import__("tkinter").Tk()
        tk.title("ExamShield AI – Intelligent Smart Proctoring")
        tk.geometry("1500x920")
        tk.configure(bg="#030712")

        # Top title bar.
        header = __import__("tkinter").Frame(tk, bg="#020817", bd=0)
        header.place(x=0, y=0, width=1500, height=80)

        title = __import__("tkinter").Label(
            header,
            text="ExamShield AI – Intelligent Smart Proctoring System",
            fg="#f8fafc",
            bg="#020817",
            font=("Segoe UI", 24, "bold"),
        )
        title.place(x=24, y=20)

        subtitle = __import__("tkinter").Label(
            header,
            text="Cybersecurity-style real-time exam integrity monitoring",
            fg="#93c5fd",
            bg="#020817",
            font=("Segoe UI", 10, "bold"),
        )
        subtitle.place(x=24, y=52)

        # Control panel.
        control_panel = __import__("tkinter").Frame(tk, bg="#021126", bd=1, relief="flat")
        control_panel.place(x=24, y=92, width=1452, height=62)

        self._create_button(control_panel, "Start Monitoring", 20, 14, lambda: self.set_monitoring(True))
        self._create_button(control_panel, "Pause Monitoring", 170, 14, lambda: self.set_monitoring(False))
        self._create_button(control_panel, "Generate Reports", 350, 14, self.generate_reports)
        self._create_button(control_panel, "Reset Score", 540, 14, self.reset_score)

        self.video_status_label = __import__("tkinter").Label(
            control_panel,
            text="Live Monitoring: ACTIVE",
            fg="#22c55e",
            bg="#021126",
            font=("Segoe UI", 12, "bold"),
        )
        self.video_status_label.place(x=760, y=18)

        # Webcam display.
        video_frame = __import__("tkinter").Frame(tk, bg="#01040d", bd=2, relief="solid")
        video_frame.place(x=24, y=170, width=860, height=600)

        self.video_label = __import__("tkinter").Label(
            video_frame,
            text="Initializing webcam stream...",
            fg="#d1d5db",
            bg="#01040d",
            font=("Segoe UI", 14, "bold"),
            anchor="center",
        )
        self.video_label.place(x=0, y=0, width=860, height=600)

        # KPI cards panel.
        metrics_frame = __import__("tkinter").Frame(tk, bg="#01040d", bd=2, relief="solid")
        metrics_frame.place(x=900, y=170, width=576, height=600)

        # Create metric cards.
        self._create_metric_card(metrics_frame, 18, 20, "Face Count", "0", "#22c55e")
        self._create_metric_card(metrics_frame, 286, 20, "Typing Speed", "0 keys/5s", "#38bdf8")
        self._create_metric_card(metrics_frame, 18, 130, "Cheating Score", "0%", "#fbbf24")
        self._create_metric_card(metrics_frame, 286, 130, "Risk Level", "SAFE", "#22c55e")
        self._create_metric_card(metrics_frame, 18, 240, "Alert Status", "SAFE", "#22c55e")
        self._create_metric_card(metrics_frame, 286, 240, "Gaze Direction", "Looking Center", "#a78bfa")
        self._create_metric_card(metrics_frame, 18, 350, "Typing Status", "Normal Typing", "#60a5fa")
        self._create_metric_card(metrics_frame, 286, 350, "Detected Objects", "None", "#fb7185")

        # Historical table.
        table_frame = __import__("tkinter").Frame(tk, bg="#01040d", bd=2, relief="solid")
        table_frame.place(x=24, y=788, width=1452, height=90)

        table_title = __import__("tkinter").Label(
            table_frame,
            text="Live Event History",
            fg="#f8fafc",
            bg="#01040d",
            font=("Segoe UI", 14, "bold"),
        )
        table_title.place(x=16, y=10)

        columns = ("time", "event", "screenshot")
        self.event_tree = ttk.Treeview(table_frame, columns=columns, show="headings", height=4)
        self.event_tree.heading("time", text="Date & Time")
        self.event_tree.heading("event", text="Event")
        self.event_tree.heading("screenshot", text="Screenshot Path")
        self.event_tree.column("time", width=170, anchor="w")
        self.event_tree.column("event", width=520, anchor="w")
        self.event_tree.column("screenshot", width=680, anchor="w")
        self.event_tree.place(x=12, y=38, width=1428, height=44)

        scrollbar = __import__("tkinter").Scrollbar(table_frame, orient="vertical", command=self.event_tree.yview)
        scrollbar.place(x=1438, y=38, height=44)
        self.event_tree.configure(yscrollcommand=scrollbar.set)

        # Report status label.
        self.report_status_label = __import__("tkinter").Label(
            tk,
            text="Reports will be generated in the reports/ folder.",
            fg="#cbd5e1",
            bg="#030712",
            font=("Segoe UI", 10),
        )
        self.report_status_label.place(x=24, y=884)

        # Bind close event.
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def _create_button(self, parent, text, x, y, command):
        button = ttk.Button(parent, text=text, command=command)
        button.place(x=x, y=y, width=140, height=34)
        return button

    def _create_metric_card(self, parent, x, y, title, initial_value, accent):
        card = __import__("tkinter").Frame(parent, bg="#021126", bd=1, relief="solid")
        card.place(x=x, y=y, width=252, height=90)

        value_label = __import__("tkinter").Label(
            card,
            text=initial_value,
            fg="#f8fafc",
            bg="#021126",
            font=("Segoe UI", 20, "bold"),
        )
        value_label.place(x=16, y=32)

        title_label = __import__("tkinter").Label(
            card,
            text=title,
            fg=accent,
            bg="#021126",
            font=("Segoe UI", 10, "bold"),
        )
        title_label.place(x=16, y=10)

        attribute_name = title.lower().replace(" ", "_")
        setattr(self, f"{attribute_name}_label", value_label)

    def _start_keyboard_listener(self):
        def key_callback(event):
            try:
                key_name = event.name
                if key_name in ["shift", "ctrl", "alt", "alt gr", "tab", "caps lock", "esc"]:
                    return
                self.key_timestamps.append(time.time())
            except Exception:
                pass

        def listener_loop():
            try:
                keyboard.hook(key_callback)
                keyboard.wait()
            except Exception as exc:
                print(f"Keyboard monitoring failed: {exc}")

        thread = threading.Thread(target=listener_loop, daemon=True)
        thread.start()

    def set_monitoring(self, status):
        self.monitoring_running = status
        self.video_status_label.config(text="Live Monitoring: ACTIVE" if status else "Live Monitoring: PAUSED")
        self.video_status_label.config(fg="#22c55e" if status else "#fbbf24")

    def reset_score(self):
        self.state.cheating_score = 0
        self.state.event_history.clear()
        self.refresh_history()
        self.update_dashboard_metrics()

    def on_close(self):
        self.running = False
        if self.cap.isOpened():
            self.cap.release()
        self.root.destroy()

    def save_evidence(self, frame, category):
        category_dir = os.path.join(self.evidence_base, category)
        os.makedirs(category_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        path = os.path.join(category_dir, f"{timestamp}.jpg")
        cv2.imwrite(path, frame)
        return path

    def _trigger_alarm(self, event_type, frequency=1000, duration=220):
        now = time.time()
        cooldown = self.alert_cooldowns.get(event_type, 0)
        if now < cooldown:
            return
        self.alert_cooldowns[event_type] = now + 6
        try:
            winsound.Beep(frequency, duration)
        except Exception:
            pass

    def _record_event(self, event_text, category, points, screenshot_path="N/A"):
        if not self.monitoring_running:
            return

        cooldown_key = category
        cooldown_remaining = self.alert_cooldowns.get(cooldown_key, 0)
        if time.time() < cooldown_remaining:
            return

        self.alert_cooldowns[cooldown_key] = time.time() + 8
        self.state.cheating_score = min(100, self.state.cheating_score + points)

        record = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "event": event_text,
            "screenshot_path": screenshot_path,
            "cheating_score": self.state.cheating_score,
        }
        self.state.event_history.append(record)
        self.refresh_history()

        self.current_alert = event_text
        self.update_dashboard_metrics()

    def refresh_history(self):
        for item in self.event_tree.get_children():
            self.event_tree.delete(item)
        for index, record in enumerate(self.state.event_history[-20:]):
            self.event_tree.insert("", "end", values=(record["timestamp"], record["event"], record["screenshot_path"]))

    def update_dashboard_metrics(self):
        # Update KPI cards.
        self.face_count_label.config(text=str(self.state.face_count))
        self.typing_speed_label.config(text=f"{self.state.typing_speed} keys/5s")
        self.typing_status_label.config(text=self.state.typing_status)
        self.typing_risk_label.config(text=f"{self.state.typing_probability}%")

        score = self.state.cheating_score
        self.cheating_score_label.config(text=f"{score}%")
        if score <= 20:
            risk = "SAFE"
            color = "#22c55e"
        elif score <= 50:
            risk = "SUSPICIOUS"
            color = "#fbbf24"
        else:
            risk = "HIGH RISK"
            color = "#ef4444"

        self.risk_level_label.config(text=risk, fg=color)
        self.cheating_score_label.config(fg=color)

        self.alert_status_label.config(text=self.current_alert, fg="#f8fafc")
        self.gaze_status_label.config(text=self.state.gaze_status, fg="#a78bfa")
        self.object_status_label.config(text=self.state.object_status, fg="#fb7185")

    def generate_reports(self):
        try:
            excel_path, pdf_path = generate_reports(self.state.event_history, self.state.cheating_score, self.reports_dir)
            message = f"Reports generated: {os.path.basename(excel_path)} | {os.path.basename(pdf_path)}"
            self.report_status_label.config(text=message)
        except Exception as exc:
            self.report_status_label.config(text=f"Report generation failed: {exc}")

    def update_frame(self):
        if not self.running:
            return

        if not self.monitoring_running:
            self.video_label.config(text="Monitoring paused. Press Start Monitoring to resume.")
            self.root.after(100, self.update_frame)
            return

        self.frame_count += 1
        ret, frame = self.cap.read()
        if not ret:
            self.video_label.config(text="Camera unavailable. Please check webcam connection.")
            self.root.after(100, self.update_frame)
            return

        frame = cv2.flip(frame, 1)
        frame_display = cv2.resize(frame, (860, 600))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Face detection.
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        self.state.face_count = len(faces)

        # Draw boxes and determine gaze.
        gaze_status = "Looking Center"
        if len(faces) > 0:
            largest_face = max(faces, key=lambda box: box[2] * box[3])
            x, y, w, h = largest_face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
            face_center = x + (w / 2)
            if face_center < frame.shape[1] * 0.35:
                gaze_status = "Looking Left"
            elif face_center > frame.shape[1] * 0.65:
                gaze_status = "Looking Right"
            else:
                gaze_status = "Looking Center"

            # Looking away longer than 3 seconds.
            if gaze_status != "Looking Center":
                if self.away_start is None:
                    self.away_start = time.time()
                elif time.time() - self.away_start >= 3:
                    screenshot_path = self.save_evidence(frame, "looking_away")
                    self._record_event(f"Looking away detected: {gaze_status}", "looking_away", 20, screenshot_path)
                    self._trigger_alarm("looking_away", frequency=700, duration=260)
                    self.current_alert = f"LOOKING AWAY: {gaze_status}"
                    self.state.gaze_status = gaze_status
                    self.state.object_status = "Warning: looking away"
            else:
                self.away_start = None
        else:
            self.away_start = None
            if self.no_face_start is None:
                self.no_face_start = time.time()
            elif time.time() - self.no_face_start >= 3:
                screenshot_path = self.save_evidence(frame, "no_face")
                self._record_event("NO FACE DETECTED", "no_face", 30, screenshot_path)
                self._trigger_alarm("no_face", frequency=900, duration=320)
                self.current_alert = "NO FACE DETECTED"
                self.state.object_status = "No face detected"
        
        if len(faces) > 1:
            screenshot_path = self.save_evidence(frame, "multiple_faces")
            self._record_event("MULTIPLE FACES DETECTED!", "multiple_faces", 50, screenshot_path)
            self._trigger_alarm("multiple_faces", frequency=1200, duration=280)
            self.current_alert = "MULTIPLE FACES DETECTED!"
            self.state.object_status = "Multiple faces in frame"
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

        # YOLO object detection every 2 frames.
        if self.yolo_model is not None and self.frame_count % 2 == 0:
            results = self.yolo_model(frame, verbose=False)
            suspicious_objects = []
            for result in results:
                boxes = result.boxes.cpu().numpy()
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].astype(int)
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    class_name = self.yolo_model.names[class_id].lower()

                    if not any(keyword in class_name for keyword in ["phone", "book", "laptop", "device"]):
                        continue

                    suspicious_objects.append((class_name, confidence, (x1, y1, x2, y2)))
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (170, 0, 255), 2)
                    label = f"{class_name} {confidence:.2f}"
                    cv2.putText(frame, label, (x1, max(y1 - 10, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

            if suspicious_objects:
                suspicious_object, confidence, box_coords = suspicious_objects[0]
                self.state.object_status = f"Detected: {suspicious_object}"
                if "phone" in suspicious_object:
                    points = 40
                    alert_message = f"Phone detected: {suspicious_object}"
                elif "laptop" in suspicious_object or "book" in suspicious_object:
                    points = 20
                    alert_message = f"Unauthorized material detected: {suspicious_object}"
                else:
                    points = 15
                    alert_message = f"Additional device detected: {suspicious_object}"
                screenshot_path = self.save_evidence(frame, "phone_detected")
                self._record_event(alert_message, "phone_detected", points, screenshot_path)
                self._trigger_alarm("phone_detected", frequency=800, duration=280)
                self.current_alert = alert_message

        # Typing analysis every 5 seconds.
        self.key_timestamps = deque([timestamp for timestamp in self.key_timestamps if time.time() - timestamp <= 5])
        self.state.typing_speed = len(self.key_timestamps)
        if self.state.typing_speed < 10:
            typing_status = "Normal Typing"
            probability = 5
        elif self.state.typing_speed <= 25:
            typing_status = "Fast Typing"
            probability = 20
        else:
            typing_status = "Suspicious Typing"
            probability = 40

        self.state.typing_status = typing_status
        self.state.typing_probability = probability

        if typing_status == "Suspicious Typing":
            if time.time() >= self.alert_cooldowns.get("suspicious_typing", 0):
                self.alert_cooldowns["suspicious_typing"] = time.time() + 10
                self._record_event("Suspicious typing pattern detected", "suspicious_typing", 40)
                self._trigger_alarm("suspicious_typing", frequency=650, duration=250)
                self.current_alert = "Suspicious typing pattern detected"

        self.state.gaze_status = gaze_status
        if len(faces) == 0:
            self.no_face_start = self.no_face_start if self.no_face_start is not None else time.time()
        else:
            self.no_face_start = None

        # Update metrics and display.
        self.update_dashboard_metrics()

        # Update frame preview.
        preview_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(preview_frame)
        resized = image.resize((860, 600))
        self.photo = ImageTk.PhotoImage(resized)
        self.video_label.config(image=self.photo, text="")
        self.video_label.image = self.photo

        self.root.after(30, self.update_frame)


if __name__ == "__main__":
    app = ExamShieldDashboard()
    app.root.mainloop()
