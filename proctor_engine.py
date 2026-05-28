import base64
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime

import cv2
import keyboard
import numpy as np
import winsound
from ultralytics import YOLO


@dataclass
class MonitoringState:
    cheating_score: int = 0
    face_count: int = 0
    typing_speed: int = 0
    typing_status: str = "Normal Typing"
    typing_probability: int = 5
    alert_status: str = "SAFE"
    gaze_status: str = "Looking Center"
    object_status: str = "No suspicious objects"
    event_history: list = field(default_factory=list)


class ProctorEngine:
    def __init__(self):
        self.evidence_base = "evidence"
        self.reports_dir = "reports"
        self.model_path = "yolov8n.pt"
        self.thread_lock = threading.Lock()
        self.state = MonitoringState()
        self.known_categories = ["multiple_faces", "phone_detected", "looking_away", "no_face"]
        self.key_timestamps = deque()
        self.alert_cooldowns = {}
        self.no_face_start = None
        self.away_start = None
        self.current_alert = "SAFE"
        self.debug_frame = None

        self._ensure_directories()
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.yolo_model = self._load_yolo_model()
        self._start_keyboard_listener()

    def _ensure_directories(self):
        os.makedirs(self.evidence_base, exist_ok=True)
        for category in self.known_categories:
            os.makedirs(os.path.join(self.evidence_base, category), exist_ok=True)
        os.makedirs(self.reports_dir, exist_ok=True)

    def _load_yolo_model(self):
        try:
            return YOLO(self.model_path)
        except Exception as exc:
            print(f"YOLO model load failed: {exc}")
            return None

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

    def save_evidence(self, frame, category):
        category_dir = os.path.join(self.evidence_base, category)
        os.makedirs(category_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        path = os.path.join(category_dir, f"{timestamp}.jpg")
        cv2.imwrite(path, frame)
        return path

    def _trigger_alarm(self, event_type, frequency=1000, duration=250):
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
        self.state.cheating_score = min(100, self.state.cheating_score + points)
        record = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "event": event_text,
            "screenshot_path": screenshot_path,
            "cheating_score": self.state.cheating_score,
        }
        self.state.event_history.append(record)
        self.current_alert = event_text

    def _update_typing_stats(self):
        self.key_timestamps = deque([ts for ts in self.key_timestamps if time.time() - ts <= 5])
        self.state.typing_speed = len(self.key_timestamps)
        if self.state.typing_speed < 10:
            self.state.typing_status = "Normal Typing"
            self.state.typing_probability = 5
        elif self.state.typing_speed <= 25:
            self.state.typing_status = "Fast Typing"
            self.state.typing_probability = 20
        else:
            self.state.typing_status = "Suspicious Typing"
            self.state.typing_probability = 40

    def process_frame(self, frame):
        with self.thread_lock:
            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            self.state.face_count = len(faces)
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

                if gaze_status != "Looking Center":
                    if self.away_start is None:
                        self.away_start = time.time()
                    elif time.time() - self.away_start >= 3:
                        screenshot_path = self.save_evidence(frame, "looking_away")
                        self._record_event(f"Looking away detected: {gaze_status}", "looking_away", 20, screenshot_path)
                        self._trigger_alarm("looking_away", frequency=700, duration=260)
                        self.current_alert = f"LOOKING AWAY: {gaze_status}"
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

            suspicious_objects = []
            if self.yolo_model is not None:
                results = self.yolo_model(frame, verbose=False)
                for result in results:
                    boxes = result.boxes.cpu().numpy()
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].astype(int)
                        class_id = int(box.cls[0])
                        class_name = self.yolo_model.names[class_id].lower()
                        confidence = float(box.conf[0])
                        if any(keyword in class_name for keyword in ["phone", "book", "laptop", "device"]):
                            suspicious_objects.append((class_name, confidence, (x1, y1, x2, y2)))
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (170, 0, 255), 2)
                            label = f"{class_name} {confidence:.2f}"
                            cv2.putText(frame, label, (x1, max(y1 - 10, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

            if suspicious_objects:
                class_name, confidence, coords = suspicious_objects[0]
                self.state.object_status = f"Detected: {class_name}"
                screenshot_path = self.save_evidence(frame, "phone_detected")
                if "phone" in class_name:
                    self._record_event(f"Phone detected: {class_name}", "phone_detected", 40, screenshot_path)
                    self._trigger_alarm("phone_detected", frequency=800, duration=280)
                    self.current_alert = f"Phone detected: {class_name}"
                elif "laptop" in class_name or "book" in class_name:
                    self._record_event(f"Unauthorized material detected: {class_name}", "phone_detected", 20, screenshot_path)
                    self._trigger_alarm("phone_detected", frequency=760, duration=260)
                    self.current_alert = f"Unauthorized material: {class_name}"
                else:
                    self._record_event(f"Additional device detected: {class_name}", "phone_detected", 15, screenshot_path)
                    self._trigger_alarm("phone_detected", frequency=720, duration=240)
                    self.current_alert = f"Device detected: {class_name}"

            self._update_typing_stats()
            if self.state.typing_status == "Suspicious Typing":
                if time.time() >= self.alert_cooldowns.get("suspicious_typing", 0):
                    self.alert_cooldowns["suspicious_typing"] = time.time() + 10
                    self._record_event("Suspicious typing pattern detected", "phone_detected", 40)
                    self._trigger_alarm("suspicious_typing", frequency=650, duration=250)
                    self.current_alert = "Suspicious typing pattern detected"

            self.state.gaze_status = gaze_status
            self.state.alert_status = self._risk_level(self.state.cheating_score)
            self.debug_frame = frame
            return frame, self.get_status()

    def _risk_level(self, score):
        if score <= 20:
            return "SAFE"
        if score <= 50:
            return "SUSPICIOUS"
        return "HIGH RISK"

    def get_status(self):
        status = {
            "cheating_score": min(100, self.state.cheating_score),
            "face_count": self.state.face_count,
            "typing_speed": self.state.typing_speed,
            "typing_status": self.state.typing_status,
            "typing_probability": self.state.typing_probability,
            "alert_status": self.current_alert,
            "risk_level": self._risk_level(self.state.cheating_score),
            "gaze_status": self.state.gaze_status,
            "object_status": self.state.object_status,
            "event_history": self.state.event_history[-20:],
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        return status

    def reset(self):
        with self.thread_lock:
            self.state.cheating_score = 0
            self.state.event_history.clear()
            self.current_alert = "SAFE"
            self.state.object_status = "No suspicious objects"
            self.state.gaze_status = "Looking Center"
            self.alert_cooldowns.clear()
            self.no_face_start = None
            self.away_start = None
