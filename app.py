import cv2
import numpy as np
from flask import Flask, Response
import mediapipe as mp
import tflite_runtime.interpreter as tflite
import pickle
from collections import deque
from statistics import mode
import threading
import time

# === AUPPBot ===
from auppbot_2 import AUPPBot

# ============================================================
#                 ROBOT SETUP
# ============================================================

ROBOT_PORT = "/dev/ttyUSB0"
ROBOT_BAUD = 115200
bot = AUPPBot(ROBOT_PORT, ROBOT_BAUD, auto_safe=True)

FWD_SPEED = 10
TURN_SPEED = 35
STOP_TIMEOUT_FRAMES = 9

# ============================================================
#                 TFLITE MODEL
# ============================================================

TFLITE_PATH = "gesture_mlp_model_v3.tflite"
interpreter = tflite.Interpreter(model_path=TFLITE_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def tflite_predict(features):
    interpreter.set_tensor(input_details[0]['index'], features.astype(np.float32))
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]['index'])

with open("gesture_label_encoder_v3.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# ============================================================
#                 MEDIAPIPE HANDS
# ============================================================

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.5,
)
mp_draw = mp.solutions.drawing_utils

# ============================================================
#                 CAMERA THREAD
# ============================================================

class Camera:
    def __init__(self, device=0, width=640, height=480):
        self.cap = cv2.VideoCapture(device)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.lock = threading.Lock()
        self.frame = None
        self.running = True
        threading.Thread(target=self.update, daemon=True).start()

    def update(self):
        while self.running:
            ok, f = self.cap.read()
            if ok:
                with self.lock:
                    self.frame = f
            else:
                time.sleep(0.01)

    def read(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def release(self):
        self.running = False
        try:
            self.cap.release()
        except:
            pass

cam = Camera(device=0, width=640, height=480)

# ============================================================
#                 SMOOTHING + DEBOUNCE
# ============================================================

N = 5
K = 5
THRESH = 0.60

prediction_buffer = deque(maxlen=N)
stable_gesture = None
stable_count = 0
no_hand_counter = 0

# ============================================================
#                 FEATURE EXTRACTION
# ============================================================

def normalize_landmarks(landmarks):
    lm = np.array(landmarks)
    wrist = lm[0]
    lm = lm - wrist
    max_dist = np.max(np.sqrt(np.sum(lm**2, axis=1)))
    if max_dist > 0:
        lm = lm / max_dist
    return lm.flatten()

def compute_finger_spreads(lm):
    return np.array([
        np.linalg.norm(lm[5] - lm[9]),
        np.linalg.norm(lm[9] - lm[13]),
        np.linalg.norm(lm[13] - lm[17]),
    ])

def extract_features(results):
    if not results.multi_hand_landmarks:
        return None
    hand = results.multi_hand_landmarks[0]
    raw = np.array([[lm.x, lm.y] for lm in hand.landmark], dtype="float32")
    norm = normalize_landmarks(raw)
    spreads = compute_finger_spreads(norm.reshape(21, 2))
    return np.concatenate([norm, spreads]).reshape(1, -1)

# ============================================================
#                 ROBOT ACTIONS
# ============================================================

def robot_execute(gesture):
    if gesture == "Go":
        print("ACTION: FORWARD")
        bot.motor1.forward(FWD_SPEED)
        bot.motor2.forward(FWD_SPEED)
        bot.motor3.forward(FWD_SPEED)
        bot.motor4.forward(FWD_SPEED)
    elif gesture == "Left":
        print("ACTION: TURN LEFT")
        bot.motor1.backward(TURN_SPEED)
        bot.motor4.forward(TURN_SPEED)
    elif gesture == "Right":
        print("ACTION: TURN RIGHT")
        bot.motor3.backward(TURN_SPEED)
        bot.motor2.forward(TURN_SPEED)
    elif gesture == "Stop":
        print("ACTION: STOP")
        bot.motor1.stop(); bot.motor2.stop()
        bot.motor3.stop(); bot.motor4.stop()
    else:
        print("ACTION: FAILSAFE STOP")
        bot.motor1.stop(); bot.motor2.stop()
        bot.motor3.stop(); bot.motor4.stop()

# ============================================================
#                 VIDEO GENERATOR
# ============================================================

def gen_frames():
    global prediction_buffer, stable_gesture, stable_count, no_hand_counter

    while True:
        frame = cam.read()
        if frame is None:
            time.sleep(0.01)
            continue

        # rotate or flip if needed
        frame = cv2.rotate(frame, cv2.ROTATE_180)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        final_text = "No hand"
        if not results.multi_hand_landmarks:
            no_hand_counter += 1
            stable_gesture = None       # reset stable gesture
            stable_count = 0            # reset debounce count
            if no_hand_counter >= STOP_TIMEOUT_FRAMES:
                robot_execute("Stop")
        else:
            no_hand_counter = 0
            feats = extract_features(results)
            if feats is not None:
                pred = tflite_predict(feats)
                class_id = np.argmax(pred)
                confidence = np.max(pred)
                gesture = label_encoder.inverse_transform([class_id])[0]
                if confidence >= THRESH:
                    prediction_buffer.append(gesture)
                else:
                    prediction_buffer.append("unknown")

                # Majority vote + debounce
                if len(prediction_buffer) == N:
                    majority_vote = mode(prediction_buffer)
                    if majority_vote == stable_gesture:
                        stable_count += 1
                    else:
                        stable_gesture = majority_vote
                        stable_count = 1

                    if stable_count >= K:
                        robot_execute(stable_gesture)
                        final_text = f"Gesture: {stable_gesture}"
                    else:
                        final_text = f"Smoothing: {majority_vote} ({stable_count}/{K})"
                else:
                    final_text = "Calculating..."

        cv2.putText(frame, final_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        _, buffer = cv2.imencode(".jpg", frame)
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")

# ============================================================
#                 FLASK ROUTES
# ============================================================

app = Flask(__name__)

@app.route("/")
def home():
    return """
    <h1>Robot Gesture Recognition (TFLite)</h1>
    <p>Commands: Go, Stop, Left, Right</p>
    <p><a href="/video_feed">View Camera Feed</a></p>
    """

@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

# ============================================================

if __name__ == "__main__":
    try:
        app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
    finally:
        cam.release()
        bot.stop_all()
        bot.close()
