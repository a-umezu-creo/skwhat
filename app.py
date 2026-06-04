import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import av
import ssl
from streamlit_webrtc import webrtc_streamer, RTCConfiguration

# --- 1. ページ設定 (画面を横いっぱいに使う) ---
st.set_page_config(page_title="AI Squat Trainer", layout="wide")

# SSL対策
ssl._create_default_https_context = ssl._create_unverified_context

# MediaPipe初期化
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# 状態管理
if "tracker" not in st.session_state:
    class SquatTracker:
        def __init__(self):
            self.counter = 0
            self.stage = "up"
            self.feedback = "Ready"
            self.warning = ""
    st.session_state["tracker"] = SquatTracker()

tracker = st.session_state["tracker"]

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return angle if angle <= 180.0 else 360 - angle

def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    img = cv2.flip(img, 1)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_img)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
        ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

        knee_angle = calculate_angle(hip, knee, ankle)
        hip_angle = calculate_angle(shoulder, hip, knee)

        if knee_angle < 110:
            tracker.stage = "down"
            tracker.feedback = "Go Up!"
        if knee_angle > 160 and tracker.stage == "down":
            tracker.stage = "up"
            tracker.counter += 1
            tracker.feedback = "Good Job!"
        
        tracker.warning = "Keep Back Straight!" if tracker.stage == "down" and hip_angle < 70 else ""

        mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        # 描画 (文字サイズを少し大きく調整)
        cv2.rectangle(img, (0, 0), (350, 160), (245, 117, 16), -1)
        cv2.putText(img, f"COUNT: {tracker.counter}", (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(img, f"STAGE: {tracker.stage}", (15, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(img, f" {tracker.feedback}", (15, 140), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        if tracker.warning:
            cv2.rectangle(img, (0, 400), (640, 480), (0, 0, 255), -1)
            cv2.putText(img, tracker.warning, (80, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3, cv2.LINE_AA)

    return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- UI 表示 ---
st.title("AI スクワットトレーナー")

# 横並びのレイアウトを作る（カメラを左、情報を右などにする場合）
# 今回はシンプルにカメラを大きく表示します
RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

webrtc_streamer(
    key="squat-counter",
    video_frame_callback=video_frame_callback,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={
        "video": {
            "width": {"ideal": 1280}, # 高解像度を希望
            "height": {"ideal": 720},
            "facingMode": "user"      # インカメラを優先
        },
        "audio": False
    },
    # 【追加】映像を画面の横幅いっぱいに広げる設定
    video_html_attrs={
        "style": {"width": "100%", "margin-left": "auto", "margin-right": "auto", "display": "block"},
        "controls": False,
        "autoPlay": True,
    },
    async_processing=True,
)

st.write(f"## 合計回数: {tracker.counter}")
if st.button("リセット"):
    tracker.counter = 0
    st.rerun()