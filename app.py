import streamlit as st
import ssl
import urllib.request
import cv2
import numpy as np
import av

# --- MediaPipeのモデルダウンロード失敗対策 ---
ssl._create_default_https_context = ssl._create_unverified_context

# --- MediaPipeのインポート (標準的な形式に戻す) ---
import mediapipe as mp
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

from streamlit_webrtc import webrtc_streamer, RTCConfiguration

# --- 角度計算関数 ---
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0:
        angle = 360-angle
    return angle

# --- 状態管理クラス ---
class SquatTracker:
    def __init__(self):
        self.counter = 0
        self.stage = None
        self.feedback = "Ready"
        self.warning = ""

# Session Stateを使って状態を保持
if 'tracker' not in st.session_state:
    st.session_state['tracker'] = SquatTracker()

# --- MediaPipeの初期化 ---
pose = mp_pose.Pose(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# --- 映像処理用コールバック ---
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    img = cv2.flip(img, 1)
    
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_img)
    
    tracker = st.session_state['tracker']

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        
        # 左半身の座標
        shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
        ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

        knee_angle = calculate_angle(hip, knee, ankle)
        hip_angle = calculate_angle(shoulder, hip, knee)

        if knee_angle < 100:
            tracker.stage = "down"
            tracker.feedback = "Go Up!"
        if knee_angle > 160 and tracker.stage == 'down':
            tracker.stage = "up"
            tracker.counter += 1
            tracker.feedback = "Good!"
        
        if tracker.stage == "down" and hip_angle < 80:
            tracker.warning = "Keep Back Straight!"
        else:
            tracker.warning = ""

        mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        cv2.rectangle(img, (0,0), (280, 120), (245,117,16), -1)
        cv2.putText(img, f'COUNT: {tracker.counter}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, f'STAGE: {tracker.stage}', (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, f' {tracker.feedback}', (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        if tracker.warning:
            cv2.rectangle(img, (0, 400), (640, 480), (0,0,255), -1)
            cv2.putText(img, tracker.warning, (120, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3, cv2.LINE_AA)

    return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- Streamlit UI ---
st.title("AI Squat Trainer")

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

webrtc_streamer(
    key="squat-counter",
    video_frame_callback=video_frame_callback,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)