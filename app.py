import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import av
import ssl
from streamlit_webrtc import webrtc_streamer, RTCConfiguration

# --- 1. ページ設定と画面の強制拡大CSS ---
st.set_page_config(page_title="AI Squat Trainer", layout="wide")

# カメラ映像を縦長・細い線にさせないための強力なCSS
st.markdown(
    """
    <style>
    /* コンテナの幅を100%に */
    .main .block-container {
        max-width: 100%;
        padding: 1rem;
    }
    /* WebRTCのビデオ表示エリアの高さを確保 */
    div[data-testid="stWebSrtreamer"] iframe {
        min-height: 500px !important;
    }
    video {
        width: 100% !important;
        height: auto !important;
        min-height: 400px !important;
        object-fit: contain !important;
        background-color: black;
    }
    </style>
    """,
    unsafe_allow_html=True, # ここを修正しました
)

# SSL対策
ssl._create_default_https_context = ssl._create_unverified_context

# MediaPipe初期化 (Permission Error対策で1を使用)
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# --- 2. 状態管理 ---
# コールバック外でも値を保持しやすくするため、辞書形式で定義
if "tracker" not in st.session_state:
    st.session_state["tracker"] = {"count": 0, "stage": "up", "feedback": "Ready", "warning": ""}

tracker = st.session_state["tracker"]

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return angle if angle <= 180.0 else 360 - angle

# --- 3. 映像処理コールバック ---
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    img = cv2.flip(img, 1)
    
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_img)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        
        # 左側の座標取得
        shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
        ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

        knee_angle = calculate_angle(hip, knee, ankle)
        hip_angle = calculate_angle(shoulder, hip, knee)

        # カウント判定
        if knee_angle < 110:
            tracker["stage"] = "down"
            tracker["feedback"] = "Go Up!"
        if knee_angle > 160 and tracker["stage"] == "down":
            tracker["stage"] = "up"
            tracker["count"] += 1
            tracker["feedback"] = "Nice!"
        
        tracker["warning"] = "Back Straight!" if tracker["stage"] == "down" and hip_angle < 70 else ""

        # 描画
        mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        # UIオーバーレイ
        cv2.rectangle(img, (0, 0), (300, 140), (245, 117, 16), -1)
        cv2.putText(img, f"COUNT: {tracker['count']}", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, f"STAGE: {tracker['stage']}", (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, tracker["feedback"], (10, 125), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        if tracker["warning"]:
            cv2.rectangle(img, (0, 400), (640, 480), (0, 0, 255), -1)
            cv2.putText(img, tracker["warning"], (100, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- 4. メインUI ---
st.title("🏋️ AI スクワットトレーナー")

RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

# 映像が崩れるのを防ぐためのコンテナ
webrtc_streamer(
    key="squat-counter",
    video_frame_callback=video_frame_callback,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={
        "video": {
            "width": {"min": 640, "ideal": 1280},
            "height": {"min": 480, "ideal": 720},
            "facingMode": "user",
        },
        "audio": False
    },
    video_html_attrs={
        "style": {
            "width": "100%", 
            "height": "auto", 
            "min-height": "400px",
            "object-fit": "contain"
        },
        "controls": False,
        "autoPlay": True,
        "playsInline": True,
    },
    async_processing=True,
)

st.write(f"### 現在のカウント: {tracker['count']}")

if st.button("リセット"):
    tracker["count"] = 0
    tracker["stage"] = "up"
    st.rerun()

st.info("カメラを起動したら、全身が映るまで2〜3メートル離れてください。")