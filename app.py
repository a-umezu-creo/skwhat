import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import av
import ssl
from streamlit_webrtc import webrtc_streamer, RTCConfiguration

# --- 1. ページ設定と画面修正CSS ---
st.set_page_config(page_title="AI Squat Trainer", layout="wide")

st.markdown(
    """
    <style>
    /* 全体の余白を削る */
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    /* ビデオ表示エリアを横幅いっぱいに広げ、高さを出す */
    div[data-testid="stWebStreamer"] {
        width: 100% !important;
    }
    video {
        width: 100% !important;
        height: auto !important;
        aspect-ratio: 4 / 3 !important; /* 強制的に比率を固定 */
        object-fit: cover !important;  /* 細くならずに枠を埋める */
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# SSL対策
ssl._create_default_https_context = ssl._create_unverified_context

# MediaPipe初期化
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# --- 2. 状態管理 ---
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
    img = cv2.flip(img, 1) # 鏡面
    
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_img)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        
        # 判定用座標
        shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
        ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

        knee_angle = calculate_angle(hip, knee, ankle)
        hip_angle = calculate_angle(shoulder, hip, knee)

        # カウント
        if knee_angle < 110:
            tracker["stage"] = "down"
            tracker["feedback"] = "UP!"
        if knee_angle > 160 and tracker["stage"] == "down":
            tracker["stage"] = "up"
            tracker["count"] += 1
            tracker["feedback"] = "GOOD!"
        
        tracker["warning"] = "Back Straight!" if tracker["stage"] == "down" and hip_angle < 70 else ""

        # 骨格描画
        mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        # UI描画 (文字位置とサイズを調整)
        cv2.rectangle(img, (0, 0), (280, 140), (245, 117, 16), -1)
        cv2.putText(img, f"COUNT: {tracker['count']}", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, f"STAGE: {tracker['stage']}", (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, tracker["feedback"], (10, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

        if tracker["warning"]:
            cv2.rectangle(img, (0, img.shape[0]-50), (img.shape[1], img.shape[0]), (0, 0, 255), -1)
            cv2.putText(img, tracker["warning"], (50, img.shape[0]-15), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- 4. メインUI ---
st.title("🏋️ AI スクワットトレーナー")

RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

# 映像表示コンテナ
webrtc_streamer(
    key="squat-counter",
    video_frame_callback=video_frame_callback,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={
        "video": {
            "width": {"ideal": 640},
            "height": {"ideal": 480},
            "facingMode": "user",
        },
        "audio": False
    },
    video_html_attrs={
        "style": {
            "width": "100%", 
            "aspect-ratio": "4 / 3",
            "object-fit": "cover"
        },
        "controls": False,
        "autoPlay": True,
        "playsInline": True,
    },
    async_processing=True,
)

st.write(f"### カウント: {tracker['count']}")

if st.button("リセット"):
    tracker["count"] = 0
    st.rerun()

st.info("カメラを起動して、全身が映るまで離れてください。")