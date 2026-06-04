import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import av
import ssl
from streamlit_webrtc import webrtc_streamer, RTCConfiguration

# --- 1. ページ設定とスマホ縦画面専用CSS ---
st.set_page_config(page_title="AI Squat Trainer", layout="centered")

st.markdown(
    """
    <style>
    /* メインコンテンツの幅を調整 */
    .main .block-container {
        max-width: 100%;
        padding: 0.5rem;
    }
    /* カメラ映像をスマホの縦画面いっぱいに広げる */
    div[data-testid="stWebStreamer"] {
        width: 100% !important;
        margin-bottom: 10px;
    }
    video {
        width: 100% !important;
        height: auto !important;
        /* 縦長比率に設定 (9:16) */
        aspect-ratio: 9 / 14 !important; 
        object-fit: cover !important; 
        border-radius: 15px;
        background-color: #000;
    }
    /* ボタンなどのUIを大きくする */
    .stButton button {
        width: 100%;
        height: 3rem;
        font-size: 1.2rem !important;
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

# 状態管理
if "tracker" not in st.session_state:
    st.session_state["tracker"] = {"count": 0, "stage": "up", "feedback": "Ready", "warning": ""}

tracker = st.session_state["tracker"]

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return angle if angle <= 180.0 else 360 - angle

# --- 映像処理コールバック ---
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    img = cv2.flip(img, 1) 
    
    # スマホ縦画面では、映像の「中心」を使って処理
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
            tracker["stage"] = "down"
            tracker["feedback"] = "UP!"
        if knee_angle > 160 and tracker["stage"] == "down":
            tracker["stage"] = "up"
            tracker["count"] += 1
            tracker["feedback"] = "GOOD!"
        
        tracker["warning"] = "Back Straight!" if tracker["stage"] == "down" and hip_angle < 70 else ""

        mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        # 描画位置を少し内側に寄せる（object-fit:coverでの見切れ防止）
        cv2.rectangle(img, (20, 20), (300, 160), (245, 117, 16), -1)
        cv2.putText(img, f"COUNT: {tracker['count']}", (40, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(img, tracker["feedback"], (40, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        if tracker["warning"]:
            cv2.rectangle(img, (0, img.shape[0]-80), (img.shape[1], img.shape[0]), (0, 0, 255), -1)
            cv2.putText(img, tracker["warning"], (80, img.shape[0]-30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3, cv2.LINE_AA)

    return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- メインUI ---
st.title("🏋️ AI トレーナー")

RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

webrtc_streamer(
    key="squat-counter",
    video_frame_callback=video_frame_callback,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={
        "video": {
            "width": {"ideal": 480},
            "height": {"ideal": 640},
            "facingMode": "user",
        },
        "audio": False
    },
    video_html_attrs={
        "style": {
            "width": "100%", 
            "aspect-ratio": "9 / 14", # 縦長比率
            "object-fit": "cover"
        },
        "controls": False,
        "autoPlay": True,
        "playsInline": True,
    },
    async_processing=True,
)

# カウント表示を大きく
st.metric(label="現在の回数", value=f"{tracker['count']} 回")

if st.button("リセット"):
    tracker["count"] = 0
    st.rerun()

st.info("スマホを立てかけて、全身が映るまで離れてください。")