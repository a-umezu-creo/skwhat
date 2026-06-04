import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import av
import ssl
from streamlit_webrtc import webrtc_streamer, RTCConfiguration

# --- 1. 映像を「画面の端から端まで」広げるための超強力なCSS ---
st.set_page_config(page_title="AI Trainer", layout="wide")

st.markdown(
    """
    <style>
    /* 1. Streamlit全体の余白を完全にゼロにする */
    .main .block-container {
        max-width: 100vw !important;
        padding: 0px !important;
        margin: 0px !important;
    }
    
    /* 2. webrtcのコンテナを画面いっぱいに広げる */
    div[data-testid="stWebStreamer"] {
        width: 100vw !important;
        display: block;
    }

    /* 3. ビデオタグそのものを画面の横幅100%・高さ固定に強制する */
    video {
        width: 100vw !important;
        height: 65vh !important; /* 画面の65%を映像が占領 */
        object-fit: cover !important; /* アスペクト比を保ちつつ枠を埋める */
        background-color: #000;
    }

    /* 4. 「Video Input」などの操作パネルを小さくし、下に置く */
    div[data-testid="stWebStreamer"] > div {
        padding: 10px !important;
    }
    
    /* 5. カウント表示用の巨大テキストスタイル */
    .big-count {
        position: fixed;
        bottom: 15%;
        right: 10%;
        font-size: 120px !important;
        font-weight: bold;
        color: #FF4B4B;
        text-shadow: 4px 4px 0px #000;
        z-index: 100;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# SSL対策
ssl._create_default_https_context = ssl._create_unverified_context

# MediaPipe設定
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

if "tracker" not in st.session_state:
    st.session_state["tracker"] = {"count": 0, "stage": "up", "feedback": ""}

tracker = st.session_state["tracker"]

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return angle if angle <= 180.0 else 360 - angle

def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    img = cv2.flip(img, 1)
    results = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
        ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

        knee_angle = calculate_angle(hip, knee, ankle)

        if knee_angle < 110:
            tracker["stage"] = "down"
        if knee_angle > 160 and tracker["stage"] == "down":
            tracker["stage"] = "up"
            tracker["count"] += 1
        
        mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(255,255,255), thickness=4, circle_radius=2))

        # 映像内の文字をさらに巨大化 (遠距離用)
        cv2.putText(img, str(tracker["count"]), (50, 150), 
                    cv2.FONT_HERSHEY_DUPLEX, 5.0, (255, 255, 255), 12, cv2.LINE_AA)

    return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- メイン表示 ---
# webrtcのコントロールをあえて下にするため、先にコンテナを表示
st.write("### AIトレーニング")

webrtc_streamer(
    key="squat-ultra",
    video_frame_callback=video_frame_callback,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={
        "video": {"width": {"ideal": 1280}, "height": {"ideal": 720}, "facingMode": "user"},
        "audio": False
    },
    video_html_attrs={
        "style": {"width": "100%", "height": "65vh", "object-fit": "cover"},
        "autoPlay": True, "playsInline": True
    },
    async_processing=True,
)

# 画面右下に固定された巨大な回数表示
st.markdown(f'<div class="big-count">{tracker["count"]}</div>', unsafe_allow_html=True)

if st.button("RESET"):
    tracker["count"] = 0
    st.rerun()