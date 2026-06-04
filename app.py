import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import av
import ssl
from streamlit_webrtc import webrtc_streamer, RTCConfiguration

# --- 1. 極限まで余白を削り、映像を巨大化するCSS ---
st.set_page_config(page_title="AI Trainer", layout="centered")

st.markdown(
    """
    <style>
    /* ヘッダー、メニュー、フッターをすべて非表示 */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* 画面の余白をゼロにする */
    .main .block-container {
        max-width: 100% !important;
        padding: 0px !important;
        margin: 0px !important;
    }

    /* カメラ映像を縦画面で最大化 */
    video {
        width: 100vw !important;
        height: 75vh !important; /* 画面の75%をカメラに */
        object-fit: cover !important;
        background-color: black;
    }
    
    /* 回数表示を巨大にする */
    .count-text {
        font-size: 80px !important;
        font-weight: bold;
        color: #FF4B4B;
        text-align: center;
        margin-top: -20px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# SSL対策
ssl._create_default_https_context = ssl._create_unverified_context

# MediaPipe
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

if "tracker" not in st.session_state:
    st.session_state["tracker"] = {"count": 0, "stage": "up", "feedback": "", "warning": ""}

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
        hip_angle = calculate_angle(shoulder, hip, knee)

        # カウントロジック
        if knee_angle < 110:
            tracker["stage"] = "down"
            tracker["feedback"] = "UP!"
        if knee_angle > 160 and tracker["stage"] == "down":
            tracker["stage"] = "up"
            tracker["count"] += 1
            tracker["feedback"] = "OK!"
        
        tracker["warning"] = "STRAIGHT!" if tracker["stage"] == "down" and hip_angle < 70 else ""
        
        # 骨格描画（少し太くする）
        mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245,117,66), thickness=4, circle_radius=4))

        # 【超巨大文字】画面中央付近にカウントを表示
        cv2.putText(img, str(tracker["count"]), (50, 150), 
                    cv2.FONT_HERSHEY_DUPLEX, 5.0, (255, 255, 255), 10, cv2.LINE_AA)
        
        # フィードバック表示
        if tracker["feedback"]:
            cv2.putText(img, tracker["feedback"], (50, 250), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 5, cv2.LINE_AA)

        # 警告表示（画面が赤く光るように）
        if tracker["warning"]:
            cv2.rectangle(img, (0, 0), (img.shape[1], img.shape[0]), (0, 0, 255), 10)
            cv2.putText(img, "BACK!", (50, 400), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 0, 255), 10, cv2.LINE_AA)

    return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- メイン画面 ---
webrtc_streamer(
    key="squat-pro",
    video_frame_callback=video_frame_callback,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": {"facingMode": "user"}, "audio": False},
    video_html_attrs={
        "style": {"width": "100vw", "height": "75vh", "object-fit": "cover"},
        "autoPlay": True, "playsInline": True
    },
    async_processing=True,
)

# 下部に巨大な回数を表示
st.markdown(f'<div class="count-text">{tracker["count"]}</div>', unsafe_allow_html=True)

if st.button("RESET", use_container_width=True):
    tracker["count"] = 0
    st.rerun()