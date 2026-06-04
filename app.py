import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import av
import ssl
from streamlit_webrtc import webrtc_streamer, RTCConfiguration

# --- SSL証明書エラー対策 ---
ssl._create_default_https_context = ssl._create_unverified_context

# --- MediaPipeの初期設定 ---
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Permission Error回避のため complexity=1 を使用
pose = mp_pose.Pose(
    model_complexity=1, 
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# --- 状態管理用クラス (スレッドセーフにするため) ---
class SquatTracker:
    def __init__(self):
        self.counter = 0
        self.stage = "up"
        self.feedback = "Ready"
        self.warning = ""

# コールバック外でインスタンスを保持
if "tracker" not in st.session_state:
    st.session_state["tracker"] = SquatTracker()

tracker = st.session_state["tracker"]

# --- 角度計算関数 ---
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

# --- 映像処理用コールバック ---
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    img = cv2.flip(img, 1) # 鏡面
    
    # RGB変換
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_img)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        
        # 座標取得 (左側)
        shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
        ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

        # 角度計算
        knee_angle = calculate_angle(hip, knee, ankle)
        hip_angle = calculate_angle(shoulder, hip, knee)

        # カウントロジック (trackerインスタンスを直接更新)
        if knee_angle < 110:
            tracker.stage = "down"
            tracker.feedback = "Go Up!"
        
        if knee_angle > 160 and tracker.stage == "down":
            tracker.stage = "up"
            tracker.counter += 1
            tracker.feedback = "Good Job!"
        
        # 姿勢警告
        if tracker.stage == "down" and hip_angle < 70:
            tracker.warning = "Keep Back Straight!"
        else:
            tracker.warning = ""

        # 骨格描画
        mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # 画面表示
        cv2.rectangle(img, (0, 0), (280, 150), (245, 117, 16), -1)
        cv2.putText(img, f"COUNT: {tracker.counter}", (10, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, f"STAGE: {tracker.stage}", (10, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, f" {tracker.feedback}", (10, 130), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

        if tracker.warning:
            cv2.rectangle(img, (0, 400), (640, 480), (0, 0, 255), -1)
            cv2.putText(img, tracker.warning, (100, 450), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- Streamlit UI ---
st.title("AI Squat Trainer")

# WebRTC設定 (GoogleのSTUNサーバーを利用)
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# 【修正箇所】mode引数を削除し、より安定した呼び出しに変更
webrtc_streamer(
    key="squat-counter",
    video_frame_callback=video_frame_callback,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

# 画面上のステータス表示
st.write(f"### 現在のカウント: {tracker.counter}")
if st.button("リセット"):
    tracker.counter = 0
    tracker.stage = "up"
    st.rerun()

st.info("カメラを起動し、全身が映るように離れてください。")