import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import av
import ssl
from streamlit_webrtc import webrtc_streamer, RTCConfiguration

# --- 1. SSL証明書エラー対策 (MediaPipeモデルダウンロード用) ---
ssl._create_default_https_context = ssl._create_unverified_context

# --- 2. MediaPipeの初期設定 ---
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# model_complexity=1 は同梱されていることが多いため、Permission Errorを回避しやすい
pose = mp_pose.Pose(
    model_complexity=1, 
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# --- 3. 状態管理 (カウンターなど) ---
# webrtcのコールバック内で保持するため、classを使用します
class SquatState:
    def __init__(self):
        self.counter = 0
        self.stage = "up"  # "up" or "down"
        self.warning = ""
        self.feedback = "Ready"

# インスタンス生成
if "state" not in st.session_state:
    st.session_state["state"] = SquatState()

# --- 4. 角度計算ロジック ---
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
    return angle

# --- 5. 映像処理用コールバック関数 ---
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    img = cv2.flip(img, 1)  # 鏡面反転
    
    # 状態の取得
    state = st.session_state["state"]
    
    # RGB変換して解析
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_img)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        
        # 必要な部位の座標取得 (左側を基準)
        shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
        ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

        # 角度の計算
        knee_angle = calculate_angle(hip, knee, ankle)
        hip_angle = calculate_angle(shoulder, hip, knee)

        # スクワット判定ロジック
        if knee_angle < 110:  # しゃがんだ
            state.stage = "down"
            state.feedback = "Go Up!"
        
        if knee_angle > 160 and state.stage == "down":  # 立ち上がった
            state.stage = "up"
            state.counter += 1
            state.feedback = "Good Job!"
        
        # 姿勢警告 (背中が曲がりすぎていないか)
        if state.stage == "down" and hip_angle < 70:
            state.warning = "Keep Back Straight!"
        else:
            state.warning = ""

        # 骨格の描画
        mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # UI表示の描画 (左上にカウンター)
        cv2.rectangle(img, (0, 0), (280, 150), (245, 117, 16), -1)
        cv2.putText(img, f"COUNT: {state.counter}", (10, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, f"STAGE: {state.stage}", (10, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, f" {state.feedback}", (10, 130), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

        # 警告表示 (画面下部に赤色で表示)
        if state.warning:
            cv2.rectangle(img, (0, 400), (640, 480), (0, 0, 255), -1)
            cv2.putText(img, state.warning, (100, 450), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- 6. Streamlit メイン画面 ---
st.set_page_config(page_title="AI Squat Trainer", layout="wide")
st.title("AI Squat Counter & Posture Check")

st.markdown("""
### 使い方
1. カメラを許可して開始してください。
2. 全身が映る位置にスマホを立てかけてください。
3. 膝をしっかり曲げるとカウントされます。
""")

# WebRTC 設定
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

webrtc_streamer(
    key="squat-workout",
    mode=av.VideoFrame,
    video_frame_callback=video_frame_callback,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

if st.button("カウントをリセット"):
    st.session_state["state"].counter = 0
    st.experimental_rerun()

st.info("背中が丸まりすぎると画面下に赤い警告が出ます。")