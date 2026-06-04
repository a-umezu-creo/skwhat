import streamlit as st
import ssl
import urllib.request

# --- MediaPipeのモデルダウンロード失敗対策 ---
ssl._create_default_https_context = ssl._create_unverified_context
# ---------------------------------------

from streamlit_webrtc import webrtc_streamer, RTCConfiguration
import cv2
import mediapipe as mp
import numpy as np
import av


# --- MediaPipeの設定 ---
from mediapipe.python.solutions import pose as mp_pose
from mediapipe.python.solutions import drawing_utils as mp_drawing

# インポートチェック（デバッグ用）
if not hasattr(mp, 'solutions'):
    # この書き方なら solutions がなくても mp_pose が直接使えます
    pass

# pose の初期化部分を修正
pose = mp_pose.Pose(
    model_complexity=0, 
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# --- 角度計算関数 ---
def calculate_angle(a, b, c):
    a = np.array(a) # 肩または腰
    b = np.array(b) # 腰または膝
    c = np.array(c) # 膝または足首
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360-angle
    return angle

# --- 状態管理用クラス ---
class SquatTracker:
    def __init__(self):
        self.counter = 0
        self.stage = None # "down" or "up"
        self.feedback = "Ready"
        self.warning = ""

tracker = SquatTracker()

# --- 映像処理用コールバック ---
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    
    # 鏡のように反転
    img = cv2.flip(img, 1)
    
    # 解析用にRGBへ変換
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_img)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        
        # 必要な関節の座標を取得 (左半身を例に)
        shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
        ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

        # 角度計算
        knee_angle = calculate_angle(hip, knee, ankle)
        hip_angle = calculate_angle(shoulder, hip, knee)

        # --- スクワット判定ロジック ---
        # カウント
        if knee_angle < 100:
            tracker.stage = "down"
            tracker.feedback = "Go Up!"
        if knee_angle > 160 and tracker.stage == 'down':
            tracker.stage = "up"
            tracker.counter += 1
            tracker.feedback = "Good!"
        
        # 姿勢警告（例：背中が丸まっている、または膝が出すぎている）
        if tracker.stage == "down":
            if hip_angle < 80: # 腰が曲がりすぎ（極端な前傾）
                tracker.warning = "Keep Back Straight!"
            else:
                tracker.warning = ""
        else:
            tracker.warning = ""

        # --- 描画 ---
        # 骨格描画
        mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        # 情報表示
        cv2.rectangle(img, (0,0), (280, 120), (245,117,16), -1)
        cv2.putText(img, f'COUNT: {tracker.counter}', (10, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, f'STAGE: {tracker.stage}', (10, 75), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, f' {tracker.feedback}', (10, 110), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # 警告表示（赤色）
        if tracker.warning:
            cv2.rectangle(img, (0, 400), (640, 480), (0,0,255), -1)
            cv2.putText(img, tracker.warning, (120, 450), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3, cv2.LINE_AA)

    return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- Streamlit UI設定 ---

# --- 簡易パスワード認証 ---
def check_password():
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False

    if not st.session_state["authenticated"]:
        # パスワードがまだ入力されていない場合
        password = st.text_input("パスワードを入力してください", type="password")
        if st.sidebar.button("ログイン") or password:
            if password == "passlike": # ここにパスワードを設定
                st.session_state["authenticated"] = True
                st.rerun()
            else:
                st.error("パスワードが違います")
        return False
    return True

# 認証が通らない場合は、ここで処理を止める
if not check_password():
    st.stop()

st.title("AI Squat Trainer")
st.write("全身をカメラに映してください。膝を90度程度まで曲げるとカウントされます。")

# WebRTCの設定 (ICEサーバーの設定を加えると接続が安定します)
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

webrtc_streamer(
    key="squat-counter",
    video_frame_callback=video_frame_callback,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False}, # 音声不要
    async_processing=True,
)

st.info("背中が丸まりすぎると画面下に警告が出ます。")