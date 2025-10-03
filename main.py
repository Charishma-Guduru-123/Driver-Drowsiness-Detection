import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
import time
# optional windows beep
try:
    import winsound
    def beep(freq=1000, dur=300):
        winsound.Beep(freq, dur)
except Exception:
    # fallback: no-op beep (Streamlit will show alerts)
    def beep(freq=1000, dur=300):
        pass

# Streamlit page config
st.set_page_config(page_title="Driver Drowsiness Detection", layout="wide")
st.title("🚗 Driver Drowsiness Detection (EAR + MAR)")

# Initialize mediapipe face mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True,
                                  min_detection_confidence=0.5, min_tracking_confidence=0.5)

# EAR calculation (normalized landmarks work for ratio calculations)
def eye_aspect_ratio(landmarks, eye_indices):
    points = np.array([(landmarks[i].x, landmarks[i].y) for i in eye_indices])
    A = np.linalg.norm(points[1] - points[5])
    B = np.linalg.norm(points[2] - points[4])
    C = np.linalg.norm(points[0] - points[3])
    if C == 0:
        return 0.0
    ear = (A + B) / (2.0 * C)
    return ear

# MAR calculation (vertical / horizontal) using normalized landmarks
def mouth_aspect_ratio(landmarks, top_idx, bottom_idx, left_idx, right_idx):
    top = np.array((landmarks[top_idx].x, landmarks[top_idx].y))
    bottom = np.array((landmarks[bottom_idx].x, landmarks[bottom_idx].y))
    left = np.array((landmarks[left_idx].x, landmarks[left_idx].y))
    right = np.array((landmarks[right_idx].x, landmarks[right_idx].y))
    vertical = np.linalg.norm(top - bottom)
    horizontal = np.linalg.norm(left - right)
    if horizontal == 0:
        return 0.0
    return vertical / horizontal

# Eye and mouth landmark indices (MediaPipe Face Mesh)
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# Mouth indices (inner lip top/bottom and corners)
# Commonly used: top inner lip = 13, bottom inner lip = 14, left corner = 78, right corner = 308
MOUTH_TOP = 13
MOUTH_BOTTOM = 14
MOUTH_LEFT = 78
MOUTH_RIGHT = 308

# Thresholds and consecutive frame limits (tweak for your camera/lighting)
EAR_THRESHOLD = 0.25
CLOSED_FRAMES = 20

MAR_THRESHOLD = 0.60   # MAR above this suggests wide-open mouth (possible yawn)
MAR_CONSEC_FRAMES = 15

# Counters
eye_counter = 0
yawn_counter = 0

# Streamlit placeholders
frame_placeholder = st.empty()
alert_placeholder = st.empty()
stats_col1, stats_col2 = st.columns(2)
ear_text = stats_col1.empty()
mar_text = stats_col2.empty()

# Start webcam capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    st.error("Could not open webcam. Make sure camera index is correct and not in use.")
else:
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                st.warning("Empty frame received from camera. Exiting loop.")
                break

            h, w = frame.shape[:2]
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(frame_rgb)

            # defaults
            ear = 0.0
            mar = 0.0
            detected_alerts = []

            if results.multi_face_landmarks:
                # only handle first face
                face_landmarks = results.multi_face_landmarks[0]

                # compute EARs
                left_ear = eye_aspect_ratio(face_landmarks.landmark, LEFT_EYE)
                right_ear = eye_aspect_ratio(face_landmarks.landmark, RIGHT_EYE)
                ear = (left_ear + right_ear) / 2.0

                # compute MAR
                mar = mouth_aspect_ratio(face_landmarks.landmark, MOUTH_TOP, MOUTH_BOTTOM, MOUTH_LEFT, MOUTH_RIGHT)

                # --- Drowsiness (EAR) detection ---
                if ear < EAR_THRESHOLD:
                    eye_counter += 1
                    if eye_counter >= CLOSED_FRAMES:
                        detected_alerts.append("DROWSINESS")
                        alert_placeholder.error("🚨 Drowsiness Detected! Wake up!")
                        beep(2000, 500)
                else:
                    eye_counter = 0

                # --- Yawn detection (MAR) ---
                if mar > MAR_THRESHOLD:
                    yawn_counter += 1
                    if yawn_counter >= MAR_CONSEC_FRAMES:
                        detected_alerts.append("YAWN")
                        alert_placeholder.warning("😮 Yawning Detected!")
                        beep(1200, 500)
                else:
                    yawn_counter = 0

                # annotate frame: draw face mesh + debug points
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())

                # draw the specific eye & mouth points (pixel coords)
                for idx in LEFT_EYE + RIGHT_EYE + [MOUTH_TOP, MOUTH_BOTTOM, MOUTH_LEFT, MOUTH_RIGHT]:
                    lm = face_landmarks.landmark[idx]
                    x_px, y_px = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (x_px, y_px), 2, (0, 0, 255), -1)

                # put EAR and MAR on frame
                cv2.putText(frame, f"EAR: {ear:.3f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
                cv2.putText(frame, f"MAR: {mar:.3f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

            else:
                # no face found — reset counters and clear alert
                eye_counter = 0
                yawn_counter = 0
                alert_placeholder.empty()

            # update Streamlit stats & frame
            ear_text.markdown(f"**EAR:** {ear:.3f}  |  **EyeCnt:** {eye_counter}")
            mar_text.markdown(f"**MAR:** {mar:.3f}  |  **YawnCnt:** {yawn_counter}")

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb, channels="RGB")

            # small sleep to yield (helps CPU)
            time.sleep(0.01)

    except Exception as e:
        st.exception(f"Error in processing loop: {e}")
    finally:
        cap.release()
