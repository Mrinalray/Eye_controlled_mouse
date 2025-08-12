import cv2
import pyautogui
import mediapipe as mp
import time

# MediaPipe FaceMesh setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Screen size
screen_w, screen_h = pyautogui.size()

# Webcam
cap = cv2.VideoCapture(0)

# Blink detection threshold
BLINK_THRESHOLD = 0.007
blink_was_closed = False
last_left_click_time = 0

# UI click indicators
left_click_active = False
right_click_active = False
CLICK_DISPLAY_TIME = 0.3  # Seconds

def draw_mouse_buttons_ui(frame, left_active, right_active):
    # Left click button (top-left corner)
    left_color = (0, 255, 0) if not left_active else (0, 0, 255)
    right_color = (0, 255, 0) if not right_active else (0, 0, 255)

    # Draw left button
    cv2.rectangle(frame, (30, 30), (130, 80), left_color, -1)
    cv2.putText(frame, "Left", (45, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Draw right button
    cv2.rectangle(frame, (150, 30), (250, 80), right_color, -1)
    cv2.putText(frame, "Right", (165, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    frame_h, frame_w, _ = frame.shape

    current_time = time.time()

    # Reset indicators if time passed
    if current_time - last_left_click_time > CLICK_DISPLAY_TIME:
        left_click_active = False
        right_click_active = False

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark

        # === Mouse movement using left iris ===
        if len(landmarks) > 469:
            left_iris = landmarks[469]
            screen_x = screen_w * left_iris.x
            screen_y = screen_h * left_iris.y
            pyautogui.moveTo(screen_x, screen_y)

        # === Left-click using right eye blink ===
        if len(landmarks) > 386:
            top_lid = landmarks[386]
            bottom_lid = landmarks[374]
            vertical_dist = abs(top_lid.y - bottom_lid.y)

            if vertical_dist < BLINK_THRESHOLD:
                if not blink_was_closed:
                    pyautogui.click()
                    blink_was_closed = True
                    left_click_active = True
                    last_left_click_time = current_time
            else:
                blink_was_closed = False

        # === Draw right eye indicator ===
        if len(landmarks) > 473:
            right_iris = landmarks[473]
            rx = int(right_iris.x * frame_w)
            ry = int(right_iris.y * frame_h)
            color = (0, 0, 255) if left_click_active else (0, 255, 0)
            cv2.circle(frame, (rx, ry), 8, color, -1)

        # === Optional: draw left eye landmarks as dots (for visual tracking) ===
        LEFT_EYE_LANDMARKS = [
            33, 246, 161, 160, 159, 158, 157, 173,
            133, 155, 154, 153, 145, 144, 163, 7
        ]
        for idx in LEFT_EYE_LANDMARKS:
            point = landmarks[idx]
            px = int(point.x * frame_w)
            py = int(point.y * frame_h)
            cv2.circle(frame, (px, py), 2, (255, 0, 255), -1)

    # === Draw UI buttons (left/right mouse indicators) ===
    draw_mouse_buttons_ui(frame, left_click_active, right_click_active)

    # Show the output
    cv2.imshow("Eye Mouse | Left Eye = Move | Right Eye Blink = Click", frame)

    # ESC key to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
