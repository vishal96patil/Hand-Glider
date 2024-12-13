import cv2
import pyautogui
import pygetwindow as gw
import mediapipe as mp
import time

# Open the camera
cap = cv2.VideoCapture(0)

# Variables for gesture control
hand_present = False
hand_landmarks_reference = None
prev_index_tip = (0, 0)

# Initialize smoothing variables
delta_x_smooth, delta_y_smooth = 0, 0

# Initialize mediapipe hands
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Calibration phase
calibration_frames = 30  # Number of frames for calibration
calibration_count = 0

while calibration_count < calibration_frames:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks and len(results.multi_hand_landmarks) == 1:
        hand_landmarks_reference = results.multi_hand_landmarks[0].landmark[8]  # Use index finger as reference
        calibration_count += 1

    cv2.imshow("Calibration", frame)
    cv2.waitKey(1)

cv2.destroyWindow("Calibration")  # Close the calibration window

# Main control phase
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks and len(results.multi_hand_landmarks) == 1:
        hand_present = True
        hand_landmarks = results.multi_hand_landmarks[0].landmark[8]  # Use index finger for tracking

        # Calculate the movement delta with adjusted sign
        delta_x = -(hand_landmarks.x - hand_landmarks_reference.x) * 100  # Adjust the multiplier for slower movement
        delta_y = -(hand_landmarks.y - hand_landmarks_reference.y) * 100  # Adjust the multiplier for slower movement

        # Smoothen the movement by applying exponential moving average
        smoothing_factor = 0.9
        delta_x_smooth = smoothing_factor * delta_x + (1 - smoothing_factor) * delta_x_smooth
        delta_y_smooth = smoothing_factor * delta_y + (1 - smoothing_factor) * delta_y_smooth

        # Update the cursor position
        pyautogui.moveRel(delta_x_smooth, delta_y_smooth)

        # Zooming gesture control (index finger and thumb)
        if abs(hand_landmarks.x - results.multi_hand_landmarks[0].landmark[4].x) < 0.1 and \
                abs(hand_landmarks.y - results.multi_hand_landmarks[0].landmark[4].y) < 0.1:
            pyautogui.hotkey('ctrl', '+')  # Zoom in
        elif abs(hand_landmarks.x - results.multi_hand_landmarks[0].landmark[5].x) < 0.1 and \
                abs(hand_landmarks.y - results.multi_hand_landmarks[0].landmark[5].y) < 0.1:
            pyautogui.hotkey('ctrl', '-')  # Zoom out

        # Media player control gestures (index finger and middle finger)
        elif abs(hand_landmarks.x - results.multi_hand_landmarks[0].landmark[12].x) < 0.1 and \
                abs(hand_landmarks.y - results.multi_hand_landmarks[0].landmark[12].y) < 0.1:
            pyautogui.press('space')  # Play/Pause

        # Window switching gesture (index, middle, and ring finger)
        elif (   
            abs(hand_landmarks.x - results.multi_hand_landmarks[0].landmark[12].x) < 0.1 and
            abs(hand_landmarks.y - results.multi_hand_landmarks[0].landmark[12].y - results.multi_hand_landmarks[0].landmark[16].y) < 0.1
        ):
            current_window = gw.getActiveWindow()
            next_window = gw.getWindowsWithTitle(current_window.title)
            if len(next_window) > 1:
                next_window = next_window[1]
                next_window.activate()

        # Left click (thumb press to index finger)
        elif (
            abs(results.multi_hand_landmarks[0].landmark[4].x - hand_landmarks.x) < 0.1 and
            abs(results.multi_hand_landmarks[0].landmark[4].y - hand_landmarks.y) < 0.1
        ):
            pyautogui.click()

        # Right click (show pinky finger)
        elif abs(hand_landmarks.y - results.multi_hand_landmarks[0].landmark[20].y) < 0.1:
            pyautogui.rightClick()

    else:
        hand_present = False

    # Display the frame
    cv2.imshow("Gesture Control", frame)

    # Break the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
