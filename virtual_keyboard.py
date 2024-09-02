import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import time

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1,
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Define Virtual Keyboard Layout
keys = [
    ['Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P'],
    ['A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L'],
    ['Z', 'X', 'C', 'V', 'B', 'N', 'M']
]

key_size = 60
key_gap = 10
start_x = 50
start_y = 300

# Function to draw the keyboard
def draw_keyboard(img, keys, key_size, key_gap, start_x, start_y, active_key=None):
    for row_idx, row in enumerate(keys):
        for key_idx, key in enumerate(row):
            x = start_x + key_idx * (key_size + key_gap)
            y = start_y + row_idx * (key_size + key_gap)
            if active_key == key:
                cv2.rectangle(img, (x, y), (x + key_size, y + key_size), (0, 255, 0), cv2.FILLED)
                cv2.putText(img, key, (x + 15, y + 45),
                            cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
            else:
                cv2.rectangle(img, (x, y), (x + key_size, y + key_size), (255, 0, 0), 2)
                cv2.putText(img, key, (x + 15, y + 45),
                            cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)

# Function to get the center of a key
def get_key_position(row_idx, key_idx, key_size, key_gap, start_x, start_y):
    x = start_x + key_idx * (key_size + key_gap) + key_size // 2
    y = start_y + row_idx * (key_size + key_gap) + key_size // 2
    return (x, y)

# Function to find which key is pressed
def find_pressed_key(finger_tip, keys, key_size, key_gap, start_x, start_y):
    for row_idx, row in enumerate(keys):
        for key_idx, key in enumerate(row):
            key_x, key_y = get_key_position(row_idx, key_idx, key_size, key_gap, start_x, start_y)
            distance = np.hypot(finger_tip[0] - key_x, finger_tip[1] - key_y)
            if distance < key_size // 2:
                return key
    return None

# To prevent multiple key presses for a single gesture
pressed = False
last_key = ''
last_time = time.time()

# Initialize Webcam
cap = cv2.VideoCapture(0)

# Accumulate typed text
typed_text = ""

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)  # Mirror the image

    # Increase the size of the window
    img = cv2.resize(img, (1200, 800))

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    active_key = None

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get finger tip coordinates (Index finger tip is landmark 8, Middle finger tip is landmark 12)
            index_finger_tip = hand_landmarks.landmark[8]
            middle_finger_tip = hand_landmarks.landmark[12]
            h, w, c = img.shape
            index_finger_x, index_finger_y = int(index_finger_tip.x * w), int(index_finger_tip.y * h)
            middle_finger_x, middle_finger_y = int(middle_finger_tip.x * w), int(middle_finger_tip.y * h)

            # Draw circles at the finger tips
            cv2.circle(img, (index_finger_x, index_finger_y), 10, (0, 255, 255), cv2.FILLED)
            cv2.circle(img, (middle_finger_x, middle_finger_y), 10, (255, 0, 255), cv2.FILLED)

            # Check if both fingers are close enough to register a key press
            fingers_distance = np.hypot(index_finger_x - middle_finger_x, index_finger_y - middle_finger_y)
            if fingers_distance < 40:  # Adjust this value as needed
                # Find which key is pressed
                key = find_pressed_key((index_finger_x, index_finger_y), keys, key_size, key_gap, start_x, start_y)
                if key and not pressed:
                    active_key = key
                    pyautogui.press(key)
                    pressed = True
                    last_key = key
                    last_time = time.time()
                    typed_text += key  # Append the key to the typed text
                    print(f"Key Pressed: {key}")

    # Reset pressed status after a short delay
    if pressed and (time.time() - last_time) > 0.5:
        pressed = False

    # Draw the virtual keyboard
    draw_keyboard(img, keys, key_size, key_gap, start_x, start_y, active_key)

    # Draw the typed text below the keyboard
    cv2.putText(img, typed_text, (start_x, start_y + len(keys) * (key_size + key_gap) + 50),
                cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 3)

    # Display the image
    cv2.imshow("Virtual Keyboard", img)

    # Exit on pressing 'Esc'
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
