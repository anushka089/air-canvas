import cv2
import mediapipe as mp
import numpy as np

# -----------------------------
# MediaPipe Hand Detection
# -----------------------------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# -----------------------------
# Open Webcam
# -----------------------------
cap = cv2.VideoCapture(0)

# Canvas
canvas = None

# Previous finger position
prev_x = 0
prev_y = 0

while True:

    success, frame = cap.read()

    if not success:
        print("Could not access camera")
        break

    # Mirror camera
    frame = cv2.flip(frame, 1)

    # Create canvas
    if canvas is None:
        canvas = np.zeros_like(frame)

    # Convert BGR -> RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect hand
    results = hands.process(rgb_frame)

    # If hand detected
    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:

            # Draw hand landmarks
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            # -----------------------------
            # Get INDEX FINGER TIP
            # Landmark 8 = index finger tip
            # -----------------------------
            index_finger = hand_landmarks.landmark[8]

            h, w, c = frame.shape

            x = int(index_finger.x * w)
            y = int(index_finger.y * h)

            # Draw a small circle on fingertip
            cv2.circle(
                frame,
                (x, y),
                10,
                (255, 0, 255),
                -1
            )

            # -----------------------------
            # Check if index finger is UP
            # -----------------------------
            index_tip = hand_landmarks.landmark[8]
            index_pip = hand_landmarks.landmark[6]

            if index_tip.y < index_pip.y:

                # Start drawing
                if prev_x == 0 and prev_y == 0:
                    prev_x = x
                    prev_y = y

                # Draw line
                cv2.line(
                    canvas,
                    (prev_x, prev_y),
                    (x, y),
                    (255, 0, 255),
                    5
                )

                prev_x = x
                prev_y = y

            else:
                # Finger down = stop drawing
                prev_x = 0
                prev_y = 0

    # -----------------------------
    # Combine camera + canvas
    # -----------------------------
    gray_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)

    _, mask = cv2.threshold(
        gray_canvas,
        20,
        255,
        cv2.THRESH_BINARY
    )

    mask_inv = cv2.bitwise_not(mask)

    camera_background = cv2.bitwise_and(
        frame,
        frame,
        mask=mask_inv
    )

    drawing = cv2.bitwise_and(
        canvas,
        canvas,
        mask=mask
    )

    frame = cv2.add(
        camera_background,
        drawing
    )

    # -----------------------------
    # Instructions
    # -----------------------------
    cv2.putText(
        frame,
        "INDEX FINGER = DRAW",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        "Press C = Clear | Q = Quit",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    # Show window
    cv2.imshow("Air Canvas", frame)

    # Keyboard controls
    key = cv2.waitKey(1) & 0xFF

    if key == ord('c'):
        canvas = np.zeros_like(frame)

    if key == ord('q'):
        break

# -----------------------------
# Release everything
# -----------------------------
cap.release()
cv2.destroyAllWindows()
hands.close()