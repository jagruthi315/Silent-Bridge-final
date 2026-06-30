import cv2
import mediapipe as mp

# Webcam connection
cap = cv2.VideoCapture(0)

# MediaPipe Hands setup
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

while True:

    success, frame = cap.read()

    if not success:
        print("Failed to capture frame")
        break

    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    # Process frame with MediaPipe
    hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
    results = hands.process(rgb_frame)

    # If hand detected
    if results.multi_hand_landmarks:

        all_landmarks = []

        for hand_landmarks in results.multi_hand_landmarks:

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            for landmark in hand_landmarks.landmark:

                all_landmarks.append(landmark.x)
                all_landmarks.append(landmark.y)
                all_landmarks.append(landmark.z)

        # Pad to 126 features
        while len(all_landmarks) < 126:
            all_landmarks.append(0)

        print(
            "Hands:",
            len(results.multi_hand_landmarks),
            "Features:",
            len(all_landmarks)
        )

    cv2.imshow("Hand Detection", frame)

    key = cv2.waitKey(1)

    if key == 27 or key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
