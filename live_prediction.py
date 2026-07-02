import cv2
import mediapipe as mp
import time
from collections import deque, Counter

from predict_gesture import predict

# ---------------------------------
# Label Mapping
# ---------------------------------

label_map = {
    0: "A",
    1: "B",
    2: "C",
    3: "D",
    4: "E",
    5: "F",
    6: "G",
    7: "H",
    8: "I",
    9: "J",
    10: "K",
    11: "L",
    12: "M",
    13: "N",
    14: "O",
    15: "P",
    16: "Q",
    17: "R",
    18: "S",
    19: "T",
    20: "U",
    21: "V",
    22: "W",
    23: "X",
    24: "Y",
    25: "Z"
}

# ---------------------------------
# Settings
# ---------------------------------

GESTURE_HOLD_TIME = 0.3
MIN_CONFIDENCE = 70

sentence = []
current_gesture = ""
gesture_start_time = 0
last_added_gesture = ""

prediction_buffer = deque(maxlen=5)

# ---------------------------------
# Webcam
# ---------------------------------

cap = cv2.VideoCapture(0)

# ---------------------------------
# MediaPipe
# ---------------------------------

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# ---------------------------------
# FPS
# ---------------------------------

prev_time = time.time()

# ---------------------------------
# Main Loop
# ---------------------------------

while True:

    success, frame = cap.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    if results.multi_hand_landmarks:

        left_hand = [0.0] * 63
        right_hand = [0.0] * 63

        for hand_landmarks, handedness in zip(
            results.multi_hand_landmarks,
            results.multi_handedness
        ):

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            coords = []

            for lm in hand_landmarks.landmark:
                coords.extend([lm.x, lm.y, lm.z])

            hand_label = handedness.classification[0].label

            if hand_label == "Left":
                left_hand = coords
            else:
                right_hand = coords

        all_landmarks = left_hand + right_hand

        prediction, confidence, probabilities, classes = predict(all_landmarks)

       

        # ===============================

        prediction_buffer.append(int(prediction))

        prediction = Counter(prediction_buffer).most_common(1)[0][0]

        gesture = label_map[prediction]

        current_time = time.time()

        if confidence >= MIN_CONFIDENCE:

            if gesture != current_gesture:

                current_gesture = gesture
                gesture_start_time = current_time

            else:

                if current_time - gesture_start_time >= GESTURE_HOLD_TIME:

                    if gesture != last_added_gesture:

                        sentence.append(gesture)

                        last_added_gesture = gesture

                        print("Added:", gesture)

        cv2.putText(
            frame,
            f"{gesture} ({confidence:.1f}%)",
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

    else:

        cv2.putText(
            frame,
            "Waiting for hand...",
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

    # Sentence

    text = " ".join(sentence)

    cv2.putText(
        frame,
        text,
        (10, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 0, 0),
        2
    )

    # FPS

    curr_time = time.time()

    fps = 1 / (curr_time - prev_time)

    prev_time = curr_time

    cv2.putText(
        frame,
        f"FPS: {int(fps)}",
        (500, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 0),
        2
    )

    cv2.imshow("SilentBridge", frame)

    key = cv2.waitKey(1)

    if key == ord("c"):

        sentence.clear()
        prediction_buffer.clear()
        current_gesture = ""
        last_added_gesture = ""

    if key == ord("q") or key == 27:
        break

cap.release()
cv2.destroyAllWindows()
