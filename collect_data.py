# import cv2
# import mediapipe as mp
# import csv

# # Gesture label
# gesture_name = input("Enter gesture name: ")

# # CSV file
# file = open("dataset.csv", "a", newline="")
# writer = csv.writer(file)

# # Webcam
# cap = cv2.VideoCapture(0)

# # MediaPipe Hands
# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands()

# # Drawing utility
# mp_draw = mp.solutions.drawing_utils

# sample_count = 0

# while True:

#     success, frame = cap.read()

#     if not success:
#         print("Failed to capture frame")
#         break

#     # Convert BGR -> RGB
#     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#     # Process frame
#     results = hands.process(rgb_frame)

#     # If hand detected
#     if results.multi_hand_landmarks:

#         for hand_landmarks in results.multi_hand_landmarks:

#             # Draw landmarks
#             mp_draw.draw_landmarks(
#                 frame,
#                 hand_landmarks,
#                 mp_hands.HAND_CONNECTIONS
#             )

#             row = []

#             # Extract x,y,z
#             for landmark in hand_landmarks.landmark:

#                 row.append(landmark.x)
#                 row.append(landmark.y)
#                 row.append(landmark.z)
#             # Pad to 126 features
#             while len(row) < 126:
#              row.append(0)
            
            
            
#             print("Before label:", len(row))

#             row.append(gesture_name)

#             print("After label:", len(row))
#             writer.writerow(row)

#             sample_count += 1

#         print(f"Samples: {sample_count}", end="\r")

        

#     # Show webcam
#     cv2.imshow("Collect Data", frame)
    
#     if sample_count >= 150:
#         print("150 samples collected")
#         break

#     # Exit on ESC or q
#     key = cv2.waitKey(1)

#     if key == 27 or key == ord('q'):
#         break

# # Cleanup
# file.close()
# cap.release()
# cv2.destroyAllWindows()


import cv2
import mediapipe as mp
import pandas as pd

# -------------------------
# CONFIG
# -------------------------

CSV_FILE = "dataset.csv"

TARGET = int(input("Target Label (0-25): "))
TOTAL_SAMPLES = int(input("Number of samples to collect: "))

# -------------------------
# Read existing CSV
# -------------------------

df = pd.read_csv(CSV_FILE)

columns = list(df.columns)

print(f"\nDataset loaded.")
print(f"Columns : {len(columns)}")

# -------------------------
# MediaPipe
# -------------------------

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils

# -------------------------
# Webcam
# -------------------------

cap = cv2.VideoCapture(0)

samples = 0

while True:

    success, frame = cap.read()

    if not success:
        break

    frame = cv2.flip(frame,1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    if results.multi_hand_landmarks:

        left = [0.0]*63
        right = [0.0]*63

        uses_two_hands = 0

        if len(results.multi_hand_landmarks)==2:
            uses_two_hands=1

        for hand_landmarks, handedness in zip(
                results.multi_hand_landmarks,
                results.multi_handedness):

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            coords=[]

            for lm in hand_landmarks.landmark:

                coords.extend([
                    lm.x,
                    lm.y,
                    lm.z
                ])

            hand_label=handedness.classification[0].label

            if hand_label=="Left":
                left=coords
            else:
                right=coords

        row=[TARGET,uses_two_hands]

        row.extend(left)

        row.extend(right)

        if len(row)==len(columns):

            df.loc[len(df)] = row

            samples += 1

            cv2.putText(
                frame,
                f"Collected : {samples}/{TOTAL_SAMPLES}",
                (10,40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,255,0),
                2
            )

    cv2.imshow("Collect Data",frame)

    key=cv2.waitKey(1)

    if key==ord('q') or key==27:
        break

    if samples>=TOTAL_SAMPLES:
        break

cap.release()

cv2.destroyAllWindows()

df.to_csv(CSV_FILE,index=False)

print(f"\nAdded {samples} samples.")
print("Dataset saved.")
