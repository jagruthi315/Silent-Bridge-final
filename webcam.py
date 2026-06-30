import cv2

cap =cv2.VideoCapture(0)

while True:
    success , frame = cap.read()
    
    if not success:
        print("Failed to capture video")
        break
    cv2.imshow('webcam',frame)

    if cv2.waitKey(1) == 27 or cv2.waitKey(1) == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()