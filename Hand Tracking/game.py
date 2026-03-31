import cv2
import mediapipe as mp
import time
import module as htm


cap = cv2.VideoCapture(0)
detector = htm.HandDetector()

pTime = 0

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)

    frame = detector.findHands(frame)
    lmList = detector.findPosition(frame)

    if len(lmList) != 0:
        print(lmList[5])  # wrist point

    cTime = time.time()
    fps = 1 / (cTime - pTime) if (cTime - pTime) != 0 else 0
    pTime = cTime

    cv2.putText(frame, f'FPS: {int(fps)}', (10, 70),
                cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)

    cv2.imshow("Hand Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('p'):
        break

cap.release()
cv2.destroyAllWindows()