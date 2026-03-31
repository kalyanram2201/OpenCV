import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

pTime = 0

while True:
    r, frame = cap.read()
    if not r:
        break

    frame = cv2.flip(frame, 1)

    frgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = hands.process(frgb)

    if res.multi_hand_landmarks:
        for mark in res.multi_hand_landmarks:
            for id,lm in enumerate(mark.landmarl):
                h,w,c=frame.shape
                cx,cy=int(lm.x*w),int(lm.y*h)
                if id==0:
                    cv2.circle(frame,)
            mpDraw.draw_landmarks(frame, mark, mpHands.HAND_CONNECTIONS)

    cTime = time.time()
    fps = 1 / (cTime - pTime) if (cTime - pTime) != 0 else 0
    pTime = cTime

    cv2.putText(frame, str(int(fps)), (10, 70),
                cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 4)

    cv2.imshow("Video", frame)

    if cv2.waitKey(1) & 0xFF == ord('p'):
        break

cap.release()
cv2.destroyAllWindows()