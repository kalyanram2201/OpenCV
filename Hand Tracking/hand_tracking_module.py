import cv2
import mediapipe as mp
import time


class HandDetector:
    def __init__(self, mode=False, maxHands=1, detectionC=0.5, trackingC=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionC
        self.trackingCon = trackingC

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            model_complexity=0,  # 🔥 fastest model
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackingCon
        )
        self.mpDraw = mp.solutions.drawing_utils
        self.results = None

    def findHands(self, frame, draw=True):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(rgb)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(
                        frame, handLms, self.mpHands.HAND_CONNECTIONS
                    )
        return frame

    def findPosition(self, frame, handNo=0, draw=True):
        lmList = []

        if self.results and self.results.multi_hand_landmarks:
            hand = self.results.multi_hand_landmarks[handNo]

            for id, lm in enumerate(hand.landmark):
                h, w, _ = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append((id, cx, cy))

                if draw:
                    cv2.circle(frame, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

        return lmList


def main():
    cap = cv2.VideoCapture(0)

    # 🔥 Reduce camera resolution
    cap.set(3, 640)
    cap.set(4, 480)

    detector = HandDetector(maxHands=1)

    pTime = 0
    frame_count = 0

    while True:
        success, frame = cap.read()
        if not success:
            break

        # 🔥 Resize frame (major FPS boost)
        frame = cv2.resize(frame, (480, 360))
        frame = cv2.flip(frame, 1)

        frame_count += 1

        # 🔥 Skip every alternate frame
        if frame_count % 2 == 0:
            frame = detector.findHands(frame, draw=False)

        lmList = detector.findPosition(frame, draw=False)

        # Example: print index finger tip
        if len(lmList) != 0:
            print(lmList[8])

        # FPS calculation
        cTime = time.time()
        fps = 1 / (cTime - pTime) if (cTime - pTime) != 0 else 0
        pTime = cTime

        # Display FPS
        cv2.putText(frame, f'FPS: {int(fps)}', (10, 50),
                    cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

        cv2.imshow("Hand Tracking (Optimized)", frame)

        if cv2.waitKey(1) & 0xFF == ord('p'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()