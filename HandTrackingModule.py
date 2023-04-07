import cv2
import mediapipe as mp
import time


class HandDetector():
    
    def __init__(self, 
                 static_image_mode=False, 
                 max_num_hands=2,
                 modelC = 1, 
                 min_detection_confidence=0.5, 
                 min_tracking_confidence=0.5):
        self.static_image_mode =  static_image_mode
        self.max_num_hands = max_num_hands
        self.modelC = modelC
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.static_image_mode,
                                        self.max_num_hands,
                                        self.modelC,
                                        self.min_detection_confidence,
                                        self.min_tracking_confidence)
        self.draw = mp.solutions.drawing_utils
    
    
    def findHands(self,frame,draw=True):
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(frameRGB)

        if self.results.multi_hand_landmarks:
            for hand in self.results.multi_hand_landmarks:
                if draw:
                    self.draw.draw_landmarks(frame, hand, self.mpHands.HAND_CONNECTIONS)
        
        return frame
    
    def findPosition(self, frame, handNo=0, draw=True):
        
        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h,w,c = frame.shape
                cx,cy = int(lm.x*w), int(lm.y*h)
                # print(f'ID : {id} POS : {cx} {cy}')
                lmList.append([id, cx, cy])
                
                if draw:
                    cv2.circle(frame, (cx, cy), 7, (0,255,0), cv2.FILLED)
        
        return lmList
                
    


def main():
    
    cTime = 0
    pTime = 0
    
    capture = cv2.VideoCapture(0)
    
    detector = HandDetector()
    
    while True:
        success, frame = capture.read()
        frame = detector.findHands(frame)
        lmList = detector.findPosition(frame)
        if len(lmList) != 0:
            print(lmList[4])
        #FPS Calculation
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime
        cv2.putText(frame, str(int(fps)), (10,70), cv2.FONT_HERSHEY_DUPLEX, 1, (255,0,255), 3)

        cv2.imshow('Live_Stream', frame)
        cv2.waitKey(1)


if __name__ == '__main__':
    main()