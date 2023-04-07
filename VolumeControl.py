import cv2
import mediapipe as mp
import HandTrackingModule as htm
import time
import numpy as np
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume




devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
volume_range = volume.GetVolumeRange()
min_volume = volume_range[0]
max_volume = volume_range[1]
# volume.SetMasterVolumeLevel(-20.0, None)

################
wCam, hCam = 1280, 720
################

cTime = 0
pTime = 0
capture = cv2.VideoCapture(0)
capture.set(3, wCam)
capture.set(4, hCam)
detector = htm.HandDetector(min_detection_confidence=0.7)

while True:
    success, frame = capture.read()
    frame = detector.findHands(frame)
    lmList = detector.findPosition(frame, draw=False)
    if len(lmList) != 0:
        x1,y1 = lmList[4][1], lmList[4][2]
        x2,y2 = lmList[8][1], lmList[8][2]
        
        cx,cy = (x1+x2)//2, (y1+y2)//2
        
        cv2.circle(frame,(x1,y1), 7, (0,255,255), cv2.FILLED)
        cv2.circle(frame,(x2,y2), 7, (0,255,255), cv2.FILLED)
        cv2.line(frame,(x1,y1), (x2,y2), (255,0,255), 3)
        cv2.circle(frame,(cx,cy), 7, (0,255,255), cv2.FILLED)
        
        length = math.hypot(x2-x1, y2-y1)
        
        #Hand Range ---> 25 - 300
        #Volume Range ---> -63.5 - 0
        vol = np.interp(length, [10,200], [min_volume,max_volume])
        volume.SetMasterVolumeLevel(vol, None)
        print(vol)
    cv2.imshow('Live', frame)
    cv2.waitKey(1)