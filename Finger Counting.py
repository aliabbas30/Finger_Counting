import cv2
import os
import time
import mediapipe as mp
wCam, hCam = 640, 480
cap = cv2.VideoCapture(0)
cap.set(3,wCam)
cap.set(4,hCam)
folder_path = 'Images'
mylist = os.listdir(folder_path)
overlaylist = []
for path in mylist:
    img = cv2.imread(f'{folder_path}/{path}')
    overlaylist.append(img)
print(len(overlaylist))

mpHands = mp.solutions.hands
# ONLY USE RGB
hands = mpHands.Hands()
mp_draw = mp.solutions.drawing_utils


pTime = 0
tipids = [4, 8, 12, 16, 20]
while True:
    success, img =cap.read()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ## Detection started
    results = hands.process(img_rgb)
    if results.multi_hand_landmarks:
        lmlist = []
        for handlms in results.multi_hand_landmarks:
            for ids, lm in enumerate(handlms.landmark):
                ## get the pixel values
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmlist.append([ids, cx, cy])
                cv2.circle(img, (cx, cy), 5, (255, 0, 24), cv2.FILLED)
                mp_draw.draw_landmarks(img, handlms, mpHands.HAND_CONNECTIONS)
        if len(lmlist) != 0:
            fingers = []
            ## For Thumb
            if lmlist[tipids[0]][1] < lmlist[tipids[0] - 1][1]:
                fingers.append(1)
            else:
                fingers.append(0)
            ## 4 fingers
            for id in range(1, 5):
                if lmlist[tipids[id]][2] < lmlist[tipids[id]-2][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)

            # print(fingers)
            totalfingers = fingers.count(1)
            print(totalfingers)
            ##image ploting
            h, w, c = overlaylist[totalfingers].shape
            img[0:h, 0:w] = overlaylist[totalfingers-1]
            cv2.rectangle(img, (20, 255), (170, 425), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, str(totalfingers), (45, 375), cv2.FONT_HERSHEY_PLAIN, 10, (255, 0, 0), 5)
    ##Frame per second
    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (400, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 2)
    cv2.imshow('Image', img)
    cv2.waitKey(1)