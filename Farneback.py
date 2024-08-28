import cv2
import numpy as np
cap = cv2.VideoCapture(0)

ret,prev_frame = cap.read()
prev_gray = cv2.cvtColor(prev_frame,cv2.COLOR_BGR2GRAY)

while True:
    ret,frame = cap.read()
    frame_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prev_gray,frame_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv = np.zeros_like(prev_frame)
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,1] = 255
    hsv[...,2] = cv2.normalize(mag,None, 0, 255,cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    cv2.imshow('Farneback',bgr)
    if cv2.waitKey(1) & 0xFF == 27:
        break
    prev_gray = frame_gray.copy()

cap.release()
cv2.destroyAllWindows()
