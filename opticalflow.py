import numpy as np
import matplotlib.pyplot as plt
import cv2

corner_track_params = dict(maxCorners=10, qualityLevel=0.3, minDistance=7, blockSize=7)
lk_params = dict(winSize=(200, 200), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
cap = cv2.VideoCapture(0)

ret, prev_frame = cap.read()
if not ret:
    print("Failed to grab the first frame.")
    cap.release()
    cv2.destroyAllWindows()
    exit()

prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

# Points to Track
prevpts = cv2.goodFeaturesToTrack(prev_gray, mask=None, **corner_track_params)

# Initialize the mask to draw the tracks
mask = np.zeros_like(prev_frame)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab a frame.")
        break
    
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    nextpts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, frame_gray, prevpts, None, **lk_params)
    
    if nextpts is not None and status is not None:
        good_new = nextpts[status == 1]
        good_prev = prevpts[status == 1]
        
        for i, (new, prev) in enumerate(zip(good_new, good_prev)):
            x_new, y_new = map(int, new.ravel())
            x_prev, y_prev = map(int, prev.ravel())
            mask = cv2.line(mask, (x_new, y_new), (x_prev, y_prev), (0, 255, 0), 3)
            frame = cv2.circle(frame, (x_new, y_new), 8, (0, 0, 255), -1)
    
    img = cv2.add(frame, mask)
    cv2.imshow('Tracking', img)
    
    if cv2.waitKey(1) & 0xFF == 27:
        break
    
    prev_gray = frame_gray.copy()
    prevpts = good_new.reshape(-1, 1, 2)

cap.release()
cv2.destroyAllWindows()
