cap = cv2.VideoCapture(0)

ret, prev_frame = cap.read()
prev_gray = cv2.cvtColor(prev_frame,cv2.COLOR_BGR2GRAY)

#Points to Track
prevpts = cv2.goodFeaturesToTrack(prev_gray,mask=None,**corner_track_params)

mask = np.zeros_like(prev_frame)

while True:
    ret, frame = cap.read()
    frame_gray= cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    nextpts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray,frame_gray,prevpts,None,**lk_params)
    good_new = nextpts[status==1]
    good_prev = prevpts[status==1]
    for i, (new,prev) in enumerate(zip(good_new,good_prev)):
        x_new, y_new = new.ravel()
        x_prev, y_prev = prev.ravel()
        mask = cv2.line(mask,(x_new,y_new),(x_prev,y_prev),(0,255,0),3)
        frame = cv2.circle(frame,(x_new,y_new),8,(0,0,255),-1)
    img = cv2.add(frame,mask)
    cv2.imshow('Tracking',img)
    if cv2.waitKey(1) & 0xFF == 27:
        break
    prev_gray = frame_gray.copy()
    prevpts = good_new.reshape(-1,1,2)

cap.release()
cv2.destroyAllWindows()
    
    
