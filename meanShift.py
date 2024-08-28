import cv2

# Load the pre-trained Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Take the first frame of the video
ret, frame = cap.read()

# Convert the frame to grayscale (Haar cascades work better with grayscale images)
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Detect faces in the frame
face_rects = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# Check if any face is detected
if len(face_rects) > 0:
    # Get the coordinates of the first detected face
    x, y, w, h = tuple(face_rects[0])
    track_window = (x, y, w, h)

    # Set up the ROI for tracking
    roi = frame[y:y+h, x:x+w]
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    # Create a mask to filter out low light values
    mask = cv2.inRange(hsv_roi, (0, 60, 32), (180, 255, 255))
    roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])

    # Normalize the histogram
    cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

    # Set up the termination criteria, either 10 iterations or move by at least 1 pt
    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Back-projection to get the probability distribution
        dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

        ##########################################################
        # Apply mean shift to get the new location
        ret, track_window = cv2.meanShift(dst, track_window, term_crit)

        # Draw the tracked window on the frame
        x, y, w, h = track_window
        img2 = cv2.rectangle(frame, (x, y), (x+w, y+h), 255, 2)
        ##########################################################

        # Display the result
        cv2.imshow('Mean Shift Tracking', img2)

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
else:
    print("No face detected in the first frame.")
    cap.release()
    cv2.destroyAllWindows()
