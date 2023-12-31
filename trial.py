import cv2
import numpy as np

# Load the pre-trained Haar cascade classifiers for face and eyes
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

def eyetracking(frame):
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_blurred = cv2.blur(gray, (7, 7)) 

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Draw a rectangle around the detected face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Region of interest (ROI) for eyes within the face rectangle
        roi_gray = gray[y:y+h//2, x:x+w]
        roi_color = frame[y:y+h//2, x:x+w]
        roi_grayblur = gray_blurred[y:y+h//2, x:x+w]

        # Detect eyes in the ROI
        eyes = eye_cascade.detectMultiScale(roi_gray)

        for (ex, ey, ew, eh) in eyes:
            # Draw a rectangle around each eye
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
            # implement eyeball tracking here.
            detected_circles = cv2.HoughCircles(roi_grayblur, cv2.HOUGH_GRADIENT, 1, 20, param1 = 50, param2 = 30, minRadius = 4, maxRadius = 20)
            if detected_circles is not None:
                detected_circles = np.uint16(np.around(detected_circles))		
                for pt in detected_circles[0,:]:
                    a, b, r = pt[0], pt[1], pt[2]                           # circle is centred at (a,b)
                    cv2.circle(roi_color, (a, b), r, (0, 255, 255), 2) 
    
    return frame
# Initialize the webcam
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    frame = eyetracking(frame)
    # Display the resulting frame
    cv2.imshow('Eye Tracking', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
