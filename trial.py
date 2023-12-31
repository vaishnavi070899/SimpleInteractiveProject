import cv2
import numpy as np

# Load the pre-trained Haar cascade classifiers for face and eyes
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

def eyetracking(frame):    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_blurred = cv2.blur(gray, (8, 8)) 

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Draw a rectangle around the detected face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Region of interest (ROI) for eyes within the face rectangle
        roi_gray = gray[y:y+h//2, x:x+w]
        roi_color = frame[y:y+h//2, x:x+w]
        roi_grayblur = gray_blurred[y:y+h//2, x:x+w]
        
        _, threshold = cv2.threshold(roi_grayblur, 3, 255, cv2.THRESH_BINARY_INV)

        # Detect eyes in the ROI
        eyes = eye_cascade.detectMultiScale(roi_gray)

        for (ex, ey, ew, eh) in eyes:
            # Draw a rectangle around each eye
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
            # implement eyeball tracking here.
            contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
            for cnt in contours:
                (x, y, w, h) = cv2.boundingRect(cnt)
                cols = roi_color.shape[1]
                rows = roi_color.shape[0]
                cv2.rectangle(roi_color, (x, y), (x + w, y + h), (0, 255, 0), 1)  # Uncomment to visualize bounding rectangles
                cv2.line(roi_color, (x + int(w/2), 0), (x + int(w/2), rows), (0, 255, 0), 2)
                cv2.line(roi_color, (0, y + int(h/2)), (cols, y + int(h/2)), (0, 255, 0), 2)
                break
    
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
