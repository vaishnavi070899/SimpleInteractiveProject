import cv2
import numpy as np

# Load the pre-trained Haar cascade classifiers for face and eyes
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

def find_pupil(eye_roi):
    # Apply Gaussian blur to the eye region
    eye_blur = cv2.GaussianBlur(eye_roi, (5, 5), 0)

    # Use adaptive thresholding to highlight the pupil
    _, threshold = cv2.threshold(eye_blur, 60, 255, cv2.THRESH_BINARY_INV)

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the contour with the maximum area (assumed to be the pupil)
    if contours:
        pupil_contour = max(contours, key=cv2.contourArea)
        (x, y, w, h) = cv2.boundingRect(pupil_contour)

        # Draw a rectangle around the pupil
        cv2.rectangle(eye_roi, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Draw lines to indicate the center of the pupil
        cx, cy = x + w // 2, y + h // 2
        # cv2.line(eye_roi, (cx, 0), (cx, eye_roi.shape[0]), (0, 255, 0), 2)
        # cv2.line(eye_roi, (0, cy), (eye_roi.shape[1], cy), (0, 0, 255), 2)

        return cx, cy

    return None, None

def eyedetecting(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Draw a rectangle around the detected face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Region of interest (ROI) for eyes within the face rectangle
        roi_gray = gray[y:y+h//2, x:x+w]
        roi_color = frame[y:y+h//2, x:x+w]

        # Detect eyes in the ROI
        eyes = eye_cascade.detectMultiScale(roi_gray)

        for (ex, ey, ew, eh) in eyes:
            # Draw a rectangle around each eye
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

            # Extract the region of interest for the eye
            eye_roi = roi_gray[ey:ey+eh, ex:ex+ew]

            # Perform pupil detection
            pupil_x, pupil_y = find_pupil(eye_roi)

            # Draw a line connecting the center of the eye to the estimated pupil position
            if pupil_x is not None and pupil_y is not None:
                cv2.line(roi_color, (ex + int(ew / 2), ey + int(eh / 2)), (x + pupil_x, y + pupil_y), (0, 255, 255), 2)

    return frame

# Initialize the webcam
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    frame = eyedetecting(frame)
    # Display the resulting frame
    cv2.imshow('Eye Tracking', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
