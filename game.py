import cv2
import numpy
from PIL import Image, ImageDraw
import random
import numpy as np

def dot_gen():
    width, height = 480, 640
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    dot_size = 5
    for _ in range(5):
        x, y = random.randint(0, width - 1), random.randint(0, height - 1)
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        draw.ellipse((x - dot_size, y - dot_size, x + dot_size, y + dot_size), fill=color)

    center_x, center_y = width // 2, height // 2
    ball_radius = 20
    ball_color = (255, 0, 0)  
    draw.ellipse((center_x - ball_radius, center_y - ball_radius, center_x + ball_radius, center_y + ball_radius), fill=ball_color)
    return image

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

# read the input video captured by the camera
cap = cv2.VideoCapture(0)

while True:
    ret, image = cap.read()
    if not ret:
        break

# load the pretrained face detector    
    image = eyetracking(image)

    # Display the result
    cv2.imshow('Detected eyes', image)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()



