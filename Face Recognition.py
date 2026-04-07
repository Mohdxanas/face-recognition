import numpy as np
import cv2
import pickle
import os
from datetime import datetime

# --- SETUP ---
# Load cascades using relative paths
face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
eye_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_smile.xml')

# Load the trained recognizer model
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainner.yml')

# Load the labels (names)
labels = {"person_name": 1}
with open('labels.pickle', 'rb') as f:
    og_labels = pickle.load(f)
    labels = {v:k for k,v in og_labels.items()}

# --- ATTENDANCE FUNCTION ---
def mark_attendance(name):
    # We do not want to record "Unknown" faces
    if name == "Unknown":
        return
        
    file_name = "Attendance.csv"
    file_exists = os.path.isfile(file_name)
    today_date = datetime.now().strftime("%d-%m-%Y")
    already_marked = False

    # Check if the file exists and if the person is already marked present TODAY
    if file_exists:
        with open(file_name, 'r') as f:
            for line in f:
                if name in line and today_date in line:
                    already_marked = True
                    break

    # If they are not marked today, record their attendance!
    if not already_marked:
        with open(file_name, 'a') as f:
            now = datetime.now()
            time_str = now.strftime("%H:%M:%S")
            
            # If the file is brand new, add column headers first
            if not file_exists:
                f.write("Name,Time,Date\n") 
            
            # Write the data
            f.write(f"{name},{time_str},{today_date}\n")
            print(f"Attendance marked for {name} at {time_str}!")

# --- CAMERA LOOP ---
cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    # Flip the camera (1 for horizontal mirror)
    frame = cv2.flip(frame, 1)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        
        # Predict who the face belongs to
        id_, conf = recognizer.predict(roi_gray)
        
        # Check if the confidence score is good (Lower is better in OpenCV)
        # You may need to tweak these numbers (45 and 85) based on your room's lighting
        if conf >= 25 and conf <= 75:
            name = labels[id_]
            mark_attendance(name) # Trigger the attendance sheet
        else:
            name = "Unknown"
            
        # Draw the name text above the face
        font = cv2.FONT_HERSHEY_SIMPLEX
        color = (255, 255, 255) # White text
        stroke = 2
        cv2.putText(frame, name, (x, y - 10), font, 1, color, stroke, cv2.LINE_AA)
        
        # Draw a rectangle around the face
        color_box = (255, 0, 0) # Blue box (BGR format)
        stroke_box = 2
        cv2.rectangle(frame, (x, y), (x + w, y + h), color_box, stroke_box)

    # Display the resulting frame
    cv2.imshow('Face Recognition', frame)

    # Press 'q' to quit
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# Release the camera and close windows when done
cap.release()
cv2.destroyAllWindows()