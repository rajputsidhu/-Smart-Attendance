import face_recognition
import cv2
import numpy as np
import csv
from datetime import datetime
import os
import tkinter as tk
from tkinter import messagebox
from tkinter import filedialog

# Initialize global variables
video_capture = None
f = None
known_face_encodings = []
known_face_names = []
students = []
current_date = ""
now = None

# Function to start the face recognition process
def start_face_recognition():
    global video_capture, f, known_face_encodings, known_face_names, students, current_date, now
    
    video_capture = cv2.VideoCapture(0)
    
    if not video_capture.isOpened():
        messagebox.showerror("Error", "Could not open camera.")
        return
    
    # Load and encode known images
    try:
        load_known_faces()
    except (IndexError, FileNotFoundError) as e:
        messagebox.showerror("Error", str(e))
        return
    
    now = datetime.now()
    current_date = now.strftime("%Y-%m-%d")
    
    try:
        f = open(current_date + '.csv', 'a+', newline='')
        global lnwriter
        lnwriter = csv.writer(f)
    except Exception as e:
        messagebox.showerror("Error", f"Could not open CSV file. {e}")
        return
    
    process_frame()

# Function to load known faces
def load_known_faces():
    global known_face_encodings, known_face_names, students
    base_image_path = "/Users/sidhusingh/Devloper/Code/LEARNING PROJECT/PHOTOS/"
    
    images = ["SIDHU.jpg", "NAMO.jpg", "YOGI.jpg", "RAGA.jpg"]
    names = ["SIDHU", "NAMO", "YOGI", "RAGA"]

    for image, name in zip(images, names):
        image_path = os.path.join(base_image_path, image)
        loaded_image = face_recognition.load_image_file(image_path)
        encoding = face_recognition.face_encodings(loaded_image)
        
        if encoding:
            known_face_encodings.append(encoding[0])
            known_face_names.append(name)
        else:
            raise IndexError(f"Could not locate face in {image}.")
    
    students = known_face_names.copy()

# Function to process video frames for face recognition
def process_frame():
    global video_capture, now  # Ensure 'now' is used correctly in this function
    
    ret, frame = video_capture.read()
    
    if not ret:
        messagebox.showerror("Error", "Could not read frame from camera.")
        return
    
    # Resize the frame for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Find face locations
    face_locations = face_recognition.face_locations(rgb_small_frame)
    
    # Initialize face_names as an empty list in case no faces are found
    face_names = []

    # Only process if faces are found
    if face_locations:
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)

            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

            if name in students:
                students.remove(name)
                current_time = now.strftime("%H-%M-%S")
                lnwriter.writerow([name, current_time])
    
    # Draw rectangles around faces and show video feed
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
        cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)

    cv2.imshow('Face Recognition Attendance', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        stop_face_recognition()
        return

    # Call the process_frame function again to keep the loop going
    root.after(10, process_frame)

# Function to stop face recognition and release resources
def stop_face_recognition():
    global video_capture, f
    
    if video_capture is not None:
        video_capture.release()
    
    cv2.destroyAllWindows()
    
    if f is not None:
        f.close()
    
    messagebox.showinfo("Info", "Face recognition stopped and attendance saved.")

# Function to handle window closing event
def on_closing():
    if messagebox.askokcancel("Quit", "Do you want to quit?"):
        stop_face_recognition()
        root.destroy()

# Setting up the Tkinter GUI
root = tk.Tk()
root.title("Face Recognition Attendance System")
root.geometry("400x300")
root.protocol("WM_DELETE_WINDOW", on_closing)

# Creating buttons to start and stop the face recognition process
start_button = tk.Button(root, text="Start Face Recognition", command=start_face_recognition, font=("Arial", 14))
start_button.pack(pady=20)

stop_button = tk.Button(root, text="Stop Face Recognition", command=stop_face_recognition, font=("Arial", 14))
stop_button.pack(pady=20)

# Run the Tkinter event loop
root.mainloop()