from __future__ import print_function
from imutils.object_detection import non_max_suppression

# NumPy is the fundamental package for scientific computing with Python.
import numpy as np

# A series of convenience functions to make basic processing functions such as translation, rotation, resizing,
# skeletonization, displaying Matplotlib images, sorting contours, detecting edges, and much more easier with
# OpenCV and Python 2.7 and 3.
import imutils

import cv2  # Wrapper package for OpenCV python bindings.

# This module constructs higher-level threading interfaces on top of the lower level _thread module.
import threading

# The Tkinter module ("Tk interface") is the standard Python interface to the Tk GUI toolkit.
from tkinter import *

threads = []

padding = 8
windows_title = "Webcam"

# The window which shows detection number.
def result_window(facenum, bodynum, totnum):
    root = Tk()  # Initializing the interpreter and creating the root window.
    root.title("Detection Results")
    root.resizable(0, 0)  # This tells the window that it can't resized in the x or y directions.

    # Setting the content and functionality of all of labels and button.
    face_lb = Label(root, text="Number of faces detected:")
    num_face_lb = Label(root, text=str(facenum))
    body_lb = Label(root, text="Number of bodies detected:")
    num_body_lb = Label(root, text=str(bodynum))
    bt_close = Button(root, text="Close", command=root.destroy)
    tot_lb = Label(root, text="The total number of human:")
    tot_num_lb = Label(root, text=str(totnum))

    # Setting the number font size.
    num_face_lb.config(font=("",25))
    num_body_lb.config(font=("",25))
    tot_num_lb.config(font=("", 25))

    # Assigning all labels and button to their particular positions.
    face_lb.grid(row=0, column=0, padx=padding, pady=padding)
    num_face_lb.grid(row=0, column=1, padx=padding, pady=padding)
    body_lb.grid(row=1, column=0, padx=padding, pady=padding)
    num_body_lb.grid(row=1, column=1, padx=padding, pady=padding)
    tot_lb.grid(row=2, column=0, padx=padding, pady=padding)
    tot_num_lb.grid(row=2, column=1, padx=padding, pady=padding)
    bt_close.grid(row=3, column=2, padx=padding, pady=padding)

    root.mainloop()

    return

# Loading a Haar feature-based cascade classifier for object detection.
face_cascade = cv2.CascadeClassifier('C:\OpenCV-3.3.0\opencv\sources\data\haarcascades\haarcascade_frontalface_default.xml')

# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

cam = cv2.VideoCapture(0)  # Capture the video frame-by-frame.

while(cam.isOpened()):

    ret, frame = cam.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Get all of faces and store into a list named "faces" below.
    faces = face_cascade.detectMultiScale(frame, 1.05, 5)
    # Draw all of blue boxes for face detection
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Put the number of detection on the corner of the screen.
    cv2.putText(frame, str(len(faces)), (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)

    cv2.imshow(windows_title, frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cam.release()
cv2.destroyAllWindows()
