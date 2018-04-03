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
isVideo = FALSE
windows_title = "Footage"
duplicate_num = 0

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
    bt_close = Button(root, text="Close", command=root.destroy)  # Creating the "close" button
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

# loop over the image paths
filepath = sys.argv[1]
extension = filepath.split(".")[-1]

# determine if the file is a video
if (extension == 'avi' or 'mp4'):
    isVideo = TRUE
    windows_title = "Footage"
    cap = cv2.VideoCapture(filepath)  # Class for video capturing from video files, image sequences or cameras.

while(cap.isOpened() and isVideo):  # While we succeeded and it's video

    ret, frame = cap.read()
    # Get all of faces and store into a list named "faces" below.
    faces = face_cascade.detectMultiScale(frame, 1.3, 5)
    # Draw all of blue boxes for face detection
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    orig = frame.copy()  # Copy the original video capture

    # detect people in the image
    (rects, weights) = hog.detectMultiScale(frame, winStride=(4, 4),
                                            padding=(8, 8), scale=1.05)

    # draw the original bounding boxes
    for (x, y, w, h) in rects:
        cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # apply non-maxima suppression to the bounding boxes using a
    # fairly large overlap threshold to try to maintain overlapping
    # boxes that are still people
    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

    # draw the final bounding boxes
    for (xA, yA, xB, yB) in pick:
        cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)

    for (xb1, yb1, xb2, yb2) in pick:
        # print(xb1, yb1, xb2, yb2)
        for (xf, yf, wf, hf) in faces:
            mid_x = xf + (wf / 2)
            mid_y = yf + (hf / 2)
            # print(mid_x, mid_y)
            if (xb1 < mid_x and mid_x < xb2 and yb1 < mid_y and mid_y < yb2):
                duplicate_num = duplicate_num + 1
                # print(duplicate_num)

    cv2.putText(frame, str(len(faces)+len(pick)-duplicate_num), (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)

    cv2.imshow(windows_title, frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
