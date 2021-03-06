from __future__ import print_function
from imutils.object_detection import non_max_suppression
import numpy as np
import imutils
import cv2
import threading
from tkinter import *

padding = 8
total_num = 0
duplicate_num = 0
zoom_index = 1


def num_update(face_num, body_num, total_num):
    root = Tk()
    root.title("Detection Results")
    root.resizable(0, 0)

    face_lb = Label(root, text="Number of faces detected:")
    num_face_lb = Label(root, text=str(face_num))
    body_lb = Label(root, text="Number of bodies detected:")
    num_body_lb = Label(root, text=str(body_num))
    bt_close = Button(root, text="Close", command=root.destroy)
    tot_lb = Label(root, text="The total number of human:")
    tot_num_lb = Label(root, text=str(total_num))

    num_face_lb.config(font=("",25))
    num_body_lb.config(font=("",25))
    tot_num_lb.config(font=("", 25))

    face_lb.grid(row=0, column=0, padx=padding, pady=padding)
    num_face_lb.grid(row=0, column=1, padx=padding, pady=padding)
    body_lb.grid(row=1, column=0, padx=padding, pady=padding)
    num_body_lb.grid(row=1, column=1, padx=padding, pady=padding)
    tot_lb.grid(row=2, column=0, padx=padding, pady=padding)
    tot_num_lb.grid(row=2, column=1, padx=padding, pady=padding)
    bt_close.grid(row=3, column=2, padx=padding, pady=padding)

    total_num = 0

    root.mainloop()
    return


face_cascade = cv2.CascadeClassifier('C:\OpenCV-3.3.0\opencv\sources\data\haarcascades\haarcascade_frontalface_default.xml')

# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# loop over the image paths
imagePath = sys.argv[1]

image = cv2.imread(imagePath)

faces = face_cascade.detectMultiScale(image, 1.3, 5)
for (x,y,w,h) in faces:
    cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)

img_w, img_h, img_channel = image.shape
orig = image.copy()
image = imutils.resize(image, width=min(400, image.shape[1]))

zoom_index = img_w/min(400, image.shape[1])

# detect people in the image
(rects, weights) = hog.detectMultiScale(image, winStride=(4, 4),
    padding=(8, 8), scale=1.05)

# apply non-maxima suppression to the bounding boxes using a
# fairly large overlap threshold to try to maintain overlapping
# boxes that are still people
rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

# draw the final bounding boxes
for (xA, yA, xB, yB) in pick:
    cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)


for (xb1, yb1, xb2, yb2) in pick:
    # print(xb1, yb1, xb2, yb2)
    for (xf, yf, wf, hf) in faces:
        mid_x = xf + (wf/2)
        mid_y = yf + (hf/2)
        #print(mid_x, mid_y)
        mid_x /= zoom_index
        mid_y /= zoom_index
        # print(mid_x, mid_y)
        if (xb1<mid_x and mid_x<xb2 and yb1<mid_y and mid_y<yb2):
            duplicate_num=duplicate_num+1
        #print(duplicate_num)

image = imutils.resize(image, width=min(orig.shape[1], 800))

threads = []
t = threading.Thread(target=num_update, args=(len(faces), len(pick), len(faces)+len(pick)-duplicate_num))
threads.append(t)
t.start()

cv2.imshow("Image", image)
c = cv2.waitKey(0)
