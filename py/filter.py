from __future__ import print_function
import cv2 as cv
from skimage import io
import argparse
import math
import numpy as np

def detectAndDisplay(frame, face_cascade, eyes_cascade):
    """
    Given a frame, we 
    """
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame_gray = cv.equalizeHist(frame_gray)
    #-- Detect faces
    faces = face_cascade.detectMultiScale(frame_gray)
    # Set some face resc cut off for no weird jank.
    face_radius = 100
    # Classify emotion (single face TODO expand to all faces)
    # and apply filter to frame
    # if faces[0] != None:
    #     emotion = classifyEmotion(faces[0])
    #     cv.imshow('emotion-based-filter', applyFilterToFrame)
    # s_img = cv.imread("img/dog_emote_raw.jpg")
    # s_img = cv.resize(s_img, (320, 240))
    
    # print(s_img.shape)
    eye_list = np.zeros((2,2), dtype=int)
    face = False
    for (x,y,w,h) in faces:
        # print(frame.shape)
        face = True
        center = (x + w//2, y + h//2)
        # frame = cv.ellipse(frame, center, (w//2, h//2), 0, 0, 360, (255, 0, 255), 4)
        faceROI = frame_gray[y:y+h,x:x+w]
        # radius = int(round((w + h)*0.25))
        
        x_offset2=center[0]
        y_offset2=center[1]
        radius = int(round((w + h)*0.25))
        eye_list[0] = (center[0] - radius/2, center[1]-radius/2)
        eye_list[1] = (center[0] + radius/2, center[1]-radius/2)
        if (w < face_radius) or(h < face_radius):
            face = False
        #-- In each face, detect eyes
        # eyes = eyes_cascade.detectMultiScale(faceROI)

        # for (x2,y2,w2,h2) in eyes:
        #     eye_center = (x + x2 + w2//2, y + y2 + h2//2)
        #     radius = int(round((w2 + h2)*0.25))
        #     frame = cv.circle(frame, eye_center, radius, (255, 0, 0 ), 4)

    if face and (x_offset2 < frame.shape[0]):
        left_ear_img = cv.imread("img/ear_happy_left.png", -1)
        right_ear_img = cv.imread("img/ear_happy_right.png", -1)
        nose = cv.imread("img/nose.png", -1)
        y1, y2 = eye_list[0][1]-left_ear_img.shape[0], eye_list[0][1]
        x1, x2 = eye_list[0][0]-left_ear_img.shape[1], eye_list[0][0] 
        y11, y21 = eye_list[1][1]-right_ear_img.shape[0], eye_list[1][1]
        x11, x21 = eye_list[1][0], eye_list[1][0]+right_ear_img.shape[1]
        y12, y22 = y_offset2-math.ceil(nose.shape[0]/2), y_offset2+math.floor(nose.shape[0]/2)
        x12, x22 = x_offset2-math.ceil(nose.shape[1]/2), x_offset2+math.floor(nose.shape[1]/2)
        alpha_s = left_ear_img[:, :, 3] / 255.0
        alpha_s1 = right_ear_img[:, :, 3] / 255.0
        alpha_s2 = nose[:, :, 3] / 255.0
        alpha_l = 1.0 - alpha_s
        alpha_l1 = 1.0 - alpha_s1
        alpha_l2 = 1.0 - alpha_s2
        for c in range(0, 3):
            frame[y1:y2, x1:x2, c] = (alpha_s * left_ear_img[:, :, c] + alpha_l * frame[y1:y2, x1:x2, c])
            frame[y11:y21, x11:x21, c] =  (alpha_s1 * right_ear_img[:, :, c] + alpha_l1 * frame[y11:y21, x11:x21, c])
            frame[y12:y22, x12:x22, c] =  (alpha_s2 * nose[:, :, c] + alpha_l2 * frame[y12:y22, x12:x22, c])

    cv.imshow('Capture - Face detection', frame)


def classifyEmotion(face):
    """
    For some face input, normalized to 48x48, we will determine the emotion
    """
    return None 

def applyFilterToFrame(face, frame, emotion):
    """
    For some emotion, desired face, and frame to apply it to, we swap the filter applied
    """
    return None

def main():
    parser = argparse.ArgumentParser(description='Code for Cascade Classifier tutorial.')
    parser.add_argument('--face_cascade', help='Path to face cascade.', default='ocv/haarcascade_frontalface_alt.xml')
    parser.add_argument('--eyes_cascade', help='Path to eyes cascade.', default='ocv/haarcascade_eye_tree_eyeglasses.xml')
    parser.add_argument('--camera', help='Camera divide number.', type=int, default=0)
    args = parser.parse_args()
    face_cascade_name = args.face_cascade
    eyes_cascade_name = args.eyes_cascade
    face_cascade = cv.CascadeClassifier()
    eyes_cascade = cv.CascadeClassifier()
    #-- 1. Load the cascades
    if not face_cascade.load(cv.samples.findFile(face_cascade_name)):
        print('--(!)Error loading face cascade')
        exit(0)
    if not eyes_cascade.load(cv.samples.findFile(eyes_cascade_name)):
        print('--(!)Error loading eyes cascade')
        exit(0)
    camera_device = args.camera
    #-- 2. Read the video stream
    cap = cv.VideoCapture(camera_device)
    if not cap.isOpened:
        print('--(!)Error opening video capture')
        exit(0)
    while True:
        ret, frame = cap.read()
        if frame is None:
            print('--(!) No captured frame -- Break!')
            break
        detectAndDisplay(frame, face_cascade, eyes_cascade)
        if cv.waitKey(10) == 27:
            cv.VideoCapture(camera_device).release()
            break

if __name__ == "__main__":
    main()