from __future__ import print_function
import hyperparameters as hp
from hyperparameters import Emotion, EMOTIONS, INPUT_DIM
from emotion import classify_emotion
import cv2 as cv
import math
import numpy as np

def detect_and_display(frame, face_cascade, eyes_cascade, model):
    """
    Given a frame, we 
    """
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame_gray = cv.equalizeHist(frame_gray)
    #-- Detect faces
    faces = face_cascade.detectMultiScale(frame_gray)
    # Set some face resc cut off for no weird jank.
    face_radius = 100
    
    # print(s_img.shape)
    eye_list = np.zeros((2,2), dtype=int)
    face = False
    if len(faces) > 0: # at least one face is detected, use the first one! TODO: add support for more than one face in frame 
        (x, y, w, h) = faces[0]
        emotion = classify_emotion(frame_gray, faces[0], model)
        print(EMOTIONS[emotion])
    # for (x,y,w,h) in faces:
        face = True
        center = (x + w//2, y + h//2)        
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
            # Determine which ear combo should be used 
            (left_ear_img, right_ear_img) = choose_filter(emotion)

            nose = cv.imread("img/nose.png", -1)
            y1, y2 = eye_list[0][1]-left_ear_img.shape[0], eye_list[0][1] # TODO: error here?
            x1, x2 = eye_list[0][0]-left_ear_img.shape[1], eye_list[0][0] # error here?
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
                frame[y1:y2, x1:x2, c] = (alpha_s * left_ear_img[:, :, c]) + (alpha_l * frame[y1:y2, x1:x2, c])
                frame[y11:y21, x11:x21, c] =  (alpha_s1 * right_ear_img[:, :, c] + alpha_l1 * frame[y11:y21, x11:x21, c])
                frame[y12:y22, x12:x22, c] =  (alpha_s2 * nose[:, :, c] + alpha_l2 * frame[y12:y22, x12:x22, c])

    cv.imshow('Capture - Face detection', frame)

def choose_filter(emotion):
    """
    Chooses the correct filter to place over the user based on their perceived emotion.
    The mappings are as follows:
        HAPPY - Happy, Surprised
        SAD - Sad, Angry, Disgusted
        NEUTRAL - Neutral, Fearful 
    Fearful is included under the neutral filter because it seems that our dataset overfit for Fearful -- 
    for this reason, misclassifications will be sent to neutral if they are fearful.
    """
    # TODO: Change the scaling of each image (as each is slightly differently sized & oriented)
    if emotion == Emotion.happy or emotion == Emotion.surprised:
        left_ear_img = cv.imread("img/ear_happy_left.png", -1)
        right_ear_img = cv.imread("img/ear_happy_right.png", -1)
    elif emotion == Emotion.sad or emotion == Emotion.angry or emotion == Emotion.disgusted:
        left_ear_img = cv.imread("img/ear_sad_left.png", -1)
        right_ear_img = cv.imread("img/ear_sad_right.png", -1)
    else: # Fearful or Neutral 
        left_ear_img = cv.imread("img/ear_neutral_left.png", -1)
        right_ear_img = cv.imread("img/ear_neutral_right.png", -1)
    return (left_ear_img, right_ear_img)