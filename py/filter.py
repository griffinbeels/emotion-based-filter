from __future__ import print_function
import cv2 as cv
from skimage import io
import argparse
import math
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# This file is run from the /py directory -- therefore, we go up a directory
TRAINING_DIRECTORY = "data/train"
TESTING_DIRECTORY = "data/test"
MODEL_DATA_FILE_PATH = "data/model.h5"

# Maps numbers [0, 6] to each emotion, for use in processing the model classification 
EMOTIONS = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# Research paper (in root directory) calls for all input to be 48x48
INPUT_DIM = 48

# Training hyperparameters
BATCH_SIZE = 64
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4

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
    if len(faces) > 0:
        emotion = classify_emotion(frame_gray, faces[0], model)
        print(EMOTIONS[emotion])

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


def classify_emotion(frame_gray, face, model):
    """
    For some face input, normalized to 48x48, we will determine the emotion
    """
    # slice out the face region
    (x, y, w, h) = face
    face_gray_from_frame = frame_gray[y:y+h,x:x+w]

    #paper called for standard 48x48 sizing of any input 
    resized = np.expand_dims(np.expand_dims(cv.resize(face_gray_from_frame, (INPUT_DIM, INPUT_DIM)), -1), 0)

    # Predict & return classification
    prediction = model.predict(resized)
    return int(np.argmax(prediction))

def apply_filter_to_frame(face, frame, emotion):
    """
    For some emotion, desired face, and frame to apply it to, we swap the filter applied
    """
    return None

def get_num_images_in_dir(directory):
    """
    Parses the provided directory to determine the number of images in the directory that 
    match the provided emotions.
    """
    # Get list of all images in training directory
    categories = ["angry", "disgusted", "fearful", "happy", "neutral", "sad", "surprised"]
    num_files = 0
    for root, _, files in os.walk(directory, categories):
        for name in files:
            num_files += 1
    return num_files

def train_or_load_model(mode):
    """
    Sets up the model used for emotion classification.  If mode "train" was selected,
    then this will train a new model and store the results in "model.h5"; if mode "display"
    was selected (or no args were entered), then this will return the stored "model.h5" 
    loaded into the model.
    """
    # Tally the number of images in both directories 
    num_training = get_num_images_in_dir(TRAINING_DIRECTORY)
    num_testing = get_num_images_in_dir(TESTING_DIRECTORY)

    # Create the model
    # 2 convolutional layers followed by max pooling & dropout, then 2x(C2D -> MaxPooling)
    # then another dropout, followed by flatten, dense (relu), dropout, dense (softmax)
    model = Sequential()

    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(INPUT_DIM,INPUT_DIM,1)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(7, activation='softmax'))

    # switch functionality based on passed in mode
    if mode == "train":
        # Training / testing data generation
        training = ImageDataGenerator(rescale=1./255)
        train_generator = training.flow_from_directory(
            TRAINING_DIRECTORY,
            target_size=(INPUT_DIM,INPUT_DIM),
            batch_size=BATCH_SIZE,
            color_mode="grayscale",
            class_mode='categorical')

        testing = ImageDataGenerator(rescale=1./255)
        validation_generator = testing.flow_from_directory(
            TESTING_DIRECTORY,
            target_size=(INPUT_DIM,INPUT_DIM),
            batch_size=BATCH_SIZE,
            color_mode="grayscale",
            class_mode='categorical')

        # Compile our model & save results in "../data/model.h5"
        model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.0001, decay=LEARNING_RATE),metrics=['accuracy'])
        model_info = model.fit_generator(
                train_generator,
                steps_per_epoch=num_training // BATCH_SIZE,
                epochs=NUM_EPOCHS,
                validation_data=validation_generator,
                validation_steps=num_testing // BATCH_SIZE)
        model.save_weights(MODEL_DATA_FILE_PATH)
    # display live, so simply load the weights and return
    elif mode == "display":
        model.load_weights(MODEL_DATA_FILE_PATH)
    else:
        print("ERROR: Invalid mode please enter one of [train, display]")
        return None

    return model

def parse_input_and_load():
    """
    Helper method to parse all input & do any intial loading of data.

    output:
        -tuple containing (mode, face_cascade, eyes_cascade, camera_device)
    """
    #-- argument setup
    parser = argparse.ArgumentParser(description='Code for Emotion Based Filtering.')
    parser.add_argument('--face_cascade', help='Path to face cascade.', default='ocv/haarcascade_frontalface_alt.xml')
    parser.add_argument('--eyes_cascade', help='Path to eyes cascade.', default='ocv/haarcascade_eye_tree_eyeglasses.xml')
    parser.add_argument("--model_mode",help='Whether or not we want to [train] the model, or [display] results.', default='display')
    parser.add_argument('--camera', help='Camera divide number.', type=int, default=0)

    #-- store each input to be returned
    args = parser.parse_args()
    mode = args.model_mode
    face_cascade = cv.CascadeClassifier()
    eyes_cascade = cv.CascadeClassifier()
    camera_device = args.camera

    #-- load the cascades
    if not face_cascade.load(cv.samples.findFile(args.face_cascade)):
        print('--(!)Error loading face cascade')
        exit(0)
    if not eyes_cascade.load(cv.samples.findFile(args.eyes_cascade)):
        print('--(!)Error loading eyes cascade')
        exit(0)
    
    return (mode, face_cascade, eyes_cascade, camera_device)

def main():
    #-- parse the arguments passed into this py file in prep for the main read loop
    (mode, face_cascade, eyes_cascade, camera_device) = parse_input_and_load()

    #-- populate the emotion classification model (if preloaded, simply load, else train).
    model = train_or_load_model(mode)

    #-- read the video stream
    cap = cv.VideoCapture(camera_device)
    if not cap.isOpened:
        print('--(!)Error opening video capture')
        exit(0)
    while True:
        # read the current frame and verify successful capture
        ret, frame = cap.read()
        if frame is None:
            print('--(!) No captured frame -- Break!')
            break

        # if valid, we process the frame -- i.e., detect face + emotion + apply filter
        detect_and_display(frame, face_cascade, eyes_cascade, model)

        # corresponds to ESC key
        if cv.waitKey(10) == 27:
            cv.VideoCapture(camera_device).release()
            break

if __name__ == "__main__":
    main()