from __future__ import print_function
from hyperparameters import Emotion, TRAINING_DIRECTORY, TESTING_DIRECTORY, MODEL_DATA_FILE_PATH, EMOTIONS, INPUT_DIM, BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE
import cv2 as cv
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

def classify_emotion(frame_gray, face, model):
    """
    Classifies a face into one of seven Emotion categories (see hyperparameters.py) 
    based on the face's features.

    Input:
        frame_gray :: array 
            - desc: original camera frame after grayscale filter applied
        face :: tuple 
            - desc: the x,y position (top left) of the desired face to classify,
                    as well as its w,h (width, height).  Defines the face's bounding
                    box
        model :: Sequential Tensorflow Model
            - desc: model trained to classify Emotion

    Output: 
        -the softmax result of the model's prediction cast to an Emotion.
    """
    # slice out the face region
    (x, y, w, h) = face
    face_gray_from_frame = frame_gray[y:y+h,x:x+w]

    #paper called for standard 48x48 sizing of any input 
    resized = np.expand_dims(np.expand_dims(cv.resize(face_gray_from_frame, (INPUT_DIM, INPUT_DIM)), -1), 0)

    # Predict & return classification
    prediction = model.predict(resized)
    return Emotion(np.argmax(prediction))

def get_num_images_in_dir(directory):
    """
    Parses the provided directory to determine the total number of files in the directory.
    This assumes that the directory only contains images to be used in training / testing.

    Input:
        directory :: string
            - desc: path to the directory to parse; should be either the training or testing directory.
    
    Output:
        -the number of files contained in the provided directory, including inside each emotion folders.
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
    Sets up the model used for emotion classification, by either training a new model,
    or loading a pre-existing model.  
    
    Input: 
        mode :: string
            -desc: If mode "train" was selected, then this will train a new model and
            store the results in "model.h5"; if mode "display" was selected (or no args were entered),
            then this will return the stored "model.h5" loaded into the model.
    
    Output:
        - the trained or loaded sequential model
    """
    # Tally the number of images in both directories 
    num_training = get_num_images_in_dir(TRAINING_DIRECTORY)
    num_testing = get_num_images_in_dir(TESTING_DIRECTORY)

    # Create the model
    #2 convolutional layers followed by max pooling & dropout, then 2x(C2D -> MaxPooling)
    #then another dropout, followed by flatten, dense (relu), dropout, dense (softmax)
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
        #model = load_model(MODEL_DATA_FILE_PATH) # UNCOMMENT IF YOU WANT TO USE THE ALT MODEL
    else:
        print("ERROR: Invalid mode please enter one of [train, display]")
        return None

    return model