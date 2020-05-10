from __future__ import print_function
import argparse
import cv2 as cv
from hyperparameters import HAAR_EYE_FILE_PATH, HAAR_FACE_FILE_PATH, CAMERA_IDX
from emotion import train_or_load_model
from filter import Filter

def parse_input_and_load():
    """
    Parses all input & does any intial loading of data, including error checks.

    Input: 
        None

    Output: 
        a tuple containing (mode, face_cascade, eyes_cascade, camera_device):
            -mode is either ["train" or "display"]
            -face_cascade is the loaded cascade file from OpenCV for faces
            -eye_cascade is the loaded cascade file from OpenCV for eyes
            -camera_device is the index of the loaded camera
    """
    #-- argument setup
    parser = argparse.ArgumentParser(description='Code for Emotion Based Filtering.')
    parser.add_argument('--face_cascade', help='Path to face cascade.', default=HAAR_FACE_FILE_PATH)
    parser.add_argument('--eyes_cascade', help='Path to eyes cascade.', default=HAAR_EYE_FILE_PATH)
    parser.add_argument("--model_mode",help='Whether or not we want to [train] the model, or [display] results.', default='display')
    parser.add_argument('--camera', help='Camera divide number.', type=int, default=CAMERA_IDX)

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
    """ 
    Parses input, loads or trains the CNN, and then begins a read loop on the 
    user's webcam until stopped by the ESC key, in order to perform face & emotion
    detection.
    """
    #-- parse the arguments passed into this py file in prep for the main read loop
    (mode, face_cascade, eyes_cascade, camera_device) = parse_input_and_load()

    #-- populate the emotion classification model (if preloaded, simply load, else train).
    model = train_or_load_model(mode)

    #-- read the video stream
    cap = cv.VideoCapture(camera_device)
    if not cap.isOpened:
        print('--(!)Error opening video capture')
        exit(0)
    
    # Generate a filter object to maintain state
    my_filter = Filter()

    while True:
        # read the current frame and verify successful capture
        ret, frame = cap.read()
        if frame is None:
            print('--(!) No captured frame -- Break!')
            break

        # if valid, we process the frame -- i.e., detect face + emotion + apply filter
        my_filter.detect_and_display(frame, face_cascade, eyes_cascade, model)

        # corresponds to ESC key
        if cv.waitKey(10) == 27:
            cv.VideoCapture(camera_device).release()
            break

if __name__ == "__main__":
    main()