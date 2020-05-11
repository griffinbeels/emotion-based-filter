from __future__ import print_function
import hyperparameters as hp
from hyperparameters import Emotion, EMOTIONS, INPUT_DIM, VOTES_REQUIRED_FOR_ELECTION
from emotion import classify_emotion
import cv2 as cv
import math
import numpy as np
import operator

class Filter:
    """
    Encapsulates all relevant logic for filtering and maintains state associated with
    filter elections.
    """
    def __init__(self):
        # Stores state regarding the current election cycle.
        # Elections allow for us to VOTE for a filter rather than swap between
        # filters once a change in emotion is detected; that is, a family of emotions
        # has to be displayed long enough in order for it to be displayed.
        self.reset_election()

        # The previously elected left and right ear filters; by default neutral.
        self.elected_left_ear = cv.imread("img/ear_neutral_left.png", -1)
        self.elected_right_ear =cv.imread("img/ear_neutral_right.png", -1)

        # LOAD EACH IMAGE ONCE WHEN THIS OBJECT IS CREATED
        self.happy_left_img = cv.imread("img/ear_happy_left.png", -1)
        self.happy_right_img = cv.imread("img/ear_happy_right.png", -1)
        self.sad_left_img = cv.imread("img/ear_sad_left.png", -1)
        self.sad_right_img = cv.imread("img/ear_sad_right.png", -1)
        self.neutral_left_img = cv.imread("img/ear_neutral_left.png", -1)
        self.neutral_right_img = cv.imread("img/ear_neutral_right.png", -1)
        self.nose = cv.imread("img/nose.png", -1)

        # stores the last detected nose coordinates so that the image doesn't flicker
        self.nose_cache = []

    def detect_and_display(self, frame, face_cascade, eyes_cascade, nose_cascade, model):
        """
        Given a camera frame loaded from the user's webcam, detects all visible faces,
        classifies emotion for any faces found, and changes between a set of filters depending 
        on the emotion that the model classifies.

        Input:
            -frame :: RGB array
                -desc: the frame currently being processed, provided by the webcam
            -face_cascade :: loaded xml file
                -desc: the cascade xml obtained from OpenCV for use in face detection
            -eyes_cascade :: loaded xml file
                -desc: the cascade xml obtained from OpenCV for use in eye detection
            -nose_cascade :: loaded xml file
                -desc: the cascade xml obtained from OpenCV for use in nose detection
            model :: Sequential Tensorflow Model
                -desc: model trained to classify Emotion
        
        Output:
            None (displays an image after each call using cv.imshow)
        """
        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        frame_gray = cv.equalizeHist(frame_gray)
        #-- Detect faces
        # Finds the biggest object instead (previous found all faces) and mandates a minimum size
        faces = face_cascade.detectMultiScale(image=frame_gray, flags=cv.CASCADE_FIND_BIGGEST_OBJECT, minSize=(30, 30))
        # Set some face resc cut off for no weird jank.
        face_radius = 100
        
        # print(s_img.shape)
        eye_list = np.zeros((2,2), dtype=int)
        face = False
        if len(faces) > 0: # at least one face is detected, use the first one! TODO: add support for more than one face in frame

            (x, y, w, h) = faces[0]
            
            # Classify the emotion and vote in the current filter election
            emotion = classify_emotion(frame_gray, faces[0], model)
            self.process_filter_vote(emotion)
            
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

            faceROI = frame_gray[y:y+h, x:x+w]
            #-- In each face, detect eyes
            # eyes = eyes_cascade.detectMultiScale(faceROI)

            #-- In each face, detect nose (Note: coordinates are in terms of face dimensions, not whole frame)
            nose_coords = nose_cascade.detectMultiScale(faceROI)
            if len(nose_coords) > 0:
                self.nose_cache = nose_coords
            nx, ny, nw, nh = None, None, None, None

            # Check if the nose was detected or not
            is_nose = False
            if len(self.nose_cache) > 0:
                (nx, ny, nw, nh) = self.nose_cache[0]
                is_nose = True
            

            # for (x2,y2,w2,h2) in eyes:
            #     eye_center = (x + x2 + w2//2, y + y2 + h2//2)
            #     radius = int(round((w2 + h2)*0.25))
            #     frame = cv.circle(frame, eye_center, radius, (255, 0, 0 ), 4)

            if face and (x_offset2 < frame.shape[0]):
                # Determine which ear combo should be used
                self.try_update_filter()

                # how much we want to scale the original ears pictures by
                scale_factor = radius / 100

                scale_tuple = lambda t, s : (math.floor(t[1] * s), math.floor(t[0] * s))
                scaled_left_ear = cv.resize(self.elected_left_ear, scale_tuple(self.elected_left_ear.shape, scale_factor))
                scaled_right_ear = cv.resize(self.elected_right_ear, scale_tuple(self.elected_right_ear.shape, scale_factor))
                scaled_nose = cv.resize(self.nose, scale_tuple(self.nose.shape, scale_factor))


                # Position the ears / nose according to the face orientation
                y1, y2 = eye_list[0][1]-scaled_left_ear.shape[0], eye_list[0][1] # TODO: error here?
                x1, x2 = eye_list[0][0]-scaled_left_ear.shape[1], eye_list[0][0] # error here?
                y11, y21 = eye_list[1][1]-scaled_right_ear.shape[0], eye_list[1][1]
                x11, x21 = eye_list[1][0], eye_list[1][0]+scaled_right_ear.shape[1]
                if is_nose:
                    nose_offset_y = y+ny+math.ceil(nh/2)
                    nose_offset_x = x+nx+math.ceil(nw/2)
                    y12, y22 = nose_offset_y-math.ceil(scaled_nose.shape[0]/2), nose_offset_y+math.floor(scaled_nose.shape[0]/2)
                    x12, x22 =nose_offset_x-math.ceil(scaled_nose.shape[1]/2), nose_offset_x+math.floor(scaled_nose.shape[1]/2)

                alpha_s = scaled_left_ear[:, :, 3] / 255.0
                alpha_s1 = scaled_right_ear[:, :, 3] / 255.0
                alpha_s2 = scaled_nose[:, :, 3] / 255.0
                alpha_l = 1.0 - alpha_s
                alpha_l1 = 1.0 - alpha_s1
                alpha_l2 = 1.0 - alpha_s2
                for c in range(0, 3):
                    # Need to make sure all shapes are in bounds
                    if (y1 > 0 and y1 < frame.shape[0] and x1 > 0 and x1 < frame.shape[1] and
                        y2 > 0 and y2 < frame.shape[0] and x2 > 0 and x2 < frame.shape[1]):
                        frame[y1:y2, x1:x2, c] = (alpha_s * scaled_left_ear[:, :, c]) + (alpha_l * frame[y1:y2, x1:x2, c])
                    if (y11 > 0 and y11 < frame.shape[0] and x11 > 0 and x11 < frame.shape[1] and
                        y21 > 0 and y21 < frame.shape[0] and x21 > 0 and x21 < frame.shape[1]):
                        frame[y11:y21, x11:x21, c] =  (alpha_s1 * scaled_right_ear[:, :, c] + alpha_l1 * frame[y11:y21, x11:x21, c])
                    if is_nose:
                        frame[y12:y22, x12:x22, c] =  (alpha_s2 * scaled_nose[:, :, c] + alpha_l2 * frame[y12:y22, x12:x22, c])

        cv.imshow('Capture - Face detection', frame)

    def process_filter_vote(self, emotion):
        """
        Votes for the provided emotion in the current filter election period.
        Happy / Surprised --> Happy vote
        Neutral / Fearful -> Neutral vote
        Sad / Angry / Disgusted -> Sad vote

        Fearful is included under the neutral filter because it seems that our dataset overfit for Fearful -- 
        for this reason, misclassifications will be sent to neutral if they are fearful.

        Input:
            emotion :: Emotion
                -desc: the emotion to be voted for
        Output:
            None (updates the voting map & monotonically increases the number of votes in this
                election period so far)
        """
        # Update the correct vote count monotonically
        if emotion == Emotion.happy or emotion == Emotion.surprised:
            self.filter_votes[Emotion.happy] += 1
        elif emotion == Emotion.sad or emotion == Emotion.angry or emotion == Emotion.disgusted:
            self.filter_votes[Emotion.sad] += 1
        else: #Neutral or fearful
            self.filter_votes[Emotion.neutral] += 1

        self.num_election_votes += 1

    def try_update_filter(self):
        """
        Chooses the correct filter to place over the user based on the elected_emotion; that is, 
        the emotion that received the most votes based on the most recent frames. 
        Additionally, processes an election if possible; if no election, simply chooses the previously
        elected left/right ears.
        
        Output:
            - the correct left / right ears loaded from cv.imread based on the emotion votes.
        """
        # Check to see if we need to complete an election cycle
        if self.num_election_votes >= VOTES_REQUIRED_FOR_ELECTION:
            # Use operator module to choose the Emotion with the highest number of votes
            elected_emotion = max(self.filter_votes.items(), key=operator.itemgetter(1))[0]

            # TODO: Change the scaling of each image (as each is slightly differently sized & oriented)
            # Choose the appropriate preloaded image
            if elected_emotion == Emotion.happy:
                self.elected_left_ear = self.happy_left_img
                self.elected_right_ear = self.happy_right_img
            elif elected_emotion == Emotion.sad:
                self.elected_left_ear = self.sad_left_img
                self.elected_right_ear = self.sad_right_img
            else: # Fearful or Neutral 
                self.elected_left_ear = self.neutral_left_img
                self.elected_right_ear = self.neutral_right_img
            print("VOTED FOR FILTER: ", EMOTIONS[elected_emotion])
            self.reset_election()

    def reset_election(self):
        """
        Resets any election states to default, to begin a new election period.
        """
        # Map to keep track of the current number of votes for each filter
        # -- initially starts at 0, and one value monotonically increases for each frame processed,
        # depending on which emotion was detected.
        # Once enough frames are detected, (hyperparameters.py's VOTES_REQUIRED_FOR_ELECTION)
        # a vote is started, and the currently displayed filter is changed.
        self.filter_votes = {Emotion.happy: 0, Emotion.neutral: 0, Emotion.sad: 0}

        # The current number of votes currently processed in this filter election cycle;
        # 0 <= num_election_votes <= VOTES_REQUIRED_FOR_ELECTION
        self.num_election_votes = 0