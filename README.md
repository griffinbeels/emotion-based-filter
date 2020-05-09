# emotion-based-filter
Emotion based filtering, intended for use with Snapchat.  The example used with our model is the classic "dog face" filter in Snapchat.  We detect the agent's emotion, and then adjust the filter contents (e.g., sad ears or happy ears) based on the classified emotion.

For now to run tut.py you need to pip install dlib in the old venv and update the opencv in the venv.
Then download the 68 point predictor for faces from here https://github.com/davisking/dlib-models,
and run as "python tut.py -predictor xx68xx.dat"


# Dependencies / Libraries used
The original venv environment for CV, but including:

1.) cmake https://github.com/python-cmake-buildsystem/python-cmake-buildsystem 
(pip3 install cmake)

2.) dlib https://gist.github.com/ageitgey/629d75c1baac34dfa5ca2a1928a7aeaf 
(pip3 install dlib)

3.) face recognition https://github.com/ageitgey/face_recognition
(pip3 install face_recognition)

4.) tensorflow / keras https://keras.io/ 
(pip3 install keras), (pip3 install tensorflow)