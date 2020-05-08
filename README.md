# emotion-based-filter
Emotion based filtering, intended for use with Snapchat.  The example used with our model is the classic "dog face" filter in Snapchat.  We detect the agent's emotion, and then adjust the filter contents (e.g., sad ears or happy ears) based on the classified emotion.

For now to run tut.py you need to pip install dlib in the old venv and update the opencv in the venv.
Then download the 68 point predictor for faces from here https://github.com/davisking/dlib-models,
and run as "python tut.py -predictor xx68xx.dat"
