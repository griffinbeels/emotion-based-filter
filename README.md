# emotion-based-filter
Emotion based filtering, intended for use with Snapchat.  The example used with our model is the classic "dog face" filter in Snapchat.  We detect the agent's emotion, and then adjust the filter contents (e.g., sad ears or happy ears) based on the classified emotion.

# Installation
In order to easily run all of the files, and make sure that all dependencies are satsified, it is easiest to use a virtual environment.
To install the virtualenv module, run the following in a new terminal session (make sure that no other virtual environments are active):

```
pip3 install virtualenv
```

Next, navigate to a directory of your choice (perhaps the parent directory to this repo) and run:

```
virtualenv --python=python3.7 emotion_venv
```

If it wasn't clear from the above command, you should be using at least python3.7, [downloadable here](https://www.python.org/downloads/). python3.7 is the path to the Python executable that you wish to use in your virtual environment. The virtualenv command above creates a new directory called "emotion_venv" that contains your new virtual environment (env). You can activate this using the following command (in the directory that contains the emotion_venv folder):

```
source ./emotion_venv/bin/activate
```

Next install the necessary packages to your virtual environment by running these two commands (assuming you're in the folder containing this repo):

```
(emotion_venv) cd emotion-based-filter
(emotion_venv) pip3 install -r requirements.txt
```
(note: (emotion_venv) is included to indicate you should have called the source command above to enter the virtual environment before installing)

At this point you should be ready to run the files!  See the next section for how to run everything.

# How to run
```
python3.7 py/face_detection.py
```
For now to run tut.py you need to pip install dlib in the old venv and update the opencv in the venv.
Then download the 68 point predictor for faces from here https://github.com/davisking/dlib-models,
and run as "python tut.py -predictor xx68xx.dat"
