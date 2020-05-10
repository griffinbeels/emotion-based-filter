# emotion-based-filter
Emotion based filtering, intended for use with Snapchat.  The example used with our model is the classic "dog face" filter in Snapchat.  We detect the agent's emotion, and then adjust the filter contents (e.g., sad ears or happy ears) based on the classified emotion.

# Credit
For face detection, we used OpenCV and Haar Feature-based Cascade Classifiers in order to effiicently process live video.

For emotion detection, we used the FER-2013 dataset, [downloadable here](https://drive.google.com/file/d/1X60B-uR3NtqPd4oosdotpbDgy8KOfUdr/view), and drew inspiration from [atulapra on Github](https://github.com/atulapra/Emotion-detection) for his implementation of the "Emotion Recognition using Deep Convolutional Neural Networks" research paper (provided in our repo as `/emotion-based-filter/emotion_detection_research.pdf`), which we also read, analyzed, and implemented.

# Installation
## Virtual Environment
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

## VSCode Setup on MacOS
If you're running this in VSCode, and are on MacOS, you might need to take an extra step to get everything going.
* In an active VSCode window, press `Command+Shift+P` which will open a search bar for VSCode related settings.
* Type `shell command` and then click on the option that says `Shell Command: Install 'code' command in PATH`.
* Close VSCode completely (right click and quit)
* Open a fresh terminal and type `sudo code`
* Cd into the `/emotion-based-filter/` directory and then run the virtual environment

At this point you should be ready to run the files!  See the next section for how to run everything.  You needed to run VSCode as root because otherwise the privacy permissions will be denied, and the video capture will always fail.  Running as root lets you enable webcam capture!

## FER-2013 Dataset
The "Facial Expression Recogition 2013" or "FER-2013" dataset is [downloadable here](https://drive.google.com/file/d/1X60B-uR3NtqPd4oosdotpbDgy8KOfUdr/view).
Once downloaded, copy the contents of the folder to the root `/emotion-based-filter/` folder.  Afterwards, we will now have a `/emotion-based-filter/data` folder containing a `test` and `train` folder for use with the Convolutional Neural Network (CNN) for emotion detection.

# How to run
```
python3.7 py/face_detection.py
```
For now to run tut.py you need to pip install dlib in the old venv and update the opencv in the venv.
Then download the 68 point predictor for faces from here https://github.com/davisking/dlib-models,
and run as "python tut.py -predictor xx68xx.dat"
