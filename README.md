# **emotion-based-filter**
## **Introduction**
Emotion based filtering in real-time.  The example used with our model is the classic "dog face" filter in Snapchat.  We first detect the agent's facial region, reformat and slice out the region defining their face, process the facial features using a pretrained Convolutional Neural Net (CNN) for emotions,  and then adjust the filter contents (e.g., happy ears, neutral ears, sad ears) based on the softmax result of the classifier.  In the ideal world, this would be used for Snapchat, but as it currently is written, it supports your webcam!  
**TL:DR:** Face detection -> emotion detection -> responsive dog ears! 

## **Credit**
For face detection, we used OpenCV and Haar Feature-based Cascade Classifiers in order to effiicently process live video.  A tutorial directly from OpenCV is located [here](https://docs.opencv.org/master/db/d28/tutorial_cascade_classifier.html), which we followed.

For emotion detection, we used the FER-2013 dataset, [downloadable here](https://drive.google.com/file/d/1X60B-uR3NtqPd4oosdotpbDgy8KOfUdr/view), which was cleaned and standardized by [atulapra on Github](https://github.com/atulapra/Emotion-detection); we also drew inspiration from his implementation of the "Emotion Recognition using Deep Convolutional Neural Networks" research paper (provided in our repo as `/emotion-based-filter/emotion_detection_research.pdf`), which we also read, analyzed, and implemented.  The FER-2013 dataset then trained our Convolutional Neural Net (CNN) using Tensorflow, and allowed us to classify a provided face for emotion!

## **Installation**
### **FER-2013 Dataset**
The "Facial Expression Recogition 2013" or "FER-2013" dataset is [downloadable here](https://drive.google.com/file/d/1X60B-uR3NtqPd4oosdotpbDgy8KOfUdr/view).
Once downloaded, copy the contents of the folder to the root `/emotion-based-filter/` folder.  Afterwards, we will now have a `/emotion-based-filter/data` folder containing a `test` and `train` folder for use to train the CNN for emotion detection.  
It is also possible that this repo actually *already* contains the `data/test` and `data/train` folders, in which case you should ignore this subsection.
Additionally, we include a model that *we did not train* from [priya-dwivedi](https://github.com/priya-dwivedi/face_and_emotion_detection), who has made their trained model public!  Check the [How to Run section](# How to Run) for details on how to swap between our model and Priya's model.

### **Virtual Environment**
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
(note: (emotion_venv) is included in the above lines to indicate you should have called the source command above to enter the virtual environment before installing; do not type (emotion_venv))

### **VSCode Setup on MacOS**
If you're running this in VSCode, and are on MacOS, you might need to take an extra step to get everything going.
* In an active VSCode window, press `Command+Shift+P` which will open a search bar for VSCode related settings.
* Type `shell command` and then click on the option that says `Shell Command: Install 'code' command in PATH`.
* Close VSCode completely (right click and quit)
* Open a fresh terminal and type `sudo code` -- this will open a fresh instance of VSCode running as the `root` user.
* Cd into the `/emotion-based-filter/` directory (or open it in VSCode) and then run the virtual environment as detailed above

At this point you should be ready to run the files!  See the next section for how to run everything.  You needed to run VSCode as root because otherwise the privacy permissions will be denied, and the video capture will always fail.  Running as root lets you enable webcam capture!

## How to Run
To run without any changes (i.e., open your webcam, load the pretrained weights, detect emotion on each frame), simply cd to this repo, and then run:

```
python3.7 py/main.py
```

The only parameters you might want to change are:

* `--model_mode`: entering "train" will train the CNN using the provided FER-2013 dataset over 50 epochs before running your webcam; entering nothing, or explicitly entering "display" will simply run the process described at the beginning of this section.

* `--camera`: by default, this is `0` which represents your built in webcam.  If you would like to use an external webcam, then change this number until it works.

If you wish to use a different model, simply change `MODEL_DATA_FILE_PATH` in `/emotion-based-filter/py/hyperparameters.py` to the file path representing your model of choice.  This may cause issues in our code, so do this at your own risk.

## **Results**
Our results are outlined in a formal paper included in our repo as `/emotion-based-filter/report.pdf`, including plenty of pictures!  
In the ideal world, this would work directly on Snapchat, but we limited our scope to simply run on live webcam footage.  We also ran into issues relating to Snapchat, because many of the APIs available are in Javascript, which means our access to Computer Vision related libraries is far more limited.