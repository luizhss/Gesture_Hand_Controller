# Authors:

* Luiz Henrique da Silva Santos - https://github.com/luizhss/ - https://www.linkedin.com/in/luizhssantos/
* Matheus Vyctor Aranda Espíndola - https://github.com/Matheus-Vyctor/ - https://www.linkedin.com/in/matheus-vyctor/

# Gesture_Hand_Controller
Our project is a navigation controller based on hand gestures. Using your webcam to capture your hand, you can move your mouse, left and right click, scroll and zoom. In general, it has a good response, however it may bug sometimes. Know that this is an alpha version.

**Note: This project was tested on Ubuntu 18.04 LTS. It also works on Windows but may not work as expected in the future.**

## How it works
We used Mediapipe Hand Solutions to get the hand landmarks predictions. Later, we generated a dataset with 1894 samples of different hand gestures and used it to train a SVC model. Putting it together with OpenCV, we read each frame from the camera, predict a possible gesture and invoke a routine to treat it. Each routine reproduce the determined action alongside the PyAutoGUI libray.

Full video demonstration (youtube):

[Video](https://youtu.be/OKRuiNP62Qc)


## Quick start - Install (Clone this repository)
    
    git clone https://github.com/luizhss/gesture_hand_controller
    
## Functions

> Mouse
* Control the mouse through the gesture detected bellow. To move the mouse, just move your 5-fingers hand and your movement will be mapped proportionally to your screen. To lock mouse for a fell seconds, just keep your hand still for a little bit and you receive a message "Position Locked".

<img src="https://github.com/luizhss/Gesture_Hand_Controller/blob/master/gifs/mouse_1.gif" width="432" height="192"/>            <img src="https://github.com/luizhss/Gesture_Hand_Controller/blob/master/gifs/mouse_2.gif" width="432" height="192"/>

> Left Click
* Left Click of the mouse through the detected gesture bellow:

![Left Click Gif](https://github.com/luizhss/Gesture_Hand_Controller/blob/master/gifs/left_click.gif)

> Right Click
* Right Click of the mouse through the detected gesture bellow:

![Right Click Gif](https://github.com/luizhss/Gesture_Hand_Controller/blob/master/gifs/right_click.gif)

> Scroll Up
* For this one, just keep the detected gesture bellow for a second. 
* Observation: in Ubuntu we suggest the value + 3 to scroll; in windows this suggested value is + 70. If you are on Windows, must to change this value in code.

![Scroll Up Gif](https://github.com/luizhss/Gesture_Hand_Controller/blob/master/gifs/scroll_up.gif)

> Scroll Down
* For this one, just keep the detected gesture bellow for a second.
* Observation: in Ubuntu we suggest the value - 3 to scroll; in windows this suggested value is - 70. If you are on Windows, must to change this value in code.

![Scroll Down Gif](https://github.com/luizhss/Gesture_Hand_Controller/blob/master/gifs/scroll_down.gif)

> Zoom
* For this one, first you must keep your hand with the gesture bellow for a little bit. Then perform the "gripper movement". To zoom in, you must spread your fingers (index finger e thumb). To zoom out, you must bring your fingers (index finger and thumb) closer.

![Zoom Gif](https://github.com/luizhss/Gesture_Hand_Controller/blob/master/gifs/zoom.gif)

## Pretrained models

Model used was a C-Support Vector Classification.

Dataset to Train/Validation/Test with shape=(1894, 64)

Qty of tuples per class

|Class|Qty Tuples|
|---|---|
|left_click|311|
|right_click|311|
|scroll_up|311|
|scroll_down|250|
|zoom|400|

Train Size: 80%
Test Size: 20%

Best Parameters Grid Search: 
{'kernel': 'linear', 'gamma': 0.1, 'C': 100}

|Model name|Accuracy|Training/Testing dataset|
| :- | :-: | -: |
|handsPoseClassifier (59KB)|1.0000|Our Dataset|

Classification Report:

|Class|Precision|Recall|f1-score|support|
|---|---|---|---|---|
|left_click|1.0000|1.0000|1.0000|62|
|right_click|1.0000|1.0000|1.0000|62|
|scroll_up|1.0000|1.0000|1.0000|62|
|scroll_down|1.0000|1.0000|1.0000|50|
|zoom|1.0000|1.0000|1.0000|90|

Accuracy Score:

|accuracy metric|--|--|--|--|
|---|---|---|---|---|
|macro_avg|1.0000|1.0000|1.0000|379|
|weighted_avg|1.0000|1.0000|1.0000|379|

## Generate gesture dataset

```
python generate_data.py -f poseClassifier
```
We defined some keywords to provoke specifics events. 

|Key|Action|
|---|---|
|'a'|Add 'mouse' sample|
|'s'|Add 'left_click' sample|
|'d'|Add 'right_click' sample|
|'f'|Add 'scroll_up' sample|
|'g'|Add 'scroll_down' sample|
|'h'|Add 'zoom' sample|
|'q'|Exit and save dataset|
|'w'|Erase last added element|
|'e'|Save dataset|

## Train the dataset
```
python train_hand_poses_classifier.py -d dataset_train.csv -s handsPoseClassifier.pkl
```
Do a grid-search on the dataset and generate a SVC trained model with the best parameters found.

### Performance of this Project

The Hand Detection, Pose Classifier and Gesture Controller have the FPS in CPU:

|Task|FPS (Screen=1360x768) & (Webcam=360x480)|
|---|---|
|Hand Detection + Pose Classifier + Gesture Controller|23.70|

*** Test executed in a Intel® Core™ i5-8250U CPU @ 1.60GHz × 8 (Memory 8GB) (Ubuntu 18.04LTS)

## References
This project is based on MediaPipe Paper.

1. F. Zhang, V. Bazarevsky, A. Vakunov, A. Tkachenka, G. Sung, C. Chang, M. Grundmann. _MediaPipe Hands: On-device Real-time Hand Tracking_, arXiv:2006.10214, 2020. [PDF](https://arxiv.org/abs/2006.10214)

## Documentation

Here are some documentation links:

* [MediaPipe Hands]( https://google.github.io/mediapipe/solutions/hands) (Hand Detect)

* [OpenCV]( https://docs.opencv.org/master/) (Webcam Control)

* [Pyautogui]( https://pyautogui.readthedocs.io/en/latest/mouse.html) (Mouse and Keyboard Control) 

* [Sklearn]( https://scikit-learn.org/0.21/documentation.html) (C-Support Vector Classification Model) 

