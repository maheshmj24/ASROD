# ASROD - Alert System with Real Time Object Detection

## Uses

- Tensorflow Object Detection API.
- A model chosen from the model zoo provided by tensorflow.
  - Eg: ssd_inception_v2_coco model trained on COCO[Common Objects in Context] dataset.
- OpenCV.
- Tkinter
  - Used for GUI
- Pushbullet
  - [Optional Integration] Used for sending notifications to phone upon detection.

## Usage

- Download preferred models available on [Tensorflow detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md) and add to the models folder
  - faster_rcnn_inception_v2_coco_2018_01_28
  - ssd_inception_v2_coco_11_06_2017 etc...
- Update the parameter **model_name** in parameters.json.
- Run the file object_detection_app.py

## Pre-requisites (Windows)

1. Tensorflow v1 (v1.13.*)
1. Python 3.7.* or lower
1. OpenCV
1. Matplotlib
1. Protobuf 3.20.* or lower

*Run ``` pip install -r requirements.txt ```*

- Follow this [tutorial](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html) to setup the object detection api in your machine

  - If Visual C++ 2015 build tools is throwing an error and not installing use one of the following fixes.
    - [[SOLVED] Unable to install Visual C++ build tools](<https://www.youtube.com/watch?v=p_R3tXSq0KI>)
    - Or Download the offline installer of Visual C++ Build Tools for Visual Studio 2015 with Update 3
    - Use this [fix](https://stackoverflow.com/questions/43847542/rc-exe-no-longer-found-in-vs-2015-command-prompt/45319119#45319119) if you face an error similar to 'LINK : fatal error LNK1158: cannot run 'rc.exe''
