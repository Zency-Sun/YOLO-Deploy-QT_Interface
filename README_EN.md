# YOLO-Deploy-QT_Interface
Visual QT interface for deploying YOLOv5 and YOLOv8 for ONNX and OpenVino deployment of images, folders, videos, cameras
![demo](assets/Demo.gif)

## 1.Environment Setup
This program uses Conda to create an environment, which is created and activated by following the following commands:
```bash
conda create -n qtenv python=3.8
conda activate qtenv
```
After activating the environment, go to this project folder and install the necessary packages for the environment:
```bash
pip install -r requirements.txt
```
At this point, the environment configuration is complete and can be used as the environment for the project.

## 2.Data and model preparation
- Before using the software, you need to prepare the images, folders, videos, cameras, etc. that need to be detected
-  Download this project provides YOLOv5, YOLOv8 ONNX, OpenVINO [model file](https://github.com/Zency-Sun/YOLO-Deploy-QT_Interface/releases/tag/v1.0.0)(note Unzip) and put it in the project folder
- Note: The model file provided in this project is directly converted from the YOLOv5/8 COCO pre-training model. If you need to customize the categories, you need to change to your own model and modify the categories in line 12 and line 32 of deploy_yolov5.py and deploy_yolov8.py

## 3. User guide
- Open and run main.py
- Select Model Type, Deployment Type, File Type, Model Location, and File Location, and set parameters
- Click "Save configuration", then click "Start running" to run, you can see the program running status in "Output log"
- "Stop Running" can stop the program running, "exit" can exit the program
- Program results are stored in the./results folder


## 4. Statement
- ** If the code is helpful to you, please click a Star, you can raise Issues together **
- ** Please indicate the source of the code, refuse white whoring, carry forward the open source spirit together, piracy will be prosecuted! **

