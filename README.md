# Automatic-Number-Plate-Recognition-YOLOv8
## Demo


https://github.com/Muhammad-Zeerak-Khan/Automatic-License-Plate-Recognition-using-YOLOv8/assets/79400407/1af57131-3ada-470a-b798-95fff00254e6



## Data

The video used in the tutorial can be downloaded [here](https://drive.google.com/file/d/1JbwLyqpFCXmftaJY1oap8Sa6KfjoWJta/view?usp=sharing).

## Model

A Yolov8 pre-trained model (YOLOv8n) was used to detect vehicles.

A licensed plate detector was used to detect license plates. The model was trained with Yolov8 using [this dataset](https://universe.roboflow.com/roboflow-universe-projects/license-plate-recognition-rxg4e/dataset/4). 
- The model is available [here](https://drive.google.com/file/d/1Zmf5ynaTFhmln2z7Qvv-tgjkWQYQ9Zdw/view?usp=sharing).

## Dependencies

The sort module needs to be downloaded from [this repository](https://github.com/abewley/sort).

```bash
git clone https://github.com/abewley/sort
```

## Project Setup

* Make an environment with python=3.10 using the following command 
``` bash
python -m venv C:\path\to\new\virtual\environment
```

* Install the project dependencies using the following command 
```bash
pip install -r requirements.txt
```

* Instructions

```bash
"to start the backend server"
uvicorn main:app --reload

"to start the frontend"
cd frontend
py app.py
#head to the hyperlink provided by flask and a video file...
## output part is not yet implemented but the csv are available in the output folder and 
## the final output video is available in the frontend\static\results folder
```
