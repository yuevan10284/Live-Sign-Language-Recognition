# 🤟Sign Language Recognition🤟
This project harnesses the power of machine learning (ML) and image processing to create an innovative solution for real-time sign language recognition. Using a webcam, the system can recognize and interpret sign language gestures, making communication more accessible for those who rely on sign language.

## 🔗 Developers
* [@Mitchell Cootauco](https://github.com/Mcootauc)
* [@Owen Hunger](https://github.com/ohunger)
* [@Evan Yu](https://github.com/yuevan10284)

## Tech Stack
![Python](https://img.shields.io/badge/-Python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) 
* Main programming language used.

![MediaPipe](https://img.shields.io/badge/-MediaPipe-34A853?style=for-the-badge&logo=google&logoColor=white) 
* Used for real-time hand tracking and gesture recognition.

## 🛠️  Setup Instructions
# Prerequisites
* Python 3.8.10
* Webcam for capturing sign language gestures
# Installation
1. Clone the repository to your local machine
   `git clone [repository URL]`

2. Change directory (cd) into the project folder
   `cd [project folder name]`
3. Install the required packages using pip:
   `pip install numpy opencv-python==4.7.0.68 mediapipe==0.9.0.1 scikit-learn==1.2.0`

## 📸Collecting Sign Language Data
1. Open the `collect_images.py` script in a text editor.
2. Modify the line `cap = cv2.VideoCapture(0)` to use your desired camera. The number `0` typically refers to the default webcam. If you have multiple cameras, you might need to change this to `1`, `2`, etc.
3. Save the changes and run the scrip tto start collecting data:
   `python collect_imgs.py`
4. Follow the on-screen instructions to capture images of different sign language gestures. 

## 🕵️ Sign Language Detection
1. After collecting sufficient data, run the training script to train the machine learning model:
   `python train_classifier.py`
2. To start recognizing sign language gestures in real-time, run:
   `python inference_classifier.py`
3. The program will access your webcam. Perform sign language gestures in front of the camera to see the model's predictions.
## Exiting the Program
* To exit the real-time detection, press the `Esc
