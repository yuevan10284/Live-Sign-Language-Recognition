import os
import pickle
import cv2
import matplotlib.pyplot as plt
import mediapipe as mp

# use mediapipe 
mp_hands = mp.solutions.hands

hands_processor = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)
mp_drawing = mp.solutions.drawing_utils
DATA_DIR = './data'
"""
processed_data = []
class_labels = []
"""
labels = []
data = []

#iterating over the class
for dir_ in os.listdir(DATA_DIR):
    #class_path = os.path.join(DATA_DIR, class_dir)
    for image_file in os.listdir(os.path.join(DATA_DIR, dir_)):
        landmark_data = []

        """
        landmark_x = []
        landmark_y = []
"""
        x_ = []
        y_ = []
        # Loading the ddata
        image = cv2.imread(os.path.join(DATA_DIR, dir_, image_file))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


        detection_results = hands_processor.process(image_rgb)

        # getting handmarks 
        if detection_results.multi_hand_landmarks:
            for hand_landmarks in detection_results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    landmark_data.append(x - min(x_))
                    landmark_data.append(y - min(y_))


            data.append(landmark_data)
            labels.append(dir_)

# saving data and labels
with open('data.pickle', 'wb') as data_file:
    pickle.dump({'data': data, 'labels': labels}, data_file)
