import os
import numpy as np
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tqdm import tqdm
from django.apps import AppConfig
import threading

LOCK_FILE_PATH = 'model_loaded.lock'

lock = threading.Lock()

def run_script():
    labels = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']
    X_train = [] 
    y_train = [] 
    X_test = []  
    y_test = []
    image_size = 150

    for label in labels:
        folderPath = os.path.join("D:/", 'Training', label)
        for filename in tqdm(os.listdir(folderPath)): 
            img_path = os.path.join(folderPath, filename)
            img = cv2.imread(img_path)   
            if img is not None:
                img = cv2.resize(img, (image_size, image_size))   
                X_train.append(img)
                y_train.append(labels.index(label))

    for label in labels:
        folderPath = os.path.join('D:/', 'Testing', label)
        for filename in tqdm(os.listdir(folderPath)):  
            img_path = os.path.join(folderPath, filename)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, (image_size, image_size))
                X_test.append(img)
                y_test.append(labels.index(label))

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)


    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(image_size, image_size, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(len(labels), activation='softmax'))
    model.compile(optimizer='adam', 
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), batch_size=32, verbose=1)

    
    with open(LOCK_FILE_PATH, 'w') as lock_file:
        lock_file.write('Model loaded\n')
    
    return model

class MyappConfig(AppConfig):
    name = 'myproject'
    _model = None  
    @property
    def model(self):
        if MyappConfig._model is None:
            print("Model not loaded yet.")
        return MyappConfig._model

    def ready(self):
        with lock:
            if not os.path.exists(LOCK_FILE_PATH):
                print("Starting model loading...")
                MyappConfig._model = run_script()
                print("Model loaded successfully.")
            else:
                print("Model already loaded, skipping...")
