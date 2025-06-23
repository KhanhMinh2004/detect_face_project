import os
import numpy as np

from tensorflow.keras import layers
from tensorflow.keras import models
from utils.handle_data import get_data


traindata = 'D:/Hoc_pY/detect_face_project/data/traindata'
testdata = 'D:/Hoc_pY/detect_face_project/data/testdata'

X_train = []
y_train = []
X_test = []
y_test = []

X_train = get_data(traindata, X_train)
X_test = get_data(testdata, X_test)
np.random.shuffle(X_train)

model_training_first = models.Sequential([
    layers.Conv2D(32, (3, 3), input_shape=(100, 100, 3), activation='relu',  padding='same'),
    layers.MaxPool2D((2,2)),
    layers.Dropout(0.15),

    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.MaxPool2D((2, 2)),
    layers.Dropout(0.2),

    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.MaxPool2D((2, 2)),
    layers.Dropout(0.2),

    layers.Flatten(),
    layers.Dense(1000, activation='relu'),
    layers.Dense(256, activation='relu'),
    layers.Dense(3, activation='softmax')
])

model_training_first.summary()
model_training_first.compile(optimizer='adam',
                             loss='categorical_crossentropy',
                             metrics=['accuracy'])

model_training_first.fit(np.array([x[0] for i , x in enumerate(X_train)]),np.array([y[1] for i ,
                                                y in enumerate(X_train)]), validation_split=0.2, epochs=50)
model_training_first.save('model-preFace.h5')

