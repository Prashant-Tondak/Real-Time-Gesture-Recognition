import os
import numpy as np
import cv2
from tensorflow.keras.utils import to_categorical

from keras.layers import Input, Dense
from keras.models import Model

is_init = False
size = -1

label = []
dictionary = {}
c = 0

for i in os.listdir():
    if i.endswith(".npy") and not i.startswith("labels"):
        data = np.load(i)
        if not is_init:
            is_init = True
            X = data
            size = data.shape[0]
            y = np.array([i.split('.')[0]] * size)
        else:
            if data.shape[1:] != X.shape[1:]:
                raise ValueError(f"Inconsistent shape: {data.shape} vs {X.shape}")
            X = np.concatenate((X, data))
            y = np.concatenate((y, np.array([i.split('.')[0]] * data.shape[0])))

        label.append(i.split('.')[0])
        dictionary[i.split('.')[0]] = c
        c += 1

# Convert labels to indices
y = np.array([dictionary[label] for label in y])

# Convert to one-hot encoding
y = to_categorical(y, num_classes=len(dictionary))

# Shuffle the dataset
indices = np.arange(X.shape[0])
np.random.shuffle(indices)
X = X[indices]
y = y[indices]

# Define the model
input_shape = X.shape[1]
ip = Input(shape=(input_shape,))

m = Dense(512, activation="relu")(ip)
m = Dense(256, activation="relu")(m)
op = Dense(y.shape[1], activation="softmax")(m)

model = Model(inputs=ip, outputs=op)
model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=['acc'])

# Train the model
model.fit(X, y, epochs=50)

# Save the model and labels
model.save("model.h5")
np.save("labels.npy", np.array(label)) 