import os
import argparse
import csv
import json
import glob
import numpy as np
import tensorflow as tf
import math

from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow import keras

from model.model_description import create_model

with open("input/answers/answers-test.json") as f:
    answers = json.load(f)
    train_labels = [answers[x.split('/')[-1]] for x in filenames]
train_labels = np.array(train_labels)

filenames = glob.glob("input/images/*.png")
for filename in filenames:
    img = load_img(filename, target_size=(224, 224))
    img_np = img_to_array(img)
    train_images.append(img_np)
train_images =np.array(train_images)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

# Create a basic model instance
model = create_model()

model.summary()
checkpoint_path = "training_1/cp.ckpt"
model.load_weights(checkpoint_path)

# Train the model with the new callback
model.fit(train_images, 
          train_labels,  
          epochs=10,
          validation_data=(test_images, test_labels),
          callbacks=[cp_callback])  # Pass callback to training


