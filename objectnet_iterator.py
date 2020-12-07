import os
import glob
import numpy as np
import math

from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow import keras

class ObjectNetDataset(keras.utils.Sequence):
    def __init__(self, image_path, batch_size):
        self.filenames = glob.glob(image_path + "/*.png")
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(len(self.filenames) / self.batch_size)

    def __getitem__(self, idx):
        batch_x = self.filenames[idx * self.batch_size:(idx + 1) * self.batch_size]

        batch_img = []
        for filename in batch_x:
            img = load_img(filename, target_size=(224,224))
            img_np = img_to_array(img)
            batch_img.append(img_np)
        return np.array(batch_img), batch_x
