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
            img = load_img(filename)

            width, height = img.size

            if width < height:
                img = img.resize((224, int(height*(224/width))))
            else:
                img = img.resize((int(width*(224/height)), 224))

            width, height = img.size
            crop_width = max(width-224, 0)
            crop_height = max(height-224, 0)
            cropArea = (crop_width//2, crop_height//2, width-crop_width//2, height-crop_height//2)
            img = img.crop(cropArea) 

            img = img.resize((224,224))
            img_np = img_to_array(img)

            batch_img.append(img_np)

        return np.stack(batch_img, axis=0), batch_x
