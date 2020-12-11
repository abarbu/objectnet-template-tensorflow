import os
import glob
import numpy as np
import math

from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow import keras

class ObjectNetDataset(keras.utils.Sequence):
    def __init__(self, image_path, batch_size, transform=None):
        self.filenames = glob.glob(image_path + "/*.png")
        self.batch_size = batch_size
        self.transform = transform

    def __len__(self):
        return math.ceil(len(self.filenames) / self.batch_size)

    def __getitem__(self, idx):
        targets = self.filenames[idx * self.batch_size:(idx + 1) * self.batch_size]

        batch_img = []
        for filename in targets:
            img = load_img(filename)
            
            #crop red border
            width, height = img.size
            cropArea = (2, 2, width-2, height-2)
            img = img.crop(cropArea)
            if self.transform is not None:
                img = self.transform.transforms(img)

            img_np = img_to_array(img)
            batch_img.append(img_np)
        batch_img = np.stack(batch_img, axis=0)
        return batch_img, targets
