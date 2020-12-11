'''
Define your model class here.
All classes must implement the create_model() method
'''

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.resnet50 import ResNet50

class DemoResNet50():
    def create_model():
        return ResNet50(weights="resnet50_weights_tf_dim_ordering_tf_kernels.h5")#hard-coded to prevent keras from downloading again
