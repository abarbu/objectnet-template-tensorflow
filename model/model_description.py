import tensorflow as tf
from tensorflow import keras

# Define a simple sequential model
def create_model():
  model = tf.keras.models.Sequential([
    keras.layers.Flatten(input_shape=(224,224,3)),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(1000)
  ])

  model.compile(optimizer='adam',
                loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=[tf.metrics.SparseCategoricalAccuracy()])

  return model

