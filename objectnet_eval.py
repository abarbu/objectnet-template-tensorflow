import os
import argparse
import csv
import json
import glob
import numpy as np
import tensorflow as tf
import math

from tensorflow.keras.applications import vgg16
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array

from tensorflow import keras
from tensorflow.keras.applications.imagenet_utils import decode_predictions

from model.model_description import create_model

parser = argparse.ArgumentParser(description='Evaluate a PyTorch model on ObjectNet images and output predictions to a CSV file.')
parser.add_argument('images', metavar='images-dir',
                    help='path to dataset')
parser.add_argument('output_file', metavar='output-file',
                    help='path to predictions output file')
parser.add_argument('--batch_size', default=96, type=int, metavar='N',
                    help='mini-batch size (default: 96), this is the '
                         'batch size of each GPU on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--convert_outputs_mode', default=1, type=int, metavar='N',
                    help="0: no conversion of prediction IDs, 1: convert from pytorch ImageNet prediction IDs to ObjectNet prediction IDs (default:1)")
args = parser.parse_args()

# batch batch_size
assert (args.batch_size >= 1), "Batch size must be >= 1!"
#convert outputs
assert (args.convert_outputs_mode in (0,1)), "Convert outputs mode must be either 0 or 1!"

class ObjectNetDataset(keras.utils.Sequence):
    def __init__(self, image_path, batch_size):
        self.filenames = glob.glob(args.images + "/*.png")
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(len(self.filenames) / self.batch_size)

    def __getitem__(self, idx):
        batch_x = self.filenames[idx * self.batch_size:(idx + 1) * self.batch_size]

        batch_img = []
        print(batch_x)
        for filename in batch_x:
            print(filename)
            img = load_img(filename, target_size=(224,224))
            img_np = img_to_array(img)
            batch_img.append(img_np)
        return np.array(batch_img), batch_x


#with open("input/answers/answers-test.json") as f:
#    answers = json.load(f)
#    train_labels = [answers[x.split('/')[-1]] for x in filenames]
#train_labels = np.array(train_labels)

# Create a basic model instance
model = create_model()

model.summary()
checkpoint_path = "training_1/cp.ckpt"
model.load_weights(checkpoint_path)

#for filename in filenames:
#    img = load_img(filename, target_size=(224, 224))
#    img_np = img_to_array(img)
#    train_images.append(img_np)
#train_images =np.array(train_images)
#
#with open("input/answers/answers-test.json") as f:
#    answers = json.load(f)
#    train_labels = [answers[x.split('/')[-1]] for x in filenames]
#train_labels = np.array(train_labels)

#assert (not os.path.exists(args.output_file)), "Output file: "+args.output_file+", already exists!"
#assert (os.path.exists(os.path.dirname(args.output_file)) or os.path.dirname(args.output_file)==""), "Output file path: "+os.path.dirname(args.output_file)+", does not exist!"

mapping_file = "mapping_files/imagenet_pytorch_id_to_objectnet_id.json"
with open(mapping_file,"r") as f:
    mapping = json.load(f)
    # convert string keys to ints
    mapping = {int(k): v for k, v in mapping.items()}

#model = vgg16.VGG16(weights='vgg16_weights_tf_dim_ordering_tf_kernels.h5')

#filenames = glob.glob(args.images + "/*.png")
#print("filenames", filenames)

def evalModels():
    '''
    returns:  [
                imageFileName1, prediction 1, prediction 2, ..., confidence 1, confidence 2, ...,
                imageFileName2, prediction 1, prediction 2, ..., confidence 1, confidence 2, ...,
                ...
              ]
    '''

    output_predictions = []

    data_iter = ObjectNetDataset(args.images, args.batch_size)

    for img_batch, filenames in data_iter:
        predictions = model.predict(img_batch, batch_size=args.batch_size)

        prediction_confidence, prediction_class = tf.math.top_k(predictions, 5)
        
        prediction_confidence = prediction_confidence.numpy()
        prediction_class = prediction_class.numpy()

        for i in range(len(filenames)):
            out = [filenames[i].split("/")[-1]]
            if args.convert_outputs_mode == 1:
                tfImageNetIDToObjectNetID(prediction_class[i])
            output_predictions.append(out + [val for pair in zip(prediction_class[i],prediction_confidence[i]) for val in pair])
    return output_predictions


def tfImageNetIDToObjectNetID(prediction_class):
    for i in range(len(prediction_class)):
        if prediction_class[i] in mapping:
            prediction_class[i] = mapping[prediction_class[i]]
        else:
            prediction_class[i] = -1
    

objectnet_predictions = evalModels()
print("output_predictions", objectnet_predictions)
with open(args.output_file, 'w') as csvOut:
    csvwriter = csv.writer(csvOut, delimiter=',')
    for row in objectnet_predictions:
        csvwriter.writerow(row)
print("Done. Number of predictions: ", len(objectnet_predictions))

