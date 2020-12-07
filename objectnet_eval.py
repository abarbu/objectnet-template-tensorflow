import os
import argparse
import csv
import json
import glob
import tensorflow as tf

from tensorflow import keras

from model.model_description import create_model
from objectnet_iterator import ObjectNetDataset

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

#assert (not os.path.exists(args.output_file)), "Output file: "+args.output_file+", already exists!"
#assert (os.path.exists(os.path.dirname(args.output_file)) or os.path.dirname(args.output_file)==""), "Output file path: "+os.path.dirname(args.output_file)+", does not exist!"

# batch batch_size
assert (args.batch_size >= 1), "Batch size must be >= 1!"
#convert outputs
assert (args.convert_outputs_mode in (0,1)), "Convert outputs mode must be either 0 or 1!"

# Create a basic model instance
model = create_model()

model.summary()
checkpoint_path = "training_1/cp.ckpt"
model.load_weights(checkpoint_path)

mapping_file = "mapping_files/imagenet_pytorch_id_to_objectnet_id.json"
with open(mapping_file,"r") as f:
    mapping = json.load(f)
    # convert string keys to ints
    mapping = {int(k): v for k, v in mapping.items()}

#model = vgg16.VGG16(weights='vgg16_weights_tf_dim_ordering_tf_kernels.h5')

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
with open(args.output_file, 'w') as csvOut:
    csvwriter = csv.writer(csvOut, delimiter=',')
    for row in objectnet_predictions:
        csvwriter.writerow(row)
print("Done. Number of predictions: ", len(objectnet_predictions))
