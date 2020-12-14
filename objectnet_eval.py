import os
import argparse
import csv
import json
import glob
import tensorflow as tf

from tensorflow import keras
from model import model_description
from objectnet_iterator import ObjectNetDataset
from model.data_transform_description import data_transform

gpu_count = len(tf.config.list_physical_devices('GPU'))
cpu_count = len(tf.config.list_physical_devices('CPU'))
parser = argparse.ArgumentParser(description='Evaluate a TensorFlow model on ObjectNet images and output predictions to a CSV file.')
parser.add_argument('images', metavar='images-dir',
                    help='path to dataset')
parser.add_argument('output_file', metavar='output-file',
                    help='path to predictions output file')
parser.add_argument('model_class_name', metavar='model-class-name',
                    help='model class name in model_description.py')
parser.add_argument('model_checkpoint', metavar='model-checkpoint',
                    help='path to model checkpoint')
parser.add_argument('--gpus', default=gpu_count, type=int, metavar='N',
                    help='number of GPUs to use')
parser.add_argument('--workers', default=cpu_count, type=int, metavar='N',
                    help='number of data loading workers (default: total num CPUs)')
parser.add_argument('--batch_size', default=64, type=int, metavar='N',
                    help='mini-batch size (default: 64), this is the '
                         'batch size of each GPU on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--softmax', default=True, type=bool, metavar='T/F',
                    help="apply a softmax function to network outputs to convert output magnitudes to confidence values (default:True)")
parser.add_argument('--convert_outputs_mode', default=1, type=int, metavar='N',
                    help="0: no conversion of prediction IDs, 1: convert from pytorch ImageNet prediction IDs to ObjectNet prediction IDs (default:1)")
args = parser.parse_args()

assert (not os.path.exists(args.output_file)), "Output file: "+args.output_file+", already exists!"
assert (os.path.exists(os.path.dirname(args.output_file)) or os.path.dirname(args.output_file)==""), "Output file path: "+os.path.dirname(args.output_file)+", does not exist!"
# GPUs
assert (gpu_count > 0), "No GPUs detected!"
assert (args.gpus <= gpu_count), "Requested "+args.gpus+" ,but only "+gpu_count+" are availible!"
assert (args.gpus >= 1), "You have to use at least 1 GPU!"

# model class name
try:
    getattr(model_description, args.model_class_name)
except AttributeError as e:
    print("Module: " + args.model_class_name + ", can not be found in model_description.py!")
    raise

MODEL_CLASS_NAME = args.model_class_name

try:
    architecture = getattr(model_description, MODEL_CLASS_NAME)
    architecture.create_model()
except AttributeError as e:
    print("Module: " + args.model_class_name + ", must implement create_model() method!")
    raise

# model check point file
assert (os.path.exists(args.model_checkpoint)), "Model checkpoint file: "+args.model_checkpoint+", does not exist!"

# workers
assert (args.workers <= cpu_count), "Number of workers: "+args.workers + ", should be <= the number of CPUs " + cpu_count+"!"
assert (args.workers >= 1), "Number of workers must be >= 1!"

# batch batch_size
assert (args.batch_size >= 1), "Batch size must be >= 1!"
#convert outputs
assert (args.convert_outputs_mode in (0,1)), "Convert outputs mode must be either 0 or 1!"

#input images path
print()
print("**** params ****")
for k in vars(args):
    print(k,vars(args)[k])
OBJECTNET_IMAGES_FOLDER = args.images
print("****************")
print()

mapping_file = "mapping_files/imagenet_id_to_objectnet_id.json"
with open(mapping_file,"r") as f:
    mapping = json.load(f)
    # convert string keys to ints
    mapping = {int(k): v for k, v in mapping.items()}

def load_model():
    devices = [f'/gpu:{i}' for i in range(args.gpus)]

    strategy = tf.distribute.MirroredStrategy(devices=devices)
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
    with strategy.scope():
        architecture = getattr(model_description, MODEL_CLASS_NAME)
        model = architecture.create_model()
        model.load_weights(args.model_checkpoint)
        model.summary()
    return model

def evalModels():
    '''
    returns:  [
                imageFileName1, prediction 1, prediction 2, ..., confidence 1, confidence 2, ...,
                imageFileName2, prediction 1, prediction 2, ..., confidence 1, confidence 2, ...,
                ...
              ]
    '''
    # Create a basic model instance
    output_predictions = []

    batches_per_device = args.batch_size #upper bound estimate of how much data will fit in GPU memory, tune this based ou GPU memory availible
    batch_size = (batches_per_device*args.gpus)
    data_iter = ObjectNetDataset(args.images, batch_size, transform=data_transform())


    model = load_model()

    for img_batch, filenames in data_iter:
        predictions = model.predict(img_batch, batch_size=args.batch_size, workers=args.workers, use_multiprocessing=True)

        prediction_confidence, prediction_class = tf.math.top_k(predictions, 5)
        
        if args.softmax:
            predictions = tf.keras.layers.Softmax(predictions)

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
