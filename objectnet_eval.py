import os
import argparse
import csv
import json
import glob
import tensorflow as tf

from tensorflow import keras
from model.model_description import create_model
from objectnet_iterator import ObjectNetDataset

#sanity
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions


parser = argparse.ArgumentParser(description='Evaluate a PyTorch model on ObjectNet images and output predictions to a CSV file.')
parser.add_argument('images', metavar='images-dir',
                    help='path to dataset')
parser.add_argument('output_file', metavar='output-file',
                    help='path to predictions output file')
#parser.add_argument('model_checkpoint', metavar='model-checkpoint',
#                    help='path to model checkpoint')#TODO uncomment
parser.add_argument('--batch_size', default=64, type=int, metavar='N',
                    help='mini-batch size (default: 64), this is the '
                         'batch size of each GPU on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--convert_outputs_mode', default=1, type=int, metavar='N',
                    help="0: no conversion of prediction IDs, 1: convert from pytorch ImageNet prediction IDs to ObjectNet prediction IDs (default:1)")
args = parser.parse_args()

#assert (not os.path.exists(args.output_file)), "Output file: "+args.output_file+", already exists!"#TODO uncomment
#assert (os.path.exists(os.path.dirname(args.output_file)) or os.path.dirname(args.output_file)==""), "Output file path: "+os.path.dirname(args.output_file)+", does not exist!"

# batch batch_size
assert (args.batch_size >= 1), "Batch size must be >= 1!"
#convert outputs
assert (args.convert_outputs_mode in (0,1)), "Convert outputs mode must be either 0 or 1!"

mapping_file = "mapping_files/imagenet_pytorch_id_to_objectnet_id.json"
with open(mapping_file,"r") as f:
    mapping = json.load(f)
    # convert string keys to ints
    mapping = {int(k): v for k, v in mapping.items()}

def load_model():
    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
    with strategy.scope():
        model = create_model()
        #checkpoint_path = args.model_checkpoint
        checkpoint_path = "resnet50_weights_tf_dim_ordering_tf_kernels.h5"#TODO hardcode
        #model.load_weights(checkpoint_path)#TODO uncomment

        #SANITY CHECK
        model = ResNet50(weights=checkpoint_path)

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
    model = load_model()

    output_predictions = []

    data_iter = ObjectNetDataset(args.images, args.batch_size)

    for img_batch, filenames in data_iter:
        img_batch = preprocess_input(img_batch) #TODO this is for resnet only
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
