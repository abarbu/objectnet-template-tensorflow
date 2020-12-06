import os
import argparse
import csv
import json
import glob
import numpy as np
from tensorflow.keras.applications import vgg16
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array

def decode_predictions(preds, top=5):
  if len(preds.shape) != 2 or preds.shape[1] != 1000:
    raise ValueError('`decode_predictions` expects '
                     'a batch of predictions '
                     '(i.e. a 2D array of shape (samples, 1000)). '
                     'Found array with shape: ' + str(preds.shape))

  with open("imagenet_class_index.json") as f:
    CLASS_INDEX = json.load(f)

  results = []
  for pred in preds:
    top_indices = pred.argsort()[-top:][::-1]
    result = [tuple(CLASS_INDEX[str(i)]) + (pred[i],) for i in top_indices]
    result.sort(key=lambda x: x[2], reverse=True)
    results.append(result)
  return results

parser = argparse.ArgumentParser(description='Evaluate a PyTorch model on ObjectNet images and output predictions to a CSV file.')
parser.add_argument('images', metavar='images-dir',
                    help='path to dataset')
parser.add_argument('output_file', metavar='output-file',
                    help='path to predictions output file')
args = parser.parse_args()

assert (not os.path.exists(args.output_file)), "Output file: "+args.output_file+", already exists!"
assert (os.path.exists(os.path.dirname(args.output_file)) or os.path.dirname(args.output_file)==""), "Output file path: "+os.path.dirname(args.output_file)+", does not exist!"

mapping_file = "mapping_files/imagenet_pytorch_id_to_objectnet_id.json"
with open(mapping_file,"r") as f:
    mapping = json.load(f)
    # convert string keys to ints
    mapping = {int(k): v for k, v in mapping.items()}

imagenet_label_lookup={}
with open('mapping_files/map_clsloc.txt','r') as f:
    zz = f.readlines()
    for l in zz:
        ll = l.strip().split(" ")
        imagenet_label_lookup[ll[0]]=int(ll[1])

def pytorchImageNetIDToObjectNetID(prediction_class):
    for i in range(len(prediction_class)):
        if prediction_class[i] in mapping:
            prediction_class[i] = mapping[prediction_class[i]]
        else:
            prediction_class[i] = -1

model = vgg16.VGG16(weights='vgg16_weights_tf_dim_ordering_tf_kernels.h5')

filenames = glob.glob(args.images + "/*.png")
output_predictions = []
for filename in filenames:
    img = load_img(filename, target_size=(224, 224))
    img_np = img_to_array(img)
    img_batch = np.expand_dims(img_np, axis=0)
    img_batch_processed = vgg16.preprocess_input(img_batch.copy())
    predictions = model.predict(img_batch_processed)
    predictions_decoded = decode_predictions(predictions)[0]

    out = [filename.split("/")[-1]]
    for p in predictions_decoded:
        try:
            label = p[0]
            label_idx = imagenet_label_lookup[label]
        except KeyError:
            label_idx = 0

        if label_idx in mapping:
          label_idx_objnet = mapping[label_idx]
        else:
          label_idx_objnet = -1

        out.append(label_idx_objnet)
        out.append(p[2])
    output_predictions.append(out)

with open(args.output_file, 'w') as csvOut:
    csvwriter = csv.writer(csvOut, delimiter=',')
    for row in output_predictions:
        csvwriter.writerow(row)
