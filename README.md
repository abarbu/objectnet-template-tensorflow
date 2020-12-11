# Creating your Docker image from the TensorFlow template
These instructions describe how to build a docker image using the TensorFlow deep learning framework for the [ObjectNet Challenge](https://eval.ai/web/challenges/challenge-page/726/overview). It assumes you have a custom model which you intend to submit for evaluation to the ObjectNet Challenge.

If your model is built using PyTorch, a more comprehensive docker template along with instructions can be found at [Creating your Docker image from the PyTorch template](https://abarbu.github.io/objectnet-challenge-doc-ibm-dev/dockerfile-from-template.html).

If your model is built using a different framework the docker template provided will require [additional customisation](https://abarbu.github.io/objectnet-challenge-doc-ibm-dev/dockerfile-from-scratch.html).

These instructions are split into two sections:
- *Section 1* which describes how to:
    1. run the example code & model on your local machine, and
    2. plug your own model into this example and test on a local machine.
- *Section 2* which describes how to create a docker image ready to submit to the challenge.

# Section 1: ObjectNet competition example model and code

The following section provides example code and a baseline model for the ObjectNet Challenge. The code is structured such that most existing TensorFlow models can be plugged into the example with minimal code changes necessary.

*Note:* The example code uses batching and parallel data loading to improve inference efficiency. If you are building your own customized docker image with your own code it is highly recommended to use similar optimized inferencing techniques to ensure your submission will complete within the time limit set by the challenge organisers.

## 1.1 Requirements
The following libraries are required to run this example and must be installed on the local test machine. The same libraries will be automatically installed into the Docker image when the image is built.
- python 3
- tensorflow 2.3.0
- cuda 10.1

For example, you could set up a conda environment with the necessary requirements with a few simple lines. This environment would be named objectnet_env.
```
conda create -n objectnet_env python=3.7
conda activate objectnet_env
pip install --upgrade pip
pip install tensorflow
```
Alternatively, you can  follow the [instructions here](https://www.tensorflow.org/install) to start running TensorFlow in a docker image.

## 1.2 Install NVIDIA drivers
If your local machine has NVIDIA-capable GPUs and you want to test your docker image locally using these GPUs then you will need to ensure the NVIDIA drivers have been installed on your test machine.

Instructions on how to install the CUDA toolkit and NVIDIA drivers can be found [here](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#package-manager-installation). Be sure to match the versions of CUDA/NVIDIA installed with the version of TensorFlow and CUDA used to build your docker image - see [Building the docker image](https://abarbu.github.io/objectnet-challenge-doc-ibm-dev/dockerfile-from-template.html#23-build-the-docker-image).

## 1.3 Clone git repository containing example

Clone the following git repo to a machine which has docker installed:
```
$ git clone https://github.com/abarbu/objectnet-template-tensorflow.git
```

This repo comes with python scripts to perform batch inference using a sample model, validate and score the inferences and also contains a set of test images (`input/images`) and a file containing ground truth data for those images (`input/answers/answers-test.json`). You will need to download the sample model (ResNet50) used in this example (see 1.6 Testing the example)

## 1.4 Running objectnet_eval.py

`objectnet_eval.py` is the main entry point for running this example; it essentially performs batch inference against all images in a supplied input directory (`images-dir`). Full help is available using `objectnet_eval.py --help`:

```
usage: objectnet_eval.py [-h] [--workers N] [--gpus N] [--batch_size N]
                         [--softmax T/F] [--convert_outputs_mode N]
                         images-dir output-file model-class-name
                         model-checkpoint

Evaluate a TensorFlow model on ObjectNet images and output predictions to a CSV
file.

positional arguments:
  images-dir            path to dataset
  output-file           path to predictions output file
  model-class-name      model class name in model_description.py
  model-checkpoint      path to model checkpoint

optional arguments:
  -h, --help            show this help message and exit
  --workers N           number of data loading workers (default: total num
                        CPUs)
  --gpus N              number of GPUs to use
  --batch_size N        mini-batch size (default: 96), this is the batch size
                        of each GPU on the current node when using Data
                        Parallel or Distributed Data Parallel
  --softmax T/F         apply a softmax function to network outputs to convert
                        output magnitudes to confidence values (default:True)
  --convert_outputs_mode N
                        0: no conversion of prediction IDs, 1: convert from
                        pytorch ImageNet prediction IDs to ObjectNet
                        prediction IDs (default:1)
```
**Note: The default values for `workers` and `batch_size` are tuned for this example. Please do not modify these properties when making an ObjectNet submission using the sample code.**

## 1.5 Code structure

There follows a description of the code structure used in this repo.

_./objectnet_eval.py:_
-   loads the pre-trained model (defined in model-class-name & model-checkpoint file)
-   pre-trained weights for the model parameters are loaded in `objectnet_eval.py -> load_pretrained_net()` by `model.load_state_dict`
-   evaluates batches of images using parallel data loading and 1 or more GPUs (--gpus)
-   aggregates the predictions and writes them to a CSV file (`output-file`)

_./objectnet_iterator.py:_
-   extends TensorFlow `keras.utils.Sequence` class for parallel data loading
-   scans all the images in the `images-dir` folder and makes a list of files. It ignores any subdirectory folder structures
-   loads images using PIL image loader
-   applies transforms specified in `data_transform_description.py` and crops out 2 pixel red border on ObjectNet images

Inside of the model directory: (_This is the only code that you will have to modify_):

_./model/model_description.py:_
-   TensorFlow model description class that can extend `keras.utils.Sequence` to implement any neural net model
-   the current example model is a [ResNet50](https://keras.io/api/applications/resnet/#resnet50-function)
-   add your own model description class to this file.
- make sure to implement the `create_model()` method

_./model/data_transform_description.py:_
-   contains all the dataset preprocessing transformations except cropping out the 2px red pixel border
-   `transforms()` takes a PIL image as input, performs transformations, and returns the transformed image
-   include any customized data transformation code your model requires in this class.

_./input/images:_
-   directory containing a test set of 10 ObjectNet images.

_./input/answers/answers-test.json:_
-   a file containing the ground-truth data for the test images.

## 1.6 Testing the example

Before executing the example for the first time you must download the sample model as shown below:

### Download the model:
```
$ cd objectnet-template-tensorflow
$ mkdir downloads
$ cd downloads
$ wget https://storage.googleapis.com/tensorflow/keras-applications/resnet/resnet50_weights_tf_dim_ordering_tf_kernels.h5
$ cp resnet50_weights_tf_dim_ordering_tf_kernels.h5 ../model
$ cd ..
```

**Note:** The `downloads/` directory is used to store downloaded models so they only need to be downloaded once. If you want to use a model which is in `downloads/`, make sure to copy it to `model/` as shown in the second last line above. This way, `model/` can be kept with only one active model at once, and `downloads/` can be used as storage for all models.

Use the following arguments to run `objectnet_eval.py` and test the example model:
### Perform batch inference:
```
$ python3 objectnet_eval.py input/images output/predictions.csv DemoResNet50 model/resnet50_weights_tf_dim_ordering_tf_kernels.h5

**** params ****
images input/images
output_file output/predictions.csv
model_class_name resnext101_32x48d_wsl
model_checkpoint model/ig_resnext101_32x48-3e41cc8a.pth
workers 16
gpus 2
batch_size 96
softmax True
convert_outputs_mode 1
****************

initializing model ...
loading pretrained weights from disk ...
Done. Number of predictions:  10
```
Results will be written to the `predictions.csv` file in the `output/` directory. Check the output conforms to the [format](https://eval.ai/web/challenges/challenge-page/726/evaluation) expected by the ObjectNet Challenge.

## 1.7 Modifying the code to use your own PyTorch model

You can plugin your own existing TensorFlow model into the template. There are a few considerations to keep in mind, which are listed below.

### 1.7.1 Requirements

If you want to use a python package that is not included in the default TensorFlow Docker container, then it needs to be listed in the `requirements.txt` file so that it is 'pip installed' when the docker image is built. Include it as follows:
```
# This file specifies python dependencies which are to be installed into the Docker image.
# List one library per line (not as a comment)
# e.g.
#numpy
scipy
```
### 1.7.2 Template changes

The only code changes necessary when incorporating your TensorFlow model should be in the `model/` directory.

1.  Before downloading your model checkpoint file, remove the existing checkpoint file from the `model/` directory. For example:
 ```
    $ rm -rf model/resnet50_weights_tf_dim_ordering_tf_kernels.h5
 ```
2.   Copy your model checkpoint into `model/`. For example:

```
$ cp my_model.h5 /model
```

**Note:** When your docker image is submitted to the challenge for evaluation the image **will not** have internet access and as such will not be able to download model checkpoints from the internet. For this reason it is essential that your model is included in the built docker image.

3.  Add your model description as a class to `model/model_description.py`. The class name will be used as the `model-class-name` argument to `objectnet_eval.py`.
4. Amend the following parameters in `data_transformation_description.py` to match those that your model was trained on.
5.  Test your model's inference using the test images and ground-truth data provided in the `objectnet-template-pytorch`:

```
$ python3 objectnet_eval.py input/images output/predictions.csv MyModel model/my_model.h5

  **** params ****
  images input/images
  output_file output/predictions.csv
  model_class_name Inception3
  model_checkpoint model/inception_v3_google-1a9a5a14.pth
  workers 16
  gpus 2
  batch_size 96
  softmax True
  convert_outputs_mode 1
  ****************

  initializing model ...
  loading pretrained weights from disk ...
  Done. Number of predictions:  10
```
Note: If you want to run inference again or with another model, you will first have to delete the predictions output file.
```
$ rm output/predictions.csv
```

