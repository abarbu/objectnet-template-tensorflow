# Overview
This repository contains a boilerplate for those wishing to build a custom docker image for the [ObjectNet Challenge](https://eval.ai/web/challenges/challenge-page/726/overview). It assumes you have a custom model which you intend to submit for evaluation to the ObjectNet Challenge.

If your model is built using PyTorch, a more comprehensive docker template along with instructions can be found at [Creating your Docker image from the PyTorch template](https://abarbu.github.io/objectnet-challenge-doc-ibm-dev/dockerfile-from-template.html).

If you are not familiar with docker here are instructions on how to [install docker](https://docs.docker.com/install/), along with a [quick start guide](https://docs.docker.com/get-started/).

To use this example see Section 2.5 of the [ObjectNet Challenge submission instructions](https://abarbu.github.io/objectnet-challenge-doc-ibm-dev/dockerfile-from-scratch.html)

# Section 1: Example Code
## 1.1 Requirements
- python 3
- tensorflow >= 2.3
- cuda 10.1

## Working with NVIDIA
If your local machine has NVIDIA-capable GPUs and you want to test your docker image locally using these GPUs then you will need to ensure the NVIDIA drivers have been installed on your test machine.

The easiest way to enable TensorFlow GPU support is to use [Docker](https://www.tensorflow.org/install/docker) with [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker). (See Section 2)

## 1.2 Code structure
- `objectnet_eval.py` is the main entry point for running this example. It loads the pretrained weights, defined by `model_checkpoint` path, for the model defined in `model/model_description.py`. Full help is available using `objectnet_eval.py --help`
- `model/model_description.py` should be replaced with your own architecture
- `objectnet_iterator.py` contains an example data loader 
- `objectnet_train.py` contains some example training code

# Section 2: Building and testing your docker image

You need:
    - The ~Dockerfile~
    - Your model definition and trained weights

## 2.1 Build your Docker image
```
# syntax: docker build -t <image-name>:<Tag> <Dockerfile path>
# With version tagging:
$ docker build -t  my-model:version1 .

# Or without version tagging:
$ docker build -t  my-model .
```

## 2.2 Running your docker
```
docker run -ti --rm --gpus=all -v $PWD:/workspace -v $PWD/input/images:/input/ -v $PWD/output:/output my-model:version
```
