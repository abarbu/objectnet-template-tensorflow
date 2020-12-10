# Build arguments to control base tensorflow image. Tensorflow build
# tags for docker images can be found at:
# https://hub.docker.com/r/tensorflow/tensorflow/tags
#
# From the command line (`docker build...`) use:
#  --build-arg GPU: omit or "-gpu" to build a GPU enabled image,
#                   or "" to build non GPU image.
#  --build-arg TENSORFLOW_VERSION: omit to use "2.3.0", otherwise
#                   specify version.
#  Default is to use: tensorflow/tensorflow:2.3.0-gpu

ARG GPU="-gpu"
ARG TENSORFLOW_VERSION="2.3.0"

# Build based on official TensorFlow docker images available on docker hub:
#   https://www.tensorflow.org/install/docker

FROM tensorflow/tensorflow:${TENSORFLOW_VERSION}${GPU}

# Need to redefine the ARGS here so we can print them out
ARG GPU
ARG TENSORFLOW_VERSION


RUN echo "Building from tensorflow image: tensorflow/tensorflow:$TENSORFLOW_VERSION$GPU"

# Add metadata
LABEL maintainer="your-email-address"
LABEL version="0.1"
LABEL description="Docker image for ObjectNet AI Challenge."

# Set working directory
WORKDIR /workspace

# Install python packages listed in requirements.txt
COPY requirements.txt /workspace
RUN pip install -r requirements.txt

# Copy (recursively) all files from the current directory into the
# image at /workspace
COPY . /workspace

# Define the command to execute when the container is run
ENTRYPOINT python objectnet_eval.py /input /output/predictions.csv $MODEL_CLASS_NAME $MODEL_PATH 
