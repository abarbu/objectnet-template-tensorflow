ARG TENSORFLOW_VERSION

# Build based on official TensorFlow docker images available on docker hub:
#   https://www.tensorflow.org/install/docker

FROM tensorflow/tensorflow:${TENSORFLOW_VERSION}-gpu

ARG MODEL_CHECKPOINT
ARG MODEL_CLASS_NAME
ENV MODEL_CLASS_NAME ${MODEL_CLASS_NAME}
ENV MODEL_PATH "/workspace/model/"${MODEL_CHECKPOINT}

RUN echo "Building from tensorflow image: tensorflow/tensorflow:$TENSORFLOW_VERSION-gpu"
RUN echo "Using pre-built model: $MODEL_PATH"

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
