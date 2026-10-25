FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

WORKDIR /workspace
RUN pip install --no-cache-dir numpy pyyaml scikit-learn==1.6.1
COPY . /workspace
