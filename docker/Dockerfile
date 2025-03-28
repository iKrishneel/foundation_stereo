# FROM pytorch/pytorch:2.6.0-cuda11.8-cudnn9-devel
FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-devel

LABEL org.opencontainers.image.source="https://github.com/iKrishneel/foundation_stereo/" \
    org.opencontainers.image.description="Foundation Stereo" \
    org.opencontainers.image.version="0.0.1"

ENV LANG=en_US.UTF-8
ENV DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_VISIBLE_DEVICES=all

SHELL ["/bin/bash", "-c"]

COPY . /root/foundation_stereo

RUN apt-get update && \
    apt-get install -y ffmpeg libsm6 libxext6 && \
    rm -rf /var/lib/apt/lists/*

RUN cd /root/foundation_stereo && \
    pip install -U xformers --index-url https://download.pytorch.org/whl/cu126 && \
    pip install -r requirements.txt --no-build-isolation && \
    pip install -e . && \
    rm -rf /root/foundation_stereo
