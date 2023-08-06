FROM pytorch/pytorch:1.9.0-cuda11.1-cudnn8-devel

# Use bash as default shell, rather than sh
ENV SHELL /bin/bash

WORKDIR /code

COPY requirements.txt /code
RUN apt update
RUN apt install -y libglib2.0-0 libsm6 libxext6 libxrender-dev git
RUN git clone https://github.com/lameski123/thesis /pointnet2_lib
RUN cd /pointnet2_lib/pointnet2 && python setup.py install
RUN pip install -r requirements.txt
RUN pip install chamferdist
