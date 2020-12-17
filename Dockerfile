FROM rapidsai/rapidsai:0.16-cuda11.0-runtime-ubuntu18.04-py3.8

ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /workspace

COPY ./ ./

RUN apt update && apt -y upgrade && apt install -y \
  build-essential \
  cmake \
  git \
  libboost-dev \
  libboost-system-dev \
  libboost-filesystem-dev

RUN pip install --upgrade pip && pip install -r requirements.txt

# Install LightGBM
RUN git clone --recursive https://github.com/microsoft/LightGBM && cd LightGBM \
  && mkdir build \
  && cd build \
  && cmake .. \
  && make -j4 

RUN cd LightGBM/python-package \
  && python setup.py install

RUN rm -r -f LightGBM/
