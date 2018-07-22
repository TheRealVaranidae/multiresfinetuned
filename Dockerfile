FROM ubuntu:16.04

RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y python python-setuptools python-pip

COPY requirements.txt /tmp/
RUN pip install --requirement /tmp/requirements.txt
COPY . /tmp/

ADD ./multiresfinetuned5296.pb /multiresfinetuned5296.pb
ADD ./create_detections_1.py /create_detections_1.py
ADD ./create_detections.py /create_detections.py
ADD ./det_util.py /det_util.py
ADD ./class_id_map.json /class_id_map.json
ADD ./run.sh /run.sh
ADD ./2568.tif /2568.tif
