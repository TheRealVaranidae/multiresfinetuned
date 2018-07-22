FROM ubuntu:16.04

RUN apt-get update && apt-get install -y python python-setuptools python-pip

COPY requirements.txt /tmp/
RUN pip install --requirement /tmp/requirements.txt
COPY . /tmp/

COPY ./multiresfinetuned5296.pb /multiresfinetuned5296.pb
COPY ./create_detections_1.py /create_detections_1.py
COPY ./create_detections.py /create_detections.py
COPY ./det_util.py /det_util.py
COPY ./class_id_map.json /class_id_map.json
COPY ./run.sh /run.sh
COPY ./2568.tif /2568.tif
