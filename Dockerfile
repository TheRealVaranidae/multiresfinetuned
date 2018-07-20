FROM ubuntu:16.04
FROM python:3.6.5-onbuild

COPY requirements.txt /tmp/
RUN pip install --requirement /tmp/requirements.txt
COPY . /tmp/

ADD ./multiresfinetuned5296.pb /multiresfinetuned5296.pb
ADD ./create_detections.py /create_detections.py
ADD ./det_util.py /det_util.py
ADD ./run.sh /run.sh
ADD ./2568.tif /2568.tif
