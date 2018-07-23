FROM ubuntu:16.04
FROM python:3.6.5-onbuild

COPY requirements.txt requirements.txt

RUN pip3 install -r requirements.txt


RUN mv 2568.tif /2568.tif

RUN mv multiresfinedtuned5296.pb model.pb

