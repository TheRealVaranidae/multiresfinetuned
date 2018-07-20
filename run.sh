#!/bin/bash

python create_detections.py -c ./multiresfinetuned5296.pb -cs 300 -i $1 -o $2 
