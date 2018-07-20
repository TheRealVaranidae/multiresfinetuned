#!/bin/bash

python inference.py -c ./multiresfinetuned5296.pb --input-image $1 --output-dir $2 
