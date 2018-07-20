#!/bin/bash

set -exuo pipefail
VALIDATE_FILE=$1
OUTPUT_DIR=$2
OUTPUT_FILE = ${OUTPUT_DIR}/$(basename $VALIDATE_FILE).txt
python create_detections.py -c ./multiresfinetuned5296.pb -cs 300 -i $VALIDATE_FILE -o $OUTPUT_DIR


