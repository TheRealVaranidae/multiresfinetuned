#!/bin/bash

set -euxo pipefail

VALIDATE_FILE=$1
OUTPUT_DIR=$2

OUTPUT_FILE = ${OUTPUT_DIR}/$(basename $VALIDATE_FILE).txt

python create_detections_1.py -c ./model.pb --input $VALIDATE_FILE --output $OUTPUT_FILE
