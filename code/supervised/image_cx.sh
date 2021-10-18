#!/bin/bash

# Author: Yinghao Li @ Georgia Tech ECE
# All rights reserved

# -------------------------------------
# This script is used to run image_cx.py
# -------------------------------------

# Quit if there are any errors
set -e

TRAIN_DIR="../../data/train"
TEST_DIR="../../data/test"
OUTPUT_DIR="./results/"
# OUTPUT_DIR="./output"
BATCH_SIZE=128
TRAIN_EPOCHS=20
VALID_TOLERANCE=5
LEARNING_RATE=0.001
SEED=0

CUDA_VISIBLE_DEVICES=$1 python image_cx.py \
    --train_dir $TRAIN_DIR \
    --test_dir $TEST_DIR \
    --output_dir $OUTPUT_DIR \
    --batch_size $BATCH_SIZE \
    --num_train_epochs $TRAIN_EPOCHS \
    --num_valid_tolerance $VALID_TOLERANCE \
    --learning_rate $LEARNING_RATE \
    --save_processed_data
