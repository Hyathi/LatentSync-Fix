#!/bin/bash

uv run -m preprocess.data_processing_pipeline \
    --total_num_workers 4 \
    --per_gpu_num_workers 4 \
    --resolution 256 \
    --sync_conf_threshold 3 \
    --temp_dir temp \
    --input_dir /home/pawan/LatentSync-Fix/data/raw
