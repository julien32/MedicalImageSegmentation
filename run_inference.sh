#!/bin/bash

python onnx_inference.py --onnx-checkpoint /mnt/shared/lswezel/checkpoints/sam_full2.onnx --imagepath example_img.jpg example_img.jpg --prompts-y 100 100 --prompts-x 375 375