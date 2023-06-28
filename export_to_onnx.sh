#!/bin/bash

python segment_anything/export_to_onnx.py --checkpoint /mnt/shared/lswezel/checkpoints/sam_finetuned.pth --output /mnt/shared/lswezel/checkpoints/sam_finetuned.onnx --model-type vit_b --return-single-mask --opset 14