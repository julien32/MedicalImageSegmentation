#!/bin/bash

echo "Running segmentation. Please wait."

Python "C:/Users/danie/Desktop/Master/Master SoSe 2023/Machine Learning in Graphics, Vision and Language/GithubTeamCode/onnx_inference.py" --onnx-checkpoint "C:/Users/danie/Desktop/Master/Master SoSe 2023/Machine Learning in Graphics, Vision and Language/GithubTeamCode/sam_msefocal.onnx" --input_df "C:/Users/danie/Desktop/Master/Master SoSe 2023/Machine Learning in Graphics, Vision and Language/GithubTeamCode/annotation_image_data.csv"