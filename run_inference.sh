#!/bin/bash

echo "running 1"

Python "C:/Users/danie/Desktop/Master/Master SoSe 2023/Machine Learning in Graphics, Vision and Language/GithubTeamCode/onnx_inference.py" --onnx-checkpoint "C:/Users/danie/Desktop/Master/Master SoSe 2023/Machine Learning in Graphics, Vision and Language/GithubTeamCode/sam_finetuned.onnx" --input_df "C:/Users/danie/Desktop/Master/Master SoSe 2023/Machine Learning in Graphics, Vision and Language/GithubTeamCode/example_user_input.csv"

echo "running"

$SHELL