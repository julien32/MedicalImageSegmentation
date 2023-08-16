# Getting started
You can have a first interaction with SAM by running `get_started_with_SAM.py`. This will load an example image, simulate a visual prompt by the user, and will make a prediction. You can install SAM through `pip` with
```
pip install git+https://github.com/facebookresearch/segment-anything.git
```
and download the [pre-trained weights for the ViT-b model](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth). Then, make sure to replace the path to the weights in `get_started_with_SAM.py`. Other than that, no special libraries are needed thus far.
If this:
```
import torch
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
from PIL import Image
import torchvision.transforms as transforms
import sys
```
runs, you should almost be ready to go.

Complete the setup by changing paths to your local ones. Paths to change are in the files:
```
onnx_inference.py - Path to the checkpoint.
onnx_inference.py - Path to prediction location save folder.
views.py          - Path to gallery image input locations.
views.py          - Path to gallery image input locations.
views.py          - Path to run_inference.sh script location.
views.py          - Path to annotation_image_data.csv file location.
run_inference.py  - Path to onnx_inference.py location.
run_inference.py  - Path to sam_finetuned.onnx checkpoint location
run_inference.py  - Path to annotation_image_data.csv location. 
```

SAM should now be able to be run successfully now. 

Commands to run Website:

```
run virtual environment (dir: ".../frontendePrototype")   - `venv\Scripts\activate` <br />
migrate (dir: ".../prototypeSite")                        - `python manage.py makemigrations` <br />
migrate (dir: ".../prototypeSite")                        - `manage.py migrate` <br />
create admin user (dir: ".../prototypeSite")              - `py manage.py createsuperuser` <br />
clear DB (dir: ".../prototypeSite")                       - `python manage.py flush` <br />
start server (dir: ".../prototypeSite")                   - `python manage.py runserver` <br />
```
