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
runs, you should be fine and ready to go.