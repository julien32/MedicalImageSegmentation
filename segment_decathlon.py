import torch
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
from PIL import Image
import torchvision.transforms as transforms
import sys
sys.path.append("..")
from segment_anything import sam_model_registry, SamPredictor
from datasets import Decathlon_Heart
import imageio


def sample_points(mask, class_index=0, max_num_samples=5):
    # find coordinates that fulfill class condition
    if not isinstance(mask, torch.Tensor):
        mask = torch.tensor(mask)
    indices = torch.argwhere(mask[0]==class_index) # this might be extended to multi-class setup by using mask[class_index]
    # limit number of sample points
    num_samples = min(max_num_samples, indices.shape[0])
    # sample random indices 
    indices = indices[torch.randperm(indices.shape[0])[:num_samples]]
    # return as list
    return indices.tolist()


def SAM_prediction(input_image, gt_mask, weight_path=None, plot_filename='example', model_type = "vit_b", max_num_samples=1):
    if weight_path is None or not os.path.exists(weight_path):
        raise Exception("Download weights to weight_path from https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sam = sam_model_registry[model_type](checkpoint=weight_path) 
    sam.to(device=device)
    predictor = SamPredictor(sam)
    predictor.set_image(input_image.astype(np.uint8))
    random_points = sample_points(gt_mask, class_index=1, max_num_samples=max_num_samples)
    input_points = np.array(random_points)
    input_labels = np.array([1 for _ in range(len(random_points))])
    masks, scores, logits = predictor.predict(
        point_coords=input_points,
        point_labels=input_labels,
        multimask_output=True,
    )
    image_prediction = np.tile(input_image, (1, 2, 1))
    mask = masks[2] == 1.0
    image_prediction[:1024, :1024, :][mask] = [255, 0, 0]
    mask = np.squeeze(gt_mask, axis=0) == 1.0
    image_prediction[:1024, 1024:2048, :][mask] = [255, 0, 0]
    for point in random_points:
        image_prediction[point[0]-5:point[0]+5, point[1]+1024-5:point[1]+1024+5, :] = [0, 255, 0]
    imageio.imsave(plot_filename, image_prediction)

########################################
#### Prediction for Decathlon Heart ####
########################################
decathlon_heart_dataloader = Decathlon_Heart(source='datasets/decathlon/heart', classname='decathlonheart')
#decathlon_heart_dataloader.visualize(0)
for i, (image, mask) in enumerate(decathlon_heart_dataloader):
    print(f"Predict {i}")
    SAM_prediction(image.numpy(), mask.numpy(), weight_path='/local/biermaie/segment_anything/sam_vit_b_01ec64.pth', plot_filename=f'plots/decathlon/Task02_Heart/{i}.png')
    exit()
