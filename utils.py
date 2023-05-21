import numpy as np
import torch
from torchvision import transforms
import os
import matplotlib.pyplot as plt

class ThresholdTransform(object):
  def __init__(self, threshold):
    self.threshold = threshold  # input threshold for [0..255] gray level, convert to [0..1]

  def __call__(self, x):
    return (x > self.threshold).to(x.dtype)
  

def sample_points(mask, class_index=0, max_num_samples=5):
    # find coordinates that fulfill class condition
    indices = torch.argwhere(mask[0]==class_index) # this might be extended to multi-class setup by using mask[class_index]
    # limit number of sample points
    num_samples = min(max_num_samples, indices.shape[0])
    # sample random indices 
    indices = indices[torch.randperm(indices.shape[0])[:num_samples]]
    # return as list
    return indices.tolist()


def show_points(coords, labels, ax, marker_size=50):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 1], pos_points[:, 0], color='red', marker='x', s=marker_size)
    ax.scatter(neg_points[:, 1], neg_points[:, 0], color='green', marker='x', s=marker_size)   
     

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def plot_prediction(prediction, ground_truth, input_image, input_points, input_labels, filename=""):
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    # if input_image.shape[0] == 1:
    #     input_image = np.stack([input_image[0]]*3, axis=-1)
    ax[0].imshow(input_image)
    show_points(input_points, input_labels, ax[1])

    ax[1].imshow(ground_truth[0])
    show_mask(prediction, ax[0])
    show_points(input_points, input_labels, ax[1])
    # ax[0].set_title(f"Mask {i+1}, Score: {score:.3f}", fontsize=10)
    ax[1].set_title("Ground truth mask", fontsize=10)
    ax[0].axis('off')
    ax[1].axis('off')
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)

    # plt.show()  