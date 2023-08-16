import numpy as np
import cv2
import torch
from torchvision import transforms
import os
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage.morphology import distance_transform_edt

import sys
sys.path.append("..")




class ThresholdTransform(object):
  def __init__(self, threshold):
    self.threshold = threshold  # input threshold for [0..255] gray level, convert to [0..1]

  def __call__(self, x):
    return (x > self.threshold).to(x.dtype)
  

def sample_random_points(masks, class_index=0, max_num_samples=5):
    # find coordinates that fulfill class condition
    all_coords = []
    batch_size = masks.shape[0]
    for batch_index in range(batch_size):
        indices = torch.argwhere(masks[batch_index][0]==class_index)
        num_samples = min(max_num_samples, indices.shape[0])
        indices = indices[torch.randperm(indices.shape[0])[:num_samples]]
        all_coords.append(indices.tolist())

    return all_coords


def _draw_contour_on_mask(size, cnt, color:int = 255):
    mask = np.zeros(size, dtype='uint8')
    mask = cv2.drawContours(mask, [cnt], -1, color, -1)
    return mask


def _get_furthest_point_from_edge(cnt, size):
    mask = _draw_contour_on_mask(size, cnt)
    d = distance_transform_edt(mask)
    cy, cx = np.unravel_index(d.argmax(), d.shape)
    return cx, cy


def sample_ffe_points(masks):
    batch_size = masks.shape[0]
    batch_coords = []
    min_num_points = np.inf
    for batch_index in range(batch_size):
        mask = masks[batch_index]
        mask = mask.cpu().numpy()[0].astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        coords = []
        for contour in contours:
            x,y = _get_furthest_point_from_edge(contour, mask.shape)
            if mask[y, x] == 1:
                coords.append([y,x])
        if len(coords) < min_num_points:
            min_num_points = len(coords)
        batch_coords.append(coords)
    for batch_index in range(batch_size):
        batch_coords[batch_index] = batch_coords[batch_index][:min_num_points]

    batch_coords = np.array(batch_coords)

    return batch_coords


def show_points(coords, labels, ax, marker_size=50):
    pos_points = coords[labels==1]
    # neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 1], pos_points[:, 0], color='red', marker='x', s=marker_size)
    # ax.scatter(neg_points[:, 1], neg_points[:, 0], color='green', marker='x', s=marker_size)   
     

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    def forward(self, pred, target):
        """This definition generalize to real valued pred and target vector.
        This should be differentiable.
        pred: tensor with first dimension as batch
        target: tensor with first dimension as batch
        """
        smooth = self.smooth

        # have to use contiguous since they may from a torch.view op
        iflat = pred.contiguous().view(-1)
        tflat = target.contiguous().view(-1)
        intersection = (iflat * tflat).sum()

        # A_sum = torch.sum(tflat * iflat)
        A_sum = torch.sum(iflat * iflat) 
        B_sum = torch.sum(tflat * tflat)
        
        return 1 - ((2. * intersection + smooth) / (A_sum + B_sum + smooth) )





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
    # ax[0].axis('off')
    # ax[1].axis('off')
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)


def plot_prompt( ground_truth, input_image, input_points, input_labels, save_dir="predictions", filename=""):
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    # if input_image.shape[0] == 1:
    #     input_image = np.stack([input_image[0]]*3, axis=-1)
    ax[0].imshow(input_image)
    show_points(input_points, input_labels, ax[1])

    ax[1].imshow(ground_truth[0])
    show_points(input_points, input_labels, ax[1])
    # ax[0].set_title(f"Mask {i+1}, Score: {score:.3f}", fontsize=10)
    ax[1].set_title("Ground truth mask", fontsize=10)
<<<<<<< HEAD
    # ax[0].axis('off')
    # ax[1].axis('off')
=======
    ax[0].axis('off')
    ax[1].axis('off')
>>>>>>> scriptImplementationChatGPTApproach
    plt.savefig(os.path.join(save_dir, filename), bbox_inches='tight', pad_inches=0)


def prepare_sam_inputs(image, gt_mask, preprocess_transform, device):
        point_coords = sample_ffe_points(gt_mask)
        gt_mask = gt_mask.to(device)
        # provide labels for points
        point_labels = np.array([[1]*point_coords[i].shape[0] for i in range(point_coords.shape[0])])
        original_image_size = image.shape[1:3]
        # preprocess prompts
        point_coords = preprocess_transform.apply_coords(point_coords, original_image_size)
        coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=device)
        labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=device)
        # process image
        input_image_torch = torch.as_tensor(image, device=device)
        input_image_torch = input_image_torch.permute(0, 3, 1, 2).contiguous()
        input_size = tuple(input_image_torch.shape[-2:])

        return input_image_torch, gt_mask, coords_torch, labels_torch, input_size, original_image_size




def prepare_sam_inputs_for_inference(image, device="cpu"):
        original_image_size = image.shape[:2]
        input_image_torch = torch.as_tensor(image, device=device)
        input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()
        input_size = tuple(input_image_torch.shape[-2:])

        return input_image_torch, input_size, original_image_size


def imagenet_standardize(x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        pixel_mean = [123.675, 116.28, 103.53],
        pixel_std = [58.395, 57.12, 57.375],
        x = (x - pixel_mean) / pixel_std

<<<<<<< HEAD

        return x

def overlay_mask(img, mask):
    mask = mask.astype(np.float32)
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    mask = mask*255# / 255.0
    img = img.astype(np.float32)
    img = img# / 255.0
    # Blend img with mask
    img = cv2.addWeighted(img, 0.5, mask, 0.5, 0.0)
    # img = img + 0.5 * mask
    return img
=======
        # # Pad
        # h, w = x.shape[-2:]
        # padh = self.image_encoder.img_size - h
        # padw = self.image_encoder.img_size - w
        # x = F.pad(x, (0, padw, 0, padh))
        return x

>>>>>>> scriptImplementationChatGPTApproach

