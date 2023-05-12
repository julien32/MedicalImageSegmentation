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



def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=50):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 1], pos_points[:, 0], color='red', marker='x', s=marker_size)
    ax.scatter(neg_points[:, 1], neg_points[:, 0], color='green', marker='x', s=marker_size)   
    


# load model
model_type = "vit_b"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sam = sam_model_registry[model_type](checkpoint="/mnt/shared/lswezel/weights/sam_vit_b_01ec64.pth") # make sure to update path here
# -> weights can be downloaded here: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
sam.to(device=device)
predictor = SamPredictor(sam)


# If you want to have a look at some modules:
# print(sam.prompt_encoder) # here, we will only need to look at the encoder for points
# print(sam.mask_decoder) # this is what we probably will need to fine-tune first


# load and process image and mask
image = cv2.imread("example_img.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
gt_mask = Image.open("example_mask.png")


# process mask
mask_transform = transforms.Compose([
                    transforms.ToTensor(),
                    ThresholdTransform(threshold=1e-5),
            ])
gt_mask = mask_transform(gt_mask)




# Inference
predictor.set_image(image)

# sample random points within the ground truth mask (this simulates user input)
random_points = sample_points(gt_mask, class_index=1, max_num_samples=20)
# process samples points into single array for SAM predictor (if more sample points are given by the user, concatenate them to this array)
input_points = np.array(random_points)
# provide labels for points
input_labels = np.array([1 for _ in range(len(random_points))])


# let SAM predict
masks, scores, logits = predictor.predict(
    point_coords=input_points,
    point_labels=input_labels,
    multimask_output=True,
)


# SAM predicts a 3-level hierarchy of masks, so we iterate over them (mask 3 is probably the most relevant for us)
for i, (mask, score) in enumerate(zip(masks, scores)): # plot the output and save it
    fig, ax = plt.subplots(1,2, figsize=(10,10))
    ax[0].imshow(image)
    ax[1].imshow(gt_mask[0])
    show_mask(mask, ax[0])
    show_points(input_points, input_labels, ax[1])
    ax[0].set_title(f"Mask {i+1}, Score: {score:.3f}", fontsize=10)
    ax[1].set_title("Ground truth mask", fontsize=10)
    ax[0].axis('off')
    ax[1].axis('off')
    plt.savefig(f'example_prediction_{i}.png', bbox_inches='tight', pad_inches=0)

    plt.show()  


