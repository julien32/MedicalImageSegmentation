import numpy as np
import torch
import torchvision
import torch.nn.functional as F
from torch.nn.functional import threshold, normalize
from torchvision.models._utils import IntermediateLayerGetter
import albumentations as A
import argparse

import utils
import metrics
from datasets import PromptableMetaDataset

import sys
sys.path.append("..")
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
import os
import time



model_type = "vit_b" # This is the smallest version
sam = sam_model_registry[model_type](checkpoint="/home/stefan/Downloads/sam_vit_b_01ec64.pth").eval().cpu() # replace with your own path
preprocess_transform = transform = ResizeLongestSide(sam.image_encoder.img_size)



train_dataset = PromptableMetaDataset(
                                    "/home/stefan/datasets/mri_seg",
                                    [
                                    'TCGA_CS_4941_19960909',
                                    ],)


train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4, drop_last=True) 


num_trials = 1

latencies = []

for _ in range(num_trials):
    image, gt_mask = next(iter(train_loader))
    input_image_torch, gt_mask, coords_torch, labels_torch, input_size, original_image_size = utils.prepare_sam_inputs(
        image,
        gt_mask,
        preprocess_transform,
        "cpu",
                                )

    input_image = sam.preprocess(input_image_torch)

    start_time = time.time()

    with torch.no_grad():

        # encode image
        image_embedding = sam.image_encoder(input_image)
        # image_embedding = efficientnet(input_image)["layer5"]
        image_embedding = F.pad(image_embedding, (0, 0, 0, 0, 0, 256 - image_embedding.shape[1])) # pad along channel dimension

        # encode prompt
        sparse_embeddings, dense_embeddings = sam.prompt_encoder(
            points=(coords_torch, labels_torch),
            boxes=None,
            masks=None,
        )
        # sparse_embeddings, dense_embeddings = sparse_embeddings.to(PRECISION), dense_embeddings.to(PRECISION)
        sparse_embeddings, dense_embeddings = sparse_embeddings, dense_embeddings


        # build masks given embeddings and prompt
        low_res_masks, iou_predictions = sam.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
        # upscale masks so we can compare then to the ground truth
        upscaled_masks = sam.postprocess_masks(low_res_masks, input_size, original_image_size).to(device)
        binary_mask = normalize(threshold(upscaled_masks, 0, 0))
        
        # store predictions and ground truth to compute metrics later
        predicted_masks.append(binary_mask.cpu().numpy())
        ground_truth_masks.append(gt_mask.cpu().numpy())

        latencies.append(time.time() - start_time)

print(f"Mean latency: {np.mean(latencies):.4f} ± {np.std(latencies):.4f} s")