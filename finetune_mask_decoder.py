import numpy as np
import torch
import torchvision
import torch.nn.functional as F
from torch.nn.functional import threshold, normalize
from torchvision.models._utils import IntermediateLayerGetter
import albumentations as A

import utils
from datasets import PromptableMetaDataset

import sys
sys.path.append("..")
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide

torch.manual_seed(0)
np.random.seed(0)

device = "cuda:1" if torch.cuda.is_available() else "cpu"

# load model
# model_type = "vit_h" # ViT-h seems to work best
# sam = sam_model_registry[model_type](checkpoint="/mnt/shared/lswezel/weights/sam_vit_h_4b8939.pth") # replace with your own path
model_type = "vit_b" # ViT-h seems to work best
sam = sam_model_registry[model_type](checkpoint="/mnt/shared/lswezel/weights/sam_vit_b_01ec64.pth") # replace with your own path

# alternative backbones
efficientnet = IntermediateLayerGetter(torchvision.models.efficientnet_b0(pretrained=True).features, return_layers={"5":"layer5"}).to(device=device).eval()


sam.to(device=device)


optimizer = torch.optim.Adam(sam.mask_decoder.parameters(), lr=1e-4, weight_decay=0)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

# three different loss functions that I'm experimenting with
mse_loss = torch.nn.MSELoss()
focal_loss = utils.FocalLoss(logits=True)
dice_loss = utils.DiceLoss()


preprocess_transform = transform = ResizeLongestSide(sam.image_encoder.img_size)

augmentations = A.Compose([
        # Spatial augmentations
        A.HorizontalFlip(p=0.2),
        # A.augmentations.geometric.rotate.SafeRotate(limit=20),
        A.VerticalFlip(p=0.2),
        # Texture augmentations
        A.RandomBrightnessContrast(p=0.2),
        # A.GaussianBlur(p=0.1),
        A.CLAHE(p=0.2),
        # A.ColorJitter(p=0.1),
        # A.ChannelDropout(p=0.1),
    ])


dataset = PromptableMetaDataset([
                                "...",
                                ],
                                transforms=augmentations
                            )

print(f"Training sample size: {len(dataset)}")

train_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0, drop_last=True) 


num_epochs = 30
for epoch in range(num_epochs):
    epoch_losses = []
    for batch_idx, (image, gt_mask) in enumerate(train_loader):
        
        # process image, mask and simulate prompts
        input_image_torch, gt_mask, coords_torch, labels_torch, input_size, original_image_size = \
            utils.prepare_sam_inputs(
                                    image,
                                    gt_mask,
                                    preprocess_transform,
                                    device,
                                    )

        input_image = sam.preprocess(input_image_torch)
        
        # extract image embeddings
        with torch.no_grad():

            # encode image
            # image_embedding = sam.image_encoder(input_image)
            image_embedding = efficientnet(input_image)["layer5"]
            image_embedding = F.pad(image_embedding, (0, 0, 0, 0, 0, 256 - image_embedding.shape[1])) # pad along channel dimension

            # encode prompt
            sparse_embeddings, dense_embeddings = sam.prompt_encoder(
                points=(coords_torch, labels_torch),
                boxes=None,
                masks=None,
            )

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

        # compute loss
        loss = focal_loss(upscaled_masks, gt_mask) + 0.1*mse_loss(upscaled_masks, gt_mask)

        # backpropagate loss and update mask decoder weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_losses.append(loss.item())
    
    scheduler.step()

    print(f"Epoch {epoch} loss: {np.mean(epoch_losses):.4f}")
    # TODO evaluation loop

