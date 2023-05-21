import torch
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torch.nn.functional as F
import utils
from torch.nn.functional import threshold, normalize

import sys
sys.path.append("..")
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide

# fix seeds
torch.manual_seed(0)
np.random.seed(0)


# load model
model_type = "vit_h" # ViT-h seems to work best
sam = sam_model_registry[model_type](checkpoint="/mnt/shared/lswezel/weights/sam_vit_h_4b8939.pth") # replace with your own path


device = "cuda:1" if torch.cuda.is_available() else "cpu"
sam.to(device=device)


optimizer = torch.optim.Adam(sam.mask_decoder.parameters(), lr=1e-4)
loss_fn = torch.nn.MSELoss()


mask_transform = transforms.Compose([
                    transforms.ToTensor(),
                    utils.ThresholdTransform(threshold=1e-5),
            ])

preprocess_transform = ResizeLongestSide(sam.image_encoder.img_size)


# training looop
num_epochs = 1
for epoch in range(num_epochs):

    epoch_losses = []

    for _ in range(5): # dummy loop that serves as placeholder until we have proper dataloaders

        # simulate sampling from dataloader
        image = cv2.imread("example_img.jpg")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        gt_mask = Image.open("example_mask.png")
        gt_mask = mask_transform(gt_mask)[0].unsqueeze(0).to(device=device)


        # simulate prompt by sampling random points from mask
        random_nok_points = utils.sample_points(gt_mask, class_index=1, max_num_samples=5)
        # process samples points into single array for SAM predictor
        point_coords = np.array(random_nok_points)
        # provide labels for points
        point_labels = np.array([1 for _ in range(len(random_nok_points))])

        # preprocess inputs for SAM
        original_image_size = image.shape[:2]

        point_coords = preprocess_transform.apply_coords(point_coords, original_image_size)
        coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=device)
        labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=device)
        coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]

        input_image = preprocess_transform.apply_image(image)
        input_image_torch = torch.as_tensor(input_image, device=device)
        input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]
        
        input_size = tuple(input_image_torch.shape[-2:])
        input_image = sam.preprocess(input_image_torch)

        # extract image embeddings
        with torch.no_grad():
            # encode image
            image_embedding = sam.image_encoder(input_image)
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

        # up-sample masks so we can compare then to the ground truth
        upscaled_masks = sam.postprocess_masks(low_res_masks, input_size, original_image_size).to(device)
        binary_mask = normalize(threshold(upscaled_masks, 0.0, 0))

        # compute loss
        loss = loss_fn(binary_mask, gt_mask.unsqueeze(0))

        # backpropagate loss and update mask decoder weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_losses.append(loss.item())

    print(f"Epoch {epoch} loss: {np.mean(epoch_losses):.2f}")

