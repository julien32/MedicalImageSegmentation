import numpy as np
import torch
import torchvision
import torch.nn.functional as F
from torch.nn.functional import threshold, normalize
from torchvision.models._utils import IntermediateLayerGetter
# import albumentations as A
import argparse

import utils
import metrics
from datasets import PromptableMetaDataset

import sys
sys.path.append("..")
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
import os


if __name__ == "__main__":

    parser = argparse.ArgumentParser()


    parser.add_argument('--data_src', default="/mnt/qb/work/akata/aoq833/sam_finetuning_data")
    parser.add_argument('--checkpoint_path', default="/mnt/qb/work/akata/aoq833/checkpoints")      # option that takes a value
    parser.add_argument('--checkpoint_loadname', default="sam2.pth")      # option that takes a value
    parser.add_argument('--checkpoint_savename', default="sam3.pth")      # option that takes a value
    parser.add_argument('--training_sets', type=list, default=['TCGA_CS_4941_19960909'])      # option that takes a value
    parser.add_argument('--test_sets', type=list, default=['TCGA_HT_A61B_19991127'])      # option that takes a value
    parser.add_argument('--num_epochs', type=int, default=20)      # option that takes a value
    parser.add_argument('--batch_size', type=int, default=1)      # option that takes a value
    parser.add_argument('--lr', type=float, default=1e-4)      # option that takes a value


    args = parser.parse_args()


    torch.manual_seed(0)
    np.random.seed(0)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    PRECISION = torch.float32
    # PRECISION = torch.float16


    # load model
    # model_type = "vit_h" # ViT-h seems to work best
    # sam = sam_model_registry[model_type](checkpoint="/mnt/shared/lswezel/weights/sam_vit_h_4b8939.pth") # replace with your own path
    model_type = "vit_b" # ViT-h seems to work best
    sam = sam_model_registry[model_type](checkpoint=f"{args.checkpoint_path}/sam_vit_b_01ec64.pth") # replace with your own path
    # alternative backbones
    efficientnet = IntermediateLayerGetter(torchvision.models.efficientnet_b0(pretrained=True).features, return_layers={"5":"layer5"}).to(device=device).to(PRECISION).eval()



    if os.path.exists(f"{args.checkpoint_path}/{args.checkpoint_loadname}"):
        print("Loading checkpoint from file...")
        sam.load_state_dict(torch.load(f"{args.checkpoint_path}/{args.checkpoint_loadname}"))


    sam.to(device=device).to(PRECISION)
    sam.prompt_encoder.pe_layer.precision = PRECISION


    optimizer = torch.optim.Adam(sam.mask_decoder.parameters(), lr=1e-4, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    # grad scaler for mixed precision training
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    # three different loss functions that I'm experimenting with
    mse_loss = torch.nn.MSELoss()
    focal_loss = utils.FocalLoss(logits=True)
    dice_loss = utils.DiceLoss()

    preprocess_transform = transform = ResizeLongestSide(sam.image_encoder.img_size)

    # augmentations = A.Compose([
    #         # Spatial augmentations
    #         A.HorizontalFlip(p=0.2),
    #         # A.augmentations.geometric.rotate.SafeRotate(limit=20),
    #         A.VerticalFlip(p=0.2),
    #         # Texture augmentations
    #         # A.RandomBrightnessContrast(p=0.2),
    #         # A.GaussianBlur(p=0.1),
    #         # A.CLAHE(p=0.2),
    #         # A.ColorJitter(p=0.1),
    #         # A.ChannelDropout(p=0.1),
    #     ])


    train_dataset = PromptableMetaDataset(
                                "/mnt/qb/work/akata/aoq833/sam_finetuning_data",
                                [
                                'malignant',
                                'TCGA_CS_4941_19960909',
                                'TCGA_CS_4942_19970222',
                                'TCGA_CS_4943_20000902',
                                'TCGA_CS_4944_20010208',
                                'TCGA_CS_5393_19990606',
                                'TCGA_CS_5395_19981004',
                                'TCGA_CS_5396_20010302',
                                'TCGA_CS_5397_20010315',
                                'TCGA_CS_6186_20000601',
                                'TCGA_CS_6188_20010812',
                                'TCGA_CS_6290_20000917',
                                'TCGA_CS_6665_20010817',
                                'TCGA_CS_6666_20011109',
                                'TCGA_CS_6667_20011105',
                                'TCGA_CS_6668_20011025',
                                'TCGA_CS_6669_20020102',
                                'TCGA_DU_5849_19950405',
                                'TCGA_DU_5851_19950428',
                                'TCGA_DU_5852_19950709',
                                'TCGA_DU_5853_19950823',
                                'TCGA_DU_5854_19951104',
                                'TCGA_DU_5855_19951217',
                                'TCGA_DU_5871_19941206',
                                'TCGA_DU_5872_19950223',
                                'TCGA_DU_5874_19950510',
                                'TCGA_DU_6399_19830416',
                                'TCGA_DU_6400_19830518',
                                'TCGA_DU_6401_19831001',
                                'TCGA_DU_6404_19850629',
                                'TCGA_DU_6405_19851005',
                                'TCGA_DU_6407_19860514',
                                'TCGA_DU_6408_19860521',
                                    ],
                                    # transforms=augmentations,
                                    precision=PRECISION,
                                )
    print(f"Training sample size: {len(train_dataset)}")
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=4, drop_last=True) 


    test_dataset = PromptableMetaDataset(
                                        "/mnt/qb/work/akata/aoq833/sam_finetuning_data",
                                    [
                                    'benign',
                                    'TCGA_HT_A61B_19991127',
                                    ],
                                    precision=PRECISION,
                                )

    print(f"Test sample size: {len(test_dataset)}")
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, drop_last=True)

    best_jaccard = 0
    num_epochs = 10
    for epoch in range(num_epochs):
        # Training
        epoch_losses = []    
        sam.prompt_encoder.train()
        sam.mask_decoder.train()
        for batch_idx, (image, gt_mask) in enumerate(train_loader):
            # TODO (!) find more elegant solution for empty masks (ideally sort out in dataset class)
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
            
            with torch.autocast(device_type='cuda', dtype=torch.float16):

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
                loss = focal_loss(upscaled_masks, gt_mask)# + mse_loss(upscaled_masks, gt_mask)

                # backpropagate loss and update mask decoder weights
                optimizer.zero_grad()
                # loss.backward()
                # optimizer.step()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()


            epoch_losses.append(loss.item())
        scheduler.step()

        print(f"\nEpoch {epoch} loss: {np.mean(epoch_losses):.4f}")


        # Evaluation
        predicted_masks = []
        ground_truth_masks = []

        sam.prompt_encoder.eval()
        sam.mask_decoder.eval()
        for batch_idx, (image, gt_mask) in enumerate(test_loader):
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


        # compute metrics
        results = metrics.compute_binary_segmentation_metrics(predicted_masks, ground_truth_masks)
        print(f"Epoch {epoch} results:")
        for k, v in results.items():
            print(f"{k}: {v:.4f}")
        print()

        # save checkpoint
        if results["jaccard_score"] > best_jaccard:
            best_jaccard = results["jaccard_score"]
            print(f"New best jaccard score: {best_jaccard:.4f}")
            torch.save(sam.state_dict(), f"{args.checkpoint_path}/{args.checkpoint_savename}")
        