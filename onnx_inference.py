import os
import numpy as np
import cv2
from segment_anything import sam_model_registry, SamPredictor
import onnxruntime
import argparse
import utils

parser = argparse.ArgumentParser("Run onnxruntime inference sess")
parser.add_argument("--onnx-checkpoint",
                    type=str,
                    required=True,
                    help="The path to the SAM model checkpoint.")
parser.add_argument("--imagepaths",
                    type=str,
                    nargs='+',
                    required=True,
                    help="The image you want to run inference on.")
parser.add_argument("--prompts-y",
                    type=int,
                    nargs='+',
                    help="The y coords of the prompt.")
parser.add_argument("--prompts-x",
                    type=int,
                    nargs='+',
                    help="The x coords of the prompt.")
args = parser.parse_args()

image_paths = args.imagepaths

prompts_y = args.prompts_y
prompts_x = args.prompts_x

assert len(prompts_y) == len(prompts_x) == len(
    image_paths)  # make sure there is a prompt for each image (and vice versa)

# initialize onnx runtime session
ort_session = onnxruntime.InferenceSession(args.onnx_checkpoint)

# load model
checkpoint = "/mnt/shared/lswezel/checkpoints/sam_full2.pth"
model_type = "vit_b"
sam = sam_model_registry[model_type](checkpoint=checkpoint)
sam.to(device='cpu')
predictor = SamPredictor(sam)

for i, img_path in enumerate(image_paths):

    # iterate over image dir
    # img_path = args.imagepath
    # load image
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # process input for SAM
    input_image_torch, input_size, original_image_size = \
                    utils.prepare_sam_inputs_for_inference(
                                            image,
                                            )
    image = sam.preprocess(input_image_torch)

    # get prompt coords
    y_coord = prompts_y[i]
    x_coord = prompts_x[i]
    input_point = np.array([[y_coord, x_coord]])

    # assign label
    input_label = np.array([1])

    # convert coords to onnx compatible format
    onnx_coord = np.concatenate(
        [input_point, np.array([[0.0, 0.0]])], axis=0)[None, :, :]
    onnx_label = np.concatenate([input_label, np.array([-1])],
                                axis=0)[None, :].astype(np.float32)
    onnx_coord = predictor.transform.apply_coords(
        onnx_coord, image.shape[:2]).astype(np.float32)

    # empty mask prompt and indicate no mask
    onnx_mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
    onnx_has_mask_input = np.zeros(1, dtype=np.float32)

    # collect input image and prompts into dict
    ort_inputs = {
        "images": image.numpy(),
        "point_coords": onnx_coord,
        "point_labels": onnx_label,
        "mask_input": onnx_mask_input,
        "has_mask_input": onnx_has_mask_input,
        "orig_im_size": np.array(original_image_size, dtype=np.float32)
    }

    # run inference
    masks, _, low_res_logits = ort_session.run(None, ort_inputs)
    masks = masks > predictor.model.mask_threshold

    # for debugging... see if correct shape is returned
    print(masks.shape)
