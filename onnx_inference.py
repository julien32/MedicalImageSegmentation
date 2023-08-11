import os
import numpy as np
import cv2
from segment_anything import sam_model_registry, SamPredictor
import onnxruntime
import argparse
import utils
import pandas as pd
import matplotlib.pyplot as plt
import utils
import re

checkpoint = "C:/Users/danie/Desktop/Master/Master SoSe 2023/Machine Learning in Graphics, Vision and Language/GithubTeamCode/sam_vit_b_01ec64.pth"

# Location of predictions on local machine
prediction_location = "C:/Users/danie/Desktop/Master/Master SoSe 2023/Machine Learning in Graphics, Vision and Language/GithubTeamCode/frontendPrototype/prototypeSite/prediction_media"

parser = argparse.ArgumentParser("Run onnxruntime inference sess")
parser.add_argument("--onnx-checkpoint",
                    type=str,
                    required=True,
                    help="The path to the SAM model checkpoint.")
parser.add_argument(
    "--input_df",
    type=str,
    required=True,
    help=
    "The path to the input dataframe, where all annotated paths and prompts are stored."
)

args = parser.parse_args()

user_input = pd.read_csv(args.input_df)

image_paths = user_input['filepath'].tolist()
print("Paths: ", image_paths)
prompts_y = user_input['y'].tolist()
print("Y coords: ", prompts_y)
prompts_x = user_input['x'].tolist()
print("X coords: ", prompts_x)

assert len(prompts_y) == len(prompts_x) == len(
    image_paths)  # make sure there is a prompt for each image (and vice versa)

# initialize onnx runtime session
ort_session = onnxruntime.InferenceSession(args.onnx_checkpoint)

# load model
model_type = "vit_b"
sam = sam_model_registry[model_type](checkpoint=checkpoint)
sam.to(device='cpu')
predictor = SamPredictor(sam)

plot_imgs = []
plot_masks = []
plot_masks_raw = []
plot_points = []
plot_labels = []
plot_paths = []

for i, img_path in enumerate(image_paths):
    plot_paths.append(img_path)

    # iterate over images
    # load image
    print("image path: ", img_path)
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    plot_imgs.append(image.copy())

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
    plot_points.append(input_point.copy())

    # assign label
    input_label = np.array([1])
    plot_labels.append(input_label.copy())

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
    normalized_masks = (masks - masks.min()) / (masks.max() - masks.min())
    normalized_masks *= 255
    normalized_masks = normalized_masks.astype(np.uint8)
<<<<<<< HEAD
    # Convert to numpy and save 
    plot_masks_raw.append(normalized_masks[0][0])


    

=======
    # Convert to numpy and save
    plot_masks_raw.append(normalized_masks[0][0])
>>>>>>> scriptImplementationChatGPTApproach
    masks = masks > predictor.model.mask_threshold

    # store predicted mask to be shown later
    plot_masks.append(masks[0])

regex_pattern = r"([^\\]+)$"
resultString = ""
maskString = ""
# save predictions
for j in range(len(plot_imgs)):
    fig, ax = plt.subplots()
    ax.imshow(plot_imgs[j])
    utils.show_mask(plot_masks[j], ax)
    utils.show_points(plot_points[j], plot_labels[j], ax)
    ax.axis('off')
    match = re.search(regex_pattern, plot_paths[j])
    if match:
        resultString = match.group(1)

    print("Image Names: ", resultString)
    plt.savefig(os.path.join(
        prediction_location,
        f"prediction_{resultString}"),
        bbox_inches='tight',
        pad_inches=0)

# save masks
for j in range(len(plot_masks_raw)):
    fig, ax = plt.subplots()
    ax.imshow(plot_masks_raw[j])
    ax.axis('off')
    match = re.search(regex_pattern, plot_paths[j])
    if match:
        maskString = match.group(1)
        print("Mask Names: ", maskString)

    plt.savefig(os.path.join(
        prediction_location,
        f"mask_{maskString}"),
                bbox_inches='tight',
                pad_inches=0)


# save masks
for j in range(len(plot_masks_raw)):
    fig, ax = plt.subplots()
    ax.imshow(plot_masks_raw[j])
    ax.axis('off')
    plt.savefig(os.path.join("predictions",
                             f"mask_{plot_paths[j].split('/')[-1]}"),
                bbox_inches='tight',
                pad_inches=0)
