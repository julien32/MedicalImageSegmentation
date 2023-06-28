# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

# import sys
# sys.path.append(".")

from segment_anything.modeling import Sam
from segment_anything.utils.amg import calculate_stability_score
from typing import Tuple
from torch.nn import functional as F

from segment_anything import sam_model_registry
import torch.nn as nn
# from segment_anything.utils.onnx import SamOnnxModel
import utils
import argparse
import warnings
try:
    import onnxruntime  # type: ignore

    onnxruntime_exists = True
except ImportError:
    onnxruntime_exists = False


class CustomSamOnnxModel(nn.Module):
    """
    This model should not be called directly, but is used in ONNX export.
    It combines the prompt encoder, mask decoder, and mask postprocessing of Sam,
    with some functions modified to enable model tracing. Also supports extra
    options controlling what information. See the ONNX export script for details.
    """

    def __init__(
        self,
        model: Sam,
        return_single_mask: bool,
        use_stability_score: bool = False,
        return_extra_metrics: bool = False,
    ) -> None:
        super().__init__()
        self.mask_decoder = model.mask_decoder
        self.model = model
        self.img_size = model.image_encoder.img_size
        self.return_single_mask = return_single_mask
        self.use_stability_score = use_stability_score
        self.stability_score_offset = 1.0
        self.return_extra_metrics = return_extra_metrics

    @staticmethod
    def resize_longest_image_size(input_image_size: torch.Tensor,
                                  longest_side: int) -> torch.Tensor:
        input_image_size = input_image_size.to(torch.float32)
        scale = longest_side / torch.max(input_image_size)
        transformed_size = scale * input_image_size
        transformed_size = torch.floor(transformed_size + 0.5).to(torch.int64)
        return transformed_size

    def _embed_points(self, point_coords: torch.Tensor,
                      point_labels: torch.Tensor) -> torch.Tensor:
        point_coords = point_coords + 0.5
        point_coords = point_coords / self.img_size
        point_embedding = self.model.prompt_encoder.pe_layer._pe_encoding(
            point_coords)
        point_labels = point_labels.unsqueeze(-1).expand_as(point_embedding)

        point_embedding = point_embedding * (point_labels != -1)
        point_embedding = point_embedding + self.model.prompt_encoder.not_a_point_embed.weight * (
            point_labels == -1)

        for i in range(self.model.prompt_encoder.num_point_embeddings):
            point_embedding = point_embedding + self.model.prompt_encoder.point_embeddings[
                i].weight * (point_labels == i)

        return point_embedding

    def _embed_images(self, input_images: torch.Tensor) -> torch.Tensor:
        input_images = self.model.preprocess(input_images).unsqueeze(
            0)  # add batch dim
        return self.model.image_encoder(input_images)

    def _embed_masks(self, input_mask: torch.Tensor,
                     has_mask_input: torch.Tensor) -> torch.Tensor:
        mask_embedding = has_mask_input * self.model.prompt_encoder.mask_downscaling(
            input_mask)
        mask_embedding = mask_embedding + (
            1 - has_mask_input
        ) * self.model.prompt_encoder.no_mask_embed.weight.reshape(
            1, -1, 1, 1)
        return mask_embedding

    def mask_postprocessing(self, masks: torch.Tensor,
                            orig_im_size: torch.Tensor) -> torch.Tensor:
        masks = F.interpolate(
            masks,
            size=(self.img_size, self.img_size),
            mode="bilinear",
            align_corners=False,
        )

        prepadded_size = self.resize_longest_image_size(
            orig_im_size, self.img_size).to(torch.int64)
        masks = masks[..., :prepadded_size[0], :
                      prepadded_size[1]]  # type: ignore

        orig_im_size = orig_im_size.to(torch.int64)
        h, w = orig_im_size[0], orig_im_size[1]
        masks = F.interpolate(masks,
                              size=(h, w),
                              mode="bilinear",
                              align_corners=False)
        return masks

    def select_masks(self, masks: torch.Tensor, iou_preds: torch.Tensor,
                     num_points: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Determine if we should return the multiclick mask or not from the number of points.
        # The reweighting is used to avoid control flow.
        score_reweight = torch.tensor([
            [1000] + [0] * (self.model.mask_decoder.num_mask_tokens - 1)
        ]).to(iou_preds.device)
        score = iou_preds + (num_points - 2.5) * score_reweight
        best_idx = torch.argmax(score, dim=1)
        masks = masks[torch.arange(masks.shape[0]),
                      best_idx, :, :].unsqueeze(1)
        iou_preds = iou_preds[torch.arange(masks.shape[0]),
                              best_idx].unsqueeze(1)

        return masks, iou_preds

    @torch.no_grad()
    def forward(
        self,
        images: torch.Tensor,
        point_coords: torch.Tensor,
        point_labels: torch.Tensor,
        mask_input: torch.Tensor,
        has_mask_input: torch.Tensor,
        orig_im_size: torch.Tensor,
    ):
        image_embeddings = self._embed_images(images)
        sparse_embedding = self._embed_points(point_coords, point_labels)
        dense_embedding = self._embed_masks(mask_input, has_mask_input)

        masks, scores = self.model.mask_decoder.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=self.model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embedding,
            dense_prompt_embeddings=dense_embedding,
        )

        if self.use_stability_score:
            scores = calculate_stability_score(masks,
                                               self.model.mask_threshold,
                                               self.stability_score_offset)

        if self.return_single_mask:
            masks, scores = self.select_masks(masks, scores,
                                              point_coords.shape[1])

        upscaled_masks = self.mask_postprocessing(masks, orig_im_size)

        if self.return_extra_metrics:
            stability_scores = calculate_stability_score(
                upscaled_masks, self.model.mask_threshold,
                self.stability_score_offset)
            areas = (upscaled_masks
                     > self.model.mask_threshold).sum(-1).sum(-1)
            return upscaled_masks, scores, stability_scores, areas, masks

        return upscaled_masks, scores, masks


parser = argparse.ArgumentParser(
    description=
    "Export the SAM prompt encoder and mask decoder to an ONNX model.")

parser.add_argument("--checkpoint",
                    type=str,
                    required=True,
                    help="The path to the SAM model checkpoint.")

parser.add_argument("--output",
                    type=str,
                    required=True,
                    help="The filename to save the ONNX model to.")

parser.add_argument(
    "--model-type",
    type=str,
    required=True,
    help=
    "In ['default', 'vit_h', 'vit_l', 'vit_b']. Which type of SAM model to export.",
)

parser.add_argument(
    "--return-single-mask",
    action="store_true",
    help=("If true, the exported ONNX model will only return the best mask, "
          "instead of returning multiple masks. For high resolution images "
          "this can improve runtime when upscaling masks is expensive."),
)

parser.add_argument(
    "--opset",
    type=int,
    default=17,
    help="The ONNX opset version to use. Must be >=11",
)

parser.add_argument(
    "--quantize-out",
    type=str,
    default=None,
    help=
    ("If set, will quantize the model and save it with this name. "
     "Quantization is performed with quantize_dynamic from onnxruntime.quantization.quantize."
     ),
)

parser.add_argument(
    "--gelu-approximate",
    action="store_true",
    help=
    ("Replace GELU operations with approximations using tanh. Useful "
     "for some runtimes that have slow or unimplemented erf ops, used in GELU."
     ),
)

parser.add_argument(
    "--use-stability-score",
    action="store_true",
    help=(
        "Replaces the model's predicted mask quality score with the stability "
        "score calculated on the low resolution masks using an offset of 1.0. "
    ),
)

parser.add_argument(
    "--return-extra-metrics",
    action="store_true",
    help=(
        "The model will return five results: (masks, scores, stability_scores, "
        "areas, low_res_logits) instead of the usual three. This can be "
        "significantly slower for high resolution outputs."),
)


def run_export(
    model_type: str,
    checkpoint: str,
    output: str,
    opset: int,
    return_single_mask: bool,
    gelu_approximate: bool = False,
    use_stability_score: bool = False,
    return_extra_metrics=False,
):
    print("Loading model...")
    sam = sam_model_registry[model_type](checkpoint=checkpoint)

    onnx_model = CustomSamOnnxModel(
        model=sam,
        return_single_mask=return_single_mask,
        use_stability_score=use_stability_score,
        return_extra_metrics=return_extra_metrics,
    )

    # onnx_model = SamOnnxModel(
    #     model=sam,
    #     return_single_mask=return_single_mask,
    #     use_stability_score=use_stability_score,
    #     return_extra_metrics=return_extra_metrics,
    # )

    if gelu_approximate:
        for n, m in onnx_model.named_modules():
            if isinstance(m, torch.nn.GELU):
                m.approximate = "tanh"

    dynamic_axes = {
        "point_coords": {
            1: "num_points"
        },
        "point_labels": {
            1: "num_points"
        },
    }

    embed_dim = sam.prompt_encoder.embed_dim
    embed_size = sam.prompt_encoder.image_embedding_size
    mask_input_size = [4 * x for x in embed_size]
    dummy_inputs = {
        # "image_embeddings": torch.randn(1, embed_dim, *embed_size, dtype=torch.float),
        "images":
        torch.randn(3, 1024, 1024, dtype=torch.float),
        "point_coords":
        torch.randint(low=0, high=1024, size=(1, 5, 2), dtype=torch.float),
        "point_labels":
        torch.randint(low=0, high=4, size=(1, 5), dtype=torch.float),
        "mask_input":
        torch.randn(1, 1, *mask_input_size, dtype=torch.float),
        "has_mask_input":
        torch.tensor([1], dtype=torch.float),
        "orig_im_size":
        torch.tensor([1500, 2250], dtype=torch.float),
    }

    _ = onnx_model(**dummy_inputs)

    output_names = ["masks", "iou_predictions", "low_res_masks"]

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        with open(output, "wb") as f:
            print(f"Exporting onnx model to {output}...")
            torch.onnx.export(
                onnx_model,
                tuple(dummy_inputs.values()),
                f,
                export_params=True,
                verbose=False,
                opset_version=opset,
                do_constant_folding=True,
                input_names=list(dummy_inputs.keys()),
                output_names=output_names,
                dynamic_axes=dynamic_axes,
            )

    if onnxruntime_exists:
        ort_inputs = {k: to_numpy(v) for k, v in dummy_inputs.items()}
        # set cpu provider default
        providers = ["CPUExecutionProvider"]
        ort_session = onnxruntime.InferenceSession(output, providers=providers)
        _ = ort_session.run(None, ort_inputs)
        print("Model has successfully been run with ONNXRuntime.")


def to_numpy(tensor):
    return tensor.cpu().numpy()


if __name__ == "__main__":
    args = parser.parse_args()
    run_export(
        model_type=args.model_type,
        checkpoint=args.checkpoint,
        output=args.output,
        opset=args.opset,
        return_single_mask=args.return_single_mask,
        gelu_approximate=args.gelu_approximate,
        use_stability_score=args.use_stability_score,
        return_extra_metrics=args.return_extra_metrics,
    )

    if args.quantize_out is not None:
        assert onnxruntime_exists, "onnxruntime is required to quantize the model."
        from onnxruntime.quantization import QuantType  # type: ignore
        from onnxruntime.quantization.quantize import quantize_dynamic  # type: ignore

        print(f"Quantizing model and writing to {args.quantize_out}...")
        quantize_dynamic(
            model_input=args.output,
            model_output=args.quantize_out,
            optimize_model=True,
            per_channel=False,
            reduce_range=False,
            weight_type=QuantType.QUInt8,
        )
        print("Done!")
