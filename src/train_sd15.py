#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import argparse
import copy
import gc
import logging
import math
import os
import shutil
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from packaging import version
from PIL import Image
from PIL.ImageOps import exif_transpose
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig
from diffusers.utils.torch_utils import randn_tensor
import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    DiffusionPipeline,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from tqdm import tqdm as tqdm_bar
import copy
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    retrieve_timesteps,
)

from diffusers.optimization import get_scheduler
from diffusers.training_utils import compute_snr
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module

# from DiPGO import SoftUNetHelper, FGUNetHelper, TimeCondUNetHelper
from DiPGO.extension.unethelper import UNetHelper
from torchvision.utils import save_image
from torch import nn
from torch.utils.checkpoint import checkpoint
import random
import json
from PIL import Image
from utils import ssim_loss
import time
import open_clip
from torchvision.transforms.functional import to_pil_image


# Will error if the minimal version of diffusers is not installed. Remove
# at your own risks.
check_min_version("0.28.0.dev0")

logger = get_logger(__name__)


def validation(
    unet,
    val_dataset_prompts,
    masked_helper,
    args,
    weight_dtype,
    prune_ratio=0.85,
    val_batch_size=16,
):
    logger.info(f"Running validation...")

    pipe = StableDiffusionPipeline.from_pretrained(
        sd_model_name,
        unet=unet,
        safety_checker=None,
        torch_dtype=weight_dtype,
        cache_dir=args.cache_dir,
    ).to("cuda:0")
    print("Loading model: ViT-g-14")
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-g-14", pretrained="laion2b_s34b_b88k", cache_dir=args.cache_dir
    )
    tokenizer = open_clip.get_tokenizer("ViT-g-14")

    masked_helper.eval()
    masked_helper.prune(prune_ratio)

    image_list, prompt_list = [], []
    num_batch = len(val_dataset_prompts) // val_batch_size
    if len(val_dataset_prompts) % val_batch_size != 0:
        num_batch += 1

    for i in tqdm_bar(range(num_batch)):
        start, end = val_batch_size * i, min(
            val_batch_size * (i + 1), len(val_dataset_prompts)
        )
        sample_prompts = [
            val_dataset_prompts[i] for i in range(start, end)
        ]  # [prompts[i] for i in range(start, end)]#
        generator = (
            None
            if args.seed is None
            else torch.Generator(device=unet.device).manual_seed(args.seed)
        )

        with torch.no_grad():
            pipe_output = pipe(
                sample_prompts,
                output_type="np",
                return_dict=True,
                num_inference_steps=50,
                generator=generator,
            )

        images = pipe_output.images
        images_int = (images * 255).astype("uint8")
        torch_int_images = torch.from_numpy(images_int).permute(0, 3, 1, 2)

        image_list.append(torch_int_images)
        prompt_list += sample_prompts

    model.cuda()
    all_images = torch.cat(image_list, dim=0)
    # evaluate
    all_images = [to_pil_image(i, mode=None) for i in all_images]
    with torch.no_grad(), torch.cuda.amp.autocast():
        all_score = []
        num_batch = len(prompt_list) // val_batch_size
        if len(prompt_list) % val_batch_size != 0:
            num_batch += 1

        for i in tqdm_bar(range(num_batch)):
            img_subset = torch.stack(
                [
                    preprocess(i)
                    for i in all_images[i * val_batch_size: (i + 1) * val_batch_size]
                ],
                0,
            ).cuda()
            prompt_subset = prompt_list[i * \
                val_batch_size: (i + 1) * val_batch_size]
            prompts = tokenizer(prompt_subset).cuda()

            image_features = model.encode_image(img_subset)
            text_features = model.encode_text(prompts)

            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            score = 100.0 * (image_features * text_features).sum(axis=-1)
            score = torch.max(score, torch.zeros_like(score))

            all_score.append(score.detach().cpu())

    final_score = torch.cat(all_score).mean(0).item()
    logger.info(f"Evaluation clip score= {final_score}")
    del pipe
    torch.cuda.empty_cache()

    return final_score


def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import (
            RobertaSeriesModelWithTransformation,
        )

        return RobertaSeriesModelWithTransformation
    elif model_class == "T5EncoderModel":
        from transformers import T5EncoderModel

        return T5EncoderModel
    else:
        raise ValueError(f"{model_class} is not supported.")


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(
        description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="CompVis/stable-diffusion-v1-4",
        required=False,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--disable_gamma",
        action="store_true",
        help="Whether or not to disable gamma.",
    )
    parser.add_argument(
        "--time_cond",
        default=False,
        action="store_true",
        help="Flag to use time conditional gate layer.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="poloclub/diffusiondb",
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default="large_random_1k",
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."),
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"),
    )
    parser.add_argument(
        "--sparsity_early_stop",
        default=False,
        action="store_true",
        help=("Whether to use to sparsity early stop sparse loss."),
    )

    # model related
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--fine_grained",
        default=False,
        action="store_true",
        help="Flag to use fine-grained branch-wise helper.",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default="ddim",
        choices=["dpm", "ddim", "lcm"],
        help="Select which scheduler to use.",
    )
    parser.add_argument(
        "--soft",
        default=False,
        action="store_true",
        help="Flag to use soft gate layer.",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help=("The denosing steps"),
    )
    parser.add_argument(
        "--dynamic_threshold",
        default=False,
        action="store_true",
        help="Flag to use fine-grained branch-wise helper.",
    )
    parser.add_argument(
        "--use_attn",
        default=False,
        action="store_true",
        help="Flag to use self-attention and cross-attention modules for query interaction.",
    )
    parser.add_argument(
        "--mlp_module_type",
        type=str,
        default="block_wise",
        help="Type for share mlp layer for each time step.",
    )
    parser.add_argument(
        "--self_step_attn_module_type",
        type=str,
        default="model_wise",
        help="Type for share self step attention across steps.",
    )

    parser.add_argument(
        "--activation_type",
        type=str,
        default="relu",
        help="Type of activation function.",
    )
    parser.add_argument(
        "--encoder_layer_num",
        type=int,
        default=1,
        help="Number Transformer encoder layers.",
    )
    parser.add_argument(
        "--mlp_layer_num", type=int, default=3, help="Number MLP layers."
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=512,
        help="Number embedding hidden dim.")
    parser.add_argument(
        "--attention_type",
        type=str,
        default="self_step",
        help='Choose in ["pre_self_step", "cross_step", "post_self_step", "post_mlp"], and use "-" to combine them.',
    )
    parser.add_argument(
        "--ignore_blocks",
        type=str,
        default=None,
        help="Choose in 0 to 8. example: 0_8. It means ignore the first and last block.",
    )

    parser.add_argument(
        "--stop_threshold",
        type=float,
        default=0.2,
        help=("The prune threshold for sparsity early stop ."),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."),
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--image_column",
        type=str,
        default="image",
        help="The column of the dataset containing an image.",
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="prompt",
        help="The column of the dataset containing a caption or a list of captions.",
    )
    parser.add_argument(
        "--cl_type",
        type=str,
        default=None,
        choices=["mse", "l1", "ssim", "ssim+l1"],
        help="Consistent Loss type",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--instance_data_dir",
        type=str,
        default="dog",
        required=False,
        help="A folder containing the training data of instance images.",
    )
    parser.add_argument(
        "--class_data_dir",
        type=str,
        default=None,
        required=False,
        help="A folder containing the training data of class images.",
    )
    parser.add_argument(
        "--instance_prompt",
        type=str,
        default="a photo of sks dog",
        required=False,
        help="The prompt with identifier specifying the instance",
    )
    parser.add_argument(
        "--class_prompt",
        type=str,
        default=None,
        help="The prompt to specify images in the same class as provided instance images.",
    )
    parser.add_argument(
        "--with_prior_preservation",
        default=False,
        action="store_true",
        help="Flag to add prior preservation loss.",
    )

    parser.add_argument(
        "--text_to_img",
        default=False,
        action="store_true",
        help="Use text prompts to generate image.",
    )
    parser.add_argument(
        "--prior_loss_weight",
        type=float,
        default=1.0,
        help="The weight of prior preservation loss.",
    )
    parser.add_argument(
        "--num_class_images",
        type=int,
        default=100,
        help=(
            "Minimal class images for prior preservation loss. If there are not enough images already present in"
            " class_data_dir, additional images will be sampled with class_prompt."),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="dreambooth-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="A seed for reproducible training.")
    parser.add_argument(
        "--train_text_encoder",
        action="store_true",
        help="Whether to train the text encoder. If set, the text encoder should be float32 precision.",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=4,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--sample_batch_size",
        type=int,
        default=4,
        help="Batch size (per device) for sampling images.",
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."),
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. Checkpoints can be used for resuming training via `--resume_from_checkpoint`. "
            "In the case that the checkpoint is better than the final trained model, the checkpoint can also be used for inference."
            "Using a checkpoint for inference requires separate loading of the original pipeline and the individual checkpointed model components."
            "See https://huggingface.co/docs/diffusers/main/en/training/dreambooth#performing-inference-using-a-saved-checkpoint for step by step"
            "instructions."),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=(
            "Max number of checkpoints to store. Passed as `total_limit` to the `Accelerator` `ProjectConfiguration`."
            " See Accelerator::save_state https://huggingface.co/docs/accelerate/package_reference/accelerator#accelerate.Accelerator.save_state"
            " for more details"),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-6,
        help="Initial learning rate (after the potential warmup period) to use.",
    )

    parser.add_argument(
        "--sgd_weight_decay",
        type=float,
        default=1e-4,
        help="weight_decay for SGD.",
    )

    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5,
        help="Classifier free guidance scale.",
    )

    parser.add_argument(
        "--sparse_scale",
        type=float,
        default=1.0,
        help="Loss weight for sparsity loss.",
    )

    parser.add_argument(
        "--consistent_scale",
        type=float,
        default=1.0,
        help="Loss weight for consistency loss.",
    )

    parser.add_argument(
        "--do_classifier_free_guidance",
        type=bool,
        default=True,
        help="Whether or not to do classifier free guidance scale.",
    )

    parser.add_argument(
        "--save_interval",
        type=int,
        default=10,
        help="Interval for saving results.")

    parser.add_argument(
        "--save_ckpt_interval",
        type=int,
        default=100,
        help="Interval for saving checkpoints.",
    )

    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'),
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=500,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument(
        "--lr_power",
        type=float,
        default=1.0,
        help="Power factor of the polynomial scheduler.",
    )

    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="The beta1 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="The beta2 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_weight_decay",
        type=float,
        default=1e-2,
        help="Weight decay to use.")
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer",
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether or not to push the model to the Hub.",
    )
    parser.add_argument(
        "--hub_token",
        type=str,
        default=None,
        help="The token to use to push to the Model Hub.",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'),
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        help="A prompt that is used during validation to verify that the model is learning.",
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=50,
        help=(
            "Run validation every X steps. Validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`"
            " and logging the images."),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--prior_generation_precision",
        type=str,
        default=None,
        choices=[
            "no",
            "fp32",
            "fp16",
            "bf16"],
        help=(
            "Choose prior generation precision between fp32, fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to  fp16 if a GPU is available else fp32."),
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention",
        action="store_true",
        help="Whether or not to use xformers.",
    )
    parser.add_argument(
        "--set_grads_to_none",
        action="store_true",
        help=(
            "Save more memory by using setting grads to None instead of zero. Be aware, that this changes certain"
            " behaviors, so disable this argument if it causes any problems. More info:"
            " https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html"),
    )

    parser.add_argument(
        "--offset_noise",
        action="store_true",
        default=False,
        help=(
            "Fine-tuning against a modified noise"
            " See: https://www.crosslabs.org//blog/diffusion-with-offset-noise for more information."
        ),
    )
    parser.add_argument(
        "--snr_gamma",
        type=float,
        default=None,
        help="SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0. "
        "More details here: https://arxiv.org/abs/2303.09556.",
    )
    parser.add_argument(
        "--pre_compute_text_embeddings",
        action="store_true",
        help="Whether or not to pre-compute text embeddings. If text embeddings are pre-computed, the text encoder will not be kept in memory during training and will leave more GPU memory available for training the rest of the model. This is not compatible with `--train_text_encoder`.",
    )
    parser.add_argument(
        "--tokenizer_max_length",
        type=int,
        default=None,
        required=False,
        help="The maximum length of the tokenizer. If not set, will default to the tokenizer's max length.",
    )
    parser.add_argument(
        "--text_encoder_use_attention_mask",
        action="store_true",
        required=False,
        help="Whether to use attention mask for the text encoder",
    )
    parser.add_argument(
        "--skip_save_text_encoder",
        action="store_true",
        required=False,
        help="Set to not save text encoder",
    )
    parser.add_argument(
        "--validation_images",
        required=False,
        default=None,
        nargs="+",
        help="Optional set of images to use for validation. Used when the target pipeline takes an initial image as input such as when training image variation or superresolution.",
    )
    parser.add_argument(
        "--class_labels_conditioning",
        required=False,
        default=None,
        help="The optional `class_label` conditioning to pass to the unet, available values are `timesteps`.",
    )
    parser.add_argument(
        "--validation_scheduler",
        type=str,
        default="DPMSolverMultistepScheduler",
        choices=[
            "DPMSolverMultistepScheduler",
            "DDPMScheduler"],
        help="Select which scheduler to use for validation. DDPMScheduler is recommended for DeepFloyd IF.",
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.with_prior_preservation:
        if args.class_data_dir is None:
            raise ValueError(
                "You must specify a data directory for class images.")
        if args.class_prompt is None:
            raise ValueError("You must specify prompt for class images.")
    else:
        # logger is not available yet
        if args.class_data_dir is not None:
            warnings.warn(
                "You need not use --class_data_dir without --with_prior_preservation."
            )
        if args.class_prompt is not None:
            warnings.warn(
                "You need not use --class_prompt without --with_prior_preservation."
            )

    if args.train_text_encoder and args.pre_compute_text_embeddings:
        raise ValueError(
            "`--train_text_encoder` cannot be used with `--pre_compute_text_embeddings`"
        )

    return args



class JsonDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
        self,
        dataset_version,
        tokenizer,
        size=512,
        is_train=True,
        center_crop=False,
        tokenizer_max_length=None,
        sample_num=None,
    ):
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer
        self.tokenizer_max_length = tokenizer_max_length
        with open(f"data/{dataset_version}/{dataset_version}.json", "r") as file:
            data = json.load(file)

        data_root = f"data/{dataset_version}"
        self.instance_images_path = [os.path.join(
            data_root, x) for x in data.keys()][:sample_num]  # json {}
        self.original_prompts = [x["p"] for x in data.values()][:sample_num]

        def tokenize_captions(instance_prompts, is_train=True):
            captions = []
            for caption in instance_prompts:
                if isinstance(caption, str):
                    captions.append(caption)
                elif isinstance(caption, (list, np.ndarray)):
                    # take a random caption if there are multiple
                    captions.append(
                        random.choice(caption) if is_train else caption[0])
                else:
                    raise ValueError(
                        f"Caption should contain either strings or lists of strings."
                    )
            inputs = tokenizer(
                captions,
                max_length=tokenizer.model_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            return inputs, captions

        self.instance_prompts, self.captions = tokenize_captions(
            self.original_prompts, is_train
        )
        self.num_instance_images = len(self.instance_images_path)
        self._length = self.num_instance_images

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(
                    size, interpolation=transforms.InterpolationMode.BILINEAR
                ),
                (
                    transforms.CenterCrop(size)
                    if center_crop
                    else transforms.RandomCrop(size)
                ),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        self.uncond_input = tokenize_prompt(
            self.tokenizer, "", tokenizer_max_length=self.tokenizer_max_length
        )

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        if isinstance(index, list):
            assert len(index) == 1
            index = index[0]
        instance_image = Image.open(
            self.instance_images_path[index % self.num_instance_images]
        )
        instance_image = exif_transpose(instance_image)

        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
        example["instance_images"] = self.image_transforms(instance_image)

        sample_index = index % self.num_instance_images
        example["instance_prompt_ids"] = self.instance_prompts.input_ids[
            sample_index: sample_index + 1
        ]
        example["instance_attention_mask"] = self.instance_prompts.attention_mask[
            sample_index: sample_index + 1
        ]

        # unconditional prompt tokenizer
        example["uncond_prompt_ids"] = self.uncond_input.input_ids
        example["uncond_attention_mask"] = self.uncond_input.attention_mask
        example["original_prompts"] = self.original_prompts[
            index % self.num_instance_images
        ]
        return example


def collate_fn(examples, with_prior_preservation=False):
    has_attention_mask = "instance_attention_mask" in examples[0]

    input_ids = [example["instance_prompt_ids"] for example in examples]
    pixel_values = [example["instance_images"] for example in examples]
    uncond_inputs_ids = [example["uncond_prompt_ids"] for example in examples]

    if has_attention_mask:
        attention_mask = [example["instance_attention_mask"]
                          for example in examples]
        uncond_attention_mask = [
            example["uncond_attention_mask"] for example in examples
        ]

    # Concat class and instance examples for prior preservation.
    # We do this to avoid doing two forward passes.
    if with_prior_preservation:
        input_ids += [example["class_prompt_ids"] for example in examples]
        pixel_values += [example["class_images"] for example in examples]

        if has_attention_mask:
            attention_mask += [example["class_attention_mask"]
                               for example in examples]

    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(
        memory_format=torch.contiguous_format).float()

    input_ids = torch.cat(input_ids, dim=0)
    uncond_inputs_ids = torch.cat(uncond_inputs_ids, dim=0)

    batch = {
        "input_ids": input_ids,
        "pixel_values": pixel_values,
        "uncond_inputs_ids": uncond_inputs_ids,
    }

    if has_attention_mask:
        attention_mask = torch.cat(attention_mask, dim=0)
        batch["attention_mask"] = attention_mask

        uncond_attention_mask = torch.cat(uncond_attention_mask, dim=0)
        batch["uncond_attention_mask"] = uncond_attention_mask

    return batch


def sparse_loss(all_timestep_logits):
    total_loss = []
    target = torch.tensor([0], dtype=torch.long)  # 0 for skip, 1 for keep

    for logits in all_timestep_logits:
        for _, p in logits.items():
            loss = F.cross_entropy(p.unsqueeze(0).float(), target.to(p.device))
            total_loss.append(loss)
    return sum(total_loss) / len(total_loss)


def tokenize_prompt(tokenizer, prompt, tokenizer_max_length=None):
    if tokenizer_max_length is not None:
        max_length = tokenizer_max_length
    else:
        max_length = tokenizer.model_max_length

    text_inputs = tokenizer(
        prompt,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )

    return text_inputs


def encode_prompt(
    text_encoder,
    input_ids,
    attention_mask,
    text_encoder_use_attention_mask=None,
    negative_input_ids=None,
    negative_attention_mask=None,
):
    text_input_ids = input_ids.to(text_encoder.device)

    if text_encoder_use_attention_mask:
        attention_mask = attention_mask.to(text_encoder.device)
    else:
        attention_mask = None

    # print(text_input_ids.shape)
    prompt_embeds = text_encoder(
        text_input_ids,
        attention_mask=attention_mask,
        return_dict=False,
    )
    prompt_embeds = prompt_embeds[0]

    # get unconditional embeddings for classifier free guidance
    negative_prompt_embeds = text_encoder(
        negative_input_ids.to(text_encoder.device),
        attention_mask=negative_attention_mask,
        return_dict=False,
    )
    negative_prompt_embeds = negative_prompt_embeds[0]

    return prompt_embeds, negative_prompt_embeds


def main(args):
    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub.")

    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError(
                "Make sure to install wandb if you want to use it for logging during training."
            )

    if (
        args.train_text_encoder
        and args.gradient_accumulation_steps > 1
        and accelerator.num_processes > 1
    ):
        raise ValueError(
            "Gradient accumulation is not supported when training the text encoder in distributed training. "
            "Please set gradient_accumulation_steps to 1. This feature will be supported in the future.")

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    if args.pretrained_model_name_or_path == "sd1.5":
        sd_model_name = "runwayml/stable-diffusion-v1-5"
    elif args.pretrained_model_name_or_path == "sd2.1":
        sd_model_name = "stabilityai/stable-diffusion-2-1-base"

    # torch_dtype = torch.float32
    torch_dtype = torch.float16

    pipeline = StableDiffusionPipeline.from_pretrained(
        sd_model_name,
        torch_dtype=torch_dtype,
        safety_checker=None,
        revision=args.revision,
        cache_dir=args.cache_dir,
        variant=args.variant,
    ).to(accelerator.device)
    image_processor = copy.deepcopy(pipeline.image_processor)

    if args.scheduler == "ddim":
        noise_scheduler = DDIMScheduler.from_pretrained(
            sd_model_name, subfolder="scheduler", cache_dir=args.cache_dir
        )
        noise_scheduler.set_timesteps(
            num_inference_steps=args.num_inference_steps, device="cuda:0"
        )
    elif args.scheduler == "dpm":
        noise_scheduler = DPMSolverMultistepScheduler.from_config(
            pipeline.scheduler.config
        )
        noise_scheduler.set_timesteps(
            num_inference_steps=args.num_inference_steps, device="cuda:0"
        )
    else:
        raise NotImplementedError

    del pipeline

    # Handle the repository creation
    if accelerator.is_main_process:

        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
            os.makedirs(args.output_dir + os.sep + "images/", exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name,
                exist_ok=True,
                token=args.hub_token,
            ).repo_id

    # Load the tokenizer
    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name, revision=args.revision, use_fast=False
        )
    elif sd_model_name:
        tokenizer = AutoTokenizer.from_pretrained(
            sd_model_name,
            subfolder="tokenizer",
            revision=args.revision,
            use_fast=False,
            cache_dir=args.cache_dir,
        )

    # import correct text encoder class
    text_encoder_cls = import_model_class_from_model_name_or_path(
        sd_model_name, args.revision
    )
    text_encoder = text_encoder_cls.from_pretrained(
        sd_model_name,
        subfolder="text_encoder",
        revision=args.revision,
        variant=args.variant,
        cache_dir=args.cache_dir,
    )

    try:
        vae = AutoencoderKL.from_pretrained(
            sd_model_name,
            subfolder="vae",
            revision=args.revision,
            variant=args.variant,
            cache_dir=args.cache_dir,
        )
    except OSError:
        # IF does not have a VAE so let's just set it to None
        # We don't have to error out here
        vae = None

    unet = UNet2DConditionModel.from_pretrained(
        sd_model_name,
        subfolder="unet",
        revision=args.revision,
        variant=args.variant,
        cache_dir=args.cache_dir,
    )

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # create custom saving & loading hooks so that
    # `accelerator.save_state(...)` serializes in a nice format
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            for model in models:
                sub_dir = (
                    "unet"
                    if isinstance(model, type(unwrap_model(unet)))
                    else "text_encoder"
                )
                model.save_pretrained(os.path.join(output_dir, sub_dir))
                # make sure to pop weight so that corresponding model is not
                # saved again
                weights.pop()

    def load_model_hook(models, input_dir):
        while len(models) > 0:
            # pop models so that they are not loaded again
            model = models.pop()

            if isinstance(model, type(unwrap_model(text_encoder))):
                # load transformers style into model
                load_model = text_encoder_cls.from_pretrained(
                    input_dir, subfolder="text_encoder"
                )
                model.config = load_model.config
            else:
                # load diffusers style into model
                load_model = UNet2DConditionModel.from_pretrained(
                    input_dir, subfolder="unet"
                )
                model.register_to_config(**load_model.config)

            model.load_state_dict(load_model.state_dict())
            del load_model

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    if vae is not None:
        vae.requires_grad_(False)

    if not args.train_text_encoder:
        text_encoder.requires_grad_(False)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warning(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError(
                "xformers is not available. Make sure it is installed correctly"
            )

    # Check that all trainable models are in full precision
    low_precision_error_string = (
        "Please make sure to always have all model weights in full float32 precision when starting training - even if"
        " doing mixed precision training. copy of the weights should still be float32.")

    if unwrap_model(unet).dtype != torch.float32:
        raise ValueError(
            f"Unet loaded as datatype {unwrap_model(unet).dtype}. {low_precision_error_string}"
        )

    if args.train_text_encoder and unwrap_model(
            text_encoder).dtype != torch.float32:
        raise ValueError(
            f"Text encoder loaded as datatype {unwrap_model(text_encoder).dtype}."
            f" {low_precision_error_string}")

    # Enable TF32 for faster training on Ampere GPUs,
    # cf
    # https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate
            * args.gradient_accumulation_steps
            * args.train_batch_size
            * accelerator.num_processes
        )
    else:
        raise ValueError(f"Scale lr is required.")

    if args.pre_compute_text_embeddings:

        def compute_text_embeddings(prompt):
            with torch.no_grad():
                text_inputs = tokenize_prompt(
                    tokenizer, prompt, tokenizer_max_length=args.tokenizer_max_length)
                prompt_embeds = encode_prompt(
                    text_encoder,
                    text_inputs.input_ids,
                    text_inputs.attention_mask,
                    text_encoder_use_attention_mask=args.text_encoder_use_attention_mask,
                )

            return prompt_embeds

        pre_computed_encoder_hidden_states = compute_text_embeddings(
            args.instance_prompt
        )
        validation_prompt_negative_prompt_embeds = compute_text_embeddings("")

        if args.validation_prompt is not None:
            validation_prompt_encoder_hidden_states = compute_text_embeddings(
                args.validation_prompt
            )
        else:
            validation_prompt_encoder_hidden_states = None

        if args.class_prompt is not None:
            pre_computed_class_prompt_encoder_hidden_states = compute_text_embeddings(
                args.class_prompt)
        else:
            pre_computed_class_prompt_encoder_hidden_states = None

        text_encoder = None
        tokenizer = None

        gc.collect()
        torch.cuda.empty_cache()
    else:
        pre_computed_encoder_hidden_states = None
        validation_prompt_encoder_hidden_states = None
        validation_prompt_negative_prompt_embeds = None
        pre_computed_class_prompt_encoder_hidden_states = None

    # # Dataset and DataLoaders creation:
    train_dataset = JsonDataset(
        dataset_version="part-000001",
        is_train=True,
        tokenizer=tokenizer,
        size=args.resolution,
        center_crop=args.center_crop,
        tokenizer_max_length=args.tokenizer_max_length,
    )

    val_dataset_prompts = JsonDataset(
        dataset_version="part-000002",
        is_train=False,
        sample_num=500,
        tokenizer=tokenizer,
        size=args.resolution,
        center_crop=args.center_crop,
        tokenizer_max_length=args.tokenizer_max_length,
    ).captions

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=lambda examples: collate_fn(
            examples, args.with_prior_preservation),
        # collate_fn=collate_fn,
        num_workers=args.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    weight_dtype = torch.float16

    # 4. Prepare timesteps
    num_inference_steps = 50
    # timesteps, num_inference_steps = retrieve_timesteps(noise_scheduler, num_inference_steps, None)
    timesteps = noise_scheduler.timesteps

    if args.dynamic_threshold:
        stop_threshold_list = np.linspace(0.9, 0.0, args.max_train_steps)

    helper = UNetHelper(args, unet, noise_scheduler.timesteps)

    if args.resume:
        print(f"load resume {args.resume}.")
        helper.load_state_dict(torch.load(args.resume))

    helper.enable()

    if unet is not None:
        unet.requires_grad_(False)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        if args.train_text_encoder:
            text_encoder.gradient_checkpointing_enable(use_reentrant=False)

    params_to_optimize = []
    for name, param in helper.named_parameters():
        # if "cache_mask" in name or "query_embed" in name or "encoder" in
        # name:
        if "unet" not in name:
            param.requires_grad_(True)
            params_to_optimize.append(param)
            # print(name)

    num_params = sum(p.numel() for p in params_to_optimize)
    print(
        f"Number of parameters in the model: {num_params} => {num_params / 1e6:.2f} M"
    )

    optimizer = torch.optim.SGD(
        params_to_optimize,
        lr=args.learning_rate,
        weight_decay=args.sgd_weight_decay,
    )

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # Prepare everything with our `accelerator`.
    if args.train_text_encoder:
        unet, text_encoder, optimizer, train_dataloader, lr_scheduler = (
            accelerator.prepare(
                unet, text_encoder, optimizer, train_dataloader, lr_scheduler
            )
        )
    else:
        unet, optimizer, train_dataloader, lr_scheduler, helper = accelerator.prepare(
            unet, optimizer, train_dataloader, lr_scheduler, helper)

    # module = accelerator.prepare(module)

    # Move vae and text_encoder to device and cast to weight_dtype
    if vae is not None:
        vae.to(accelerator.device, dtype=weight_dtype)

    if not args.train_text_encoder and text_encoder is not None:
        text_encoder.to(accelerator.device, dtype=weight_dtype)

    if unet is not None:
        unet.to(accelerator.device, dtype=weight_dtype)

    if helper is not None:
        helper.to(accelerator.device, dtype=weight_dtype)

    # print(unet.dtype, vae.dtype, text_encoder.dtype)
    # We need to recalculate our total training steps as the size of the
    # training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(
        args.max_train_steps /
        num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = vars(copy.deepcopy(args))
        tracker_config.pop("validation_images")
        accelerator.init_trackers("dreambooth", config=tracker_config)

    # Train!
    total_batch_size = (
        args.train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(
        f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(
        f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    best_eval_score = 0.0

    for epoch in range(first_epoch, args.num_train_epochs):
        unet.eval()
        helper.train()
        if args.train_text_encoder:
            text_encoder.train()
        for step, batch in enumerate(train_dataloader):
            generator = (
                None if args.seed is None else torch.Generator(
                    device=accelerator.device).manual_seed(
                    args.seed))
            with accelerator.accumulate([unet, helper]):
                # Get the text embedding for conditioning
                prompt_embeds, negative_prompt_embeds = encode_prompt(
                    text_encoder,
                    batch["input_ids"],
                    batch["attention_mask"],
                    text_encoder_use_attention_mask=args.text_encoder_use_attention_mask,
                    negative_input_ids=batch["uncond_inputs_ids"],
                    negative_attention_mask=batch["attention_mask"],
                )

                batch_size = prompt_embeds.shape[0]
                num_channels_latents = unet.config.in_channels
                shape = (
                    batch_size,
                    num_channels_latents,
                    unet.sample_size,
                    unet.sample_size,
                )

                latents = randn_tensor(
                    shape,
                    generator=generator,
                    device=unet.device,
                    dtype=weight_dtype)

                # scale the initial noise by the standard deviation required by
                # the scheduler
                noisy_model_input = latents * noise_scheduler.init_noise_sigma

                bsz, channels, height, width = noisy_model_input.shape
                if args.do_classifier_free_guidance:
                    prompt_embeds = torch.cat(
                        [negative_prompt_embeds, prompt_embeds])

                if unwrap_model(unet).config.in_channels == channels * 2:
                    noisy_model_input = torch.cat(
                        [noisy_model_input, noisy_model_input], dim=1
                    )

                if args.class_labels_conditioning == "timesteps":
                    class_labels = timesteps
                else:
                    class_labels = None

                # generator = None if args.seed is None else torch.Generator(device=accelerator.device).manual_seed(args.seed)
                # num_warmup_steps = len(timesteps) - num_inference_steps * noise_scheduler.order
                # self._num_timesteps = len(timesteps)
                def denoise():
                    latents = noisy_model_input.to(dtype=weight_dtype)
                    for i, t in tqdm(enumerate(timesteps)):
                        latent_model_input = (
                            torch.cat([latents] * 2)
                            if args.do_classifier_free_guidance
                            else latents
                        )
                        latent_model_input = noise_scheduler.scale_model_input(
                            latent_model_input, t
                        )
                        noise_pred = unet(
                            latent_model_input,
                            t,
                            prompt_embeds,
                            class_labels=class_labels,
                            return_dict=False,
                        )[0]

                        # classifier free guidance
                        if args.do_classifier_free_guidance:
                            noise_pred_uncond, noise_pred_text = noise_pred.chunk(
                                2)
                            noise_pred = noise_pred_uncond + args.guidance_scale * \
                                (noise_pred_text - noise_pred_uncond)

                        latents = noise_scheduler.step(
                            noise_pred, t, latents, return_dict=False
                        )[0]

                    return latents

                # 1. generate images with original pipelines
                with torch.no_grad():
                    helper.disable()
                    helper.eval()
                    original_latents = denoise()
                    decoded_original_latent = vae.decode(
                        original_latents / vae.config.scaling_factor,
                        return_dict=False,
                        generator=generator,
                    )[0]

                    if (
                        accelerator.is_main_process
                        and global_step % args.save_interval == 0
                        and accelerator.sync_gradients
                    ):
                        # generator = None if args.seed is None else torch.Generator(device=accelerator.device).manual_seed(args.seed)
                        # image = vae.decode(original_latents / vae.config.scaling_factor, return_dict=False, generator=generator)[0]
                        image = decoded_original_latent
                        image = image_processor.postprocess(
                            image,
                            output_type="pt",
                            do_denormalize=[True] * image.shape[0],
                        )
                        save_image(
                            [image[0]],
                            f"{args.output_dir}/images/img_original_{global_step}.png",
                        )

                torch.cuda.empty_cache()

                # 2. generate image with masked deep cache
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)
                with torch.enable_grad():
                    helper.enable()
                    helper.train()
                    helper.all_queries_forward()
                    masked_latents = denoise()
                    decoded_masked_latent = vae.decode(
                        masked_latents / vae.config.scaling_factor,
                        return_dict=False,
                        generator=generator,
                    )[0]

                    if args.cl_type == "mse":
                        consistency_loss = F.mse_loss(
                            decoded_original_latent.detach().float(),
                            decoded_masked_latent.float(),
                            reduction="mean",
                        )
                    elif args.cl_type == "ssim":
                        consistency_loss = ssim_loss(
                            decoded_original_latent.detach().float(),
                            decoded_masked_latent.float(),
                        )
                    elif args.cl_type == "l1":
                        consistency_loss = F.l1_loss(
                            decoded_original_latent.detach().float(),
                            decoded_masked_latent.float(),
                        )
                    elif args.cl_type == "ssim+l1":
                        consistency_loss = ssim_loss(
                            decoded_original_latent.detach().float(),
                            decoded_masked_latent.float(),
                        ) + F.l1_loss(
                            decoded_original_latent.detach().float(),
                            decoded_masked_latent.float(),
                        )
                    elif args.cl_type == "ssim+mse":
                        consistency_loss = ssim_loss(
                            decoded_original_latent.detach().float(),
                            decoded_masked_latent.float(),
                        ) + F.mse_loss(
                            decoded_original_latent.detach().float(),
                            decoded_masked_latent.float(),
                        )
                    else:
                        raise NotImplementedError

                    consistency_loss = consistency_loss * args.consistent_scale

                    if args.disable_gamma:
                        sparsity_loss = helper.sparse_loss() * args.sparse_scale
                    else:
                        sparsity_loss = helper.sparse_lossv2() * args.sparse_scale

                if args.dynamic_threshold:
                    args.stop_threshold = stop_threshold_list[global_step]

                if (
                    args.sparsity_early_stop
                    and helper.evaluate_sparsity() < args.stop_threshold
                ):
                    print("Enable sparsity early stop...")
                    loss = consistency_loss
                else:
                    loss = consistency_loss + sparsity_loss
                accelerator.backward(loss)
                # loss.backward()
                if accelerator.sync_gradients:
                    params_to_clip = params_to_optimize
                    accelerator.clip_grad_norm_(
                        params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()

            # Checks if the accelerator has performed an optimization step
            # behind the scenes
            if accelerator.sync_gradients:

                if accelerator.is_main_process:
                    if global_step % args.save_interval == 0:
                        with torch.no_grad():
                            helper.disable()
                            helper.enable()
                            helper.eval()
                            helper.all_queries_forward()
                            eval_masked_latents = denoise()
                            sparsity = helper.evaluate_sparsity()
                            logs["sparsity"] = sparsity
                            image = vae.decode(
                                eval_masked_latents /
                                vae.config.scaling_factor,
                                return_dict=False,
                                generator=generator,
                            )[0]
                            image = image_processor.postprocess(
                                image,
                                output_type="pt",
                                do_denormalize=[True] * image.shape[0],
                            )
                            save_image(
                                [image[0]],
                                f"{args.output_dir}/images/img_cache_masked_{global_step}.png",
                            )

                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % args.save_ckpt_interval == 0:
                        helper.save_state_dict(
                            args.output_dir + os.sep + f"checkpoint_{global_step}.pth")

                    if global_step % args.validation_steps == 0:
                        eval_score = validation(
                            unet,
                            val_dataset_prompts,
                            helper,
                            args,
                            weight_dtype,
                            prune_ratio=0.85,
                        )
                        if eval_score > best_eval_score:
                            best_eval_score = eval_score
                            logger.info(
                                f"Save best model with clip score {eval_score}")
                            helper.save_state_dict(
                                args.output_dir + os.sep + f"checkpoint_best.pth")

                # progress_bar.update(1)
                # global_step += 1

            logs = {
                "loss": loss.detach().item(),
                "consistency_loss": consistency_loss.detach().item(),
                "sparsity_loss": sparsity_loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
                "stop_threshold:": args.stop_threshold,
            }
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        helper.save_state_dict(args.output_dir + os.sep + f"checkpoint.pth")

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    print(args)
    print("use_attn:", args.use_attn)
    print("mlp_module_type:", args.mlp_module_type)
    print("self_step_attn_module_type:", args.self_step_attn_module_type)
    print("activation_type:", args.activation_type)
    print("encoder_layer_num:", args.encoder_layer_num)
    print("mlp_layer_num:", args.mlp_layer_num)
    print("hidden_dim:", args.hidden_dim)
    print("attention_type:", args.attention_type)
    print("consistent_loss_type:", args.cl_type)
    print("sparsity_early_stop:", args.sparsity_early_stop)
    print("stop_threshold:", args.stop_threshold)
    main(args)
