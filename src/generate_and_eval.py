import time
import argparse
import numpy as np
import random

import os
from tqdm import tqdm
import torch
from datasets import load_dataset
from diffusers import (
    DDIMScheduler,
    LCMScheduler,
    DPMSolverMultistepScheduler,
    PNDMScheduler,
)
from torchvision.transforms.functional import to_pil_image
import open_clip
from DiPGO.sd_pruner.unet_2d_condition import UNet2DConditionModel
from DiPGO.sd_pruner.pipeline_stable_diffusion import StableDiffusionPipeline
from DiPGO.sd_pruner.pipeline_stable_diffusion_xl import StableDiffusionXLPipeline
from DiPGO.sd_pruner.pruner import Pruner


MODEL_NAME_MAPPING = {
    "sd1.5": "runwayml/stable-diffusion-v1-5",
    "runwayml/stable-diffusion-v1-5": "runwayml/stable-diffusion-v1-5",
    "sdxl": "stabilityai/stable-diffusion-xl-base-1.0",
    "sdxl-lcm": "stabilityai/stable-diffusion-xl-base-1.0",
    "sd2.1": "stabilityai/stable-diffusion-2-1",
}


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main(args):
    if args.dataset == "parti":
        if args.cache_dir is None:
            prompts = load_dataset("nateraw/parti-prompts", split="train")
        else:
            prompts = load_dataset(
                "nateraw/parti-prompts",
                split="train",
                cache_dir=f"{args.cache_dir}/datasets/",
            )
    elif args.dataset == "coco2017":
        if args.cache_dir is None:
            dataset = load_dataset("phiyodr/coco2017")
        else:
            dataset = load_dataset(
                "phiyodr/coco2017", cache_dir=f"{args.cache_dir}/datasets/"
            )
        prompts = [{"Prompt": sample["captions"][0]}
                   for sample in dataset["validation"]]
    else:
        raise NotImplementedError

    torch_type = torch.float16
    print("Enable DiP-GO...")
    pretrained_model_name_or_path = MODEL_NAME_MAPPING[
        args.pretrained_model_name_or_path
    ]

    if args.pretrained_model_name_or_path == "sd1.5":
        from DiPGO.extension.unethelper import UNetHelper
    # elif args.pretrained_model_name_or_path == "sd2.1":
    #     from DiPGO import UNetHelper

    unet = UNet2DConditionModel.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="unet",
        cache_dir=args.cache_dir).cuda()
    pipe = StableDiffusionPipeline.from_pretrained(
        pretrained_model_name_or_path,
        unet=unet,
        safety_checker=None,
        torch_dtype=torch_type,
        cache_dir=args.cache_dir,
    ).to("cuda:0")

    unet = pipe.unet
    if args.scheduler == "lcm":
        noise_scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    elif args.scheduler == "dpm":
        noise_scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config)
    elif args.scheduler == "ddim":
        noise_scheduler = DDIMScheduler.from_pretrained(
            pretrained_model_name_or_path, subfolder="scheduler"
        )
    else:
        noise_scheduler = PNDMScheduler.from_pretrained(
            pretrained_model_name_or_path, subfolder="scheduler"
        )

    noise_scheduler.set_timesteps(
        num_inference_steps=args.num_inference_steps, device="cuda:0"
    )
    pipe.scheduler = noise_scheduler

    print("Loading model: ViT-g-14")
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-g-14", pretrained="laion2b_s34b_b88k", cache_dir=args.cache_dir
    )  # https://github.com/Nota-NetsPresso/BK-SDM/blob/a27494fe46d6d4ca0ea45291b0b8b5b547b635fd/src/eval_clip_score.py#L25
    tokenizer = open_clip.get_tokenizer("ViT-g-14")

    if args.prune:
        masked_helper = UNetHelper(
            args, unet=unet, timesteps=noise_scheduler.timesteps
        )
        masked_helper.eval()

        if args.pruner_model_path is not None:
            masked_helper.load_state_dict(torch.load(args.pruner_model_path))
            masked_helper.prune(args.prune_ratio)
        else:
            import pickle
            with open("save/gates/sd15_gates_0.8.pkl", "rb") as file:
                gates = pickle.load(file)
            masked_helper.all_timestep_gates = gates
        masked_helper.enable()
        masked_helper = masked_helper.cuda()
        masked_helper.to(torch_type)

    start_time = time.time()
    image_list, prompt_list = [], []
    num_batch = len(prompts) // args.batch_size
    if len(prompts) % args.batch_size != 0:
        num_batch += 1

    for i in tqdm(range(num_batch)):
        start, end = args.batch_size * \
            i, min(args.batch_size * (i + 1), len(prompts))
        sample_prompts = [
            prompts[i]["Prompt"] for i in range(start, end)
        ]  # [prompts[i] for i in range(start, end)]#
        set_random_seed(args.seed)
        pipe_output = pipe(
            sample_prompts,
            output_type="np",
            return_dict=True,
            num_inference_steps=args.num_inference_steps,
        )

        images = pipe_output.images
        images_int = (images * 255).astype("uint8")
        torch_int_images = torch.from_numpy(images_int).permute(0, 3, 1, 2)

        image_list.append(torch_int_images)
        prompt_list += sample_prompts
        # break

    use_time = round(time.time() - start_time, 2)
    model.cuda()

    all_images = torch.cat(image_list, dim=0)

    # evaluate
    all_images = [to_pil_image(i, mode=None) for i in all_images]
    batch_size = 16
    with torch.no_grad(), torch.cuda.amp.autocast():
        all_score = []
        num_batch = len(prompt_list) // batch_size
        if len(prompt_list) % batch_size != 0:
            num_batch += 1

        for i in tqdm(range(num_batch)):
            img_subset = torch.stack(
                [
                    preprocess(i)
                    for i in all_images[i * batch_size: (i + 1) * batch_size]
                ],
                0,
            ).cuda()
            prompt_subset = prompt_list[i * batch_size: (i + 1) * batch_size]
            prompts = tokenizer(prompt_subset).cuda()

            image_features = model.encode_image(img_subset)
            text_features = model.encode_text(prompts)

            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            score = 100.0 * (image_features * text_features).sum(axis=-1)
            score = torch.max(score, torch.zeros_like(score))

            all_score.append(score.detach().cpu())

    final_score = torch.cat(all_score).mean(0).item()
    print("Score=", final_score)

    save_dir = f"{os.path.dirname(args.pruner_model_path)}/{args.dataset}_ckpt/{os.path.basename(args.pruner_model_path).split('.')[0]}_{args.scheduler}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if args.prune:
        masked_helper = masked_helper.float()
        flops = masked_helper.get_average_flops()
        torch.save(
            {
                "images": all_images,
                "prompts": prompt_list,
            },
            f"{save_dir}/images-prune-{args.prune_ratio}-{args.num_inference_steps}-time-{use_time}-{use_time/len(prompts):.2f}-flops-{flops:.4f}G-clipscore-{final_score:.4f}.pt",
        )
    elif args.original:
        torch.save(
            {
                "images": all_images,
                "prompts": prompt_list,
            },
            f"{args.dataset}_ckpt/images-original-{args.num_inference_steps}-time-{use_time}.pt",
        )
    elif args.bk is not None:
        torch.save(
            {
                "images": all_images,
                "prompts": prompt_list,
            },
            f"{args.dataset}_ckpt/images-bksdm-{args.bk}-{args.num_inference_steps}-time-{use_time}.pt",
        )
    else:
        torch.save(
            {
                "images": all_images,
                "prompts": prompt_list,
            },
            f"{args.dataset}_ckpt/images-deepcache-{args.num_inference_steps}-block-{args.block}-layer-{args.layer}-interval-{args.update_interval}-uniform-{args.uniform}-pow-{args.pow}-center-{args.center}-time-{use_time}.pt",
        )

    save_path = f"{save_dir}/images-prune-{args.prune_ratio}-{args.num_inference_steps}-time-{use_time}-{use_time/len(prompts):.2f}-flops-{flops:.4f}G-clipscore-{final_score:.4f}.pt"
    fid_save_path = f"{save_dir}/images-prune-{args.prune_ratio}-{args.num_inference_steps}-time-{use_time}-{use_time/len(prompts):.2f}-flops-{flops:.4f}G-clipscore-{final_score:.4f}_fid.txt"
    os.system(
        "python metrics/fid_score.py data/val2017 {} 2>&1|tee {}".format(
            save_path, fid_save_path
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str)

    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"),
    )

    parser.add_argument("--original", action="store_true")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="sd1.5")
    parser.add_argument("--prune", action="store_true")
    parser.add_argument("--fid", action="store_true")
    parser.add_argument("--bk", type=str, default=None)

    parser.add_argument(
        "--scheduler",
        type=str,
        default="ddim",
        choices=["dpm", "ddim", "lcm", "plms"],
        help="Select which scheduler to use.",
    )
    parser.add_argument(
        "--fine_grained",
        default=False,
        action="store_true",
        help="Flag to use fine-grained branch-wise helper.",
    )
    parser.add_argument(
        "--start_branch",
        type=int,
        default=0,
        help="Number of start prior branch, 0 means tuned all branch gates.",
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

    # Hyperparameters for DeepCache
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--block", type=int, default=0)
    parser.add_argument("--update_interval", type=int, default=None)
    parser.add_argument("--uniform", action="store_true", default=False)
    parser.add_argument("--pow", type=float, default=None)
    parser.add_argument("--center", type=int, default=None)
    # Sampling setup
    parser.add_argument("--pruner_model_path", type=str, default=None, required=False)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--prune_ratio", type=float, default=0.6)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=4)

    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    print(args)
    set_random_seed(args.seed)
    main(args)
