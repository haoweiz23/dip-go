import os
import torch
from diffusers import (
    DDIMScheduler,
    LCMScheduler,
    DPMSolverMultistepScheduler,
    PNDMScheduler,
)
from DiPGO.sd_pruner.unet_2d_condition import UNet2DConditionModel
from DiPGO.sd_pruner.pipeline_stable_diffusion import StableDiffusionPipeline
from DiPGO.extension.unethelper import UNetHelper
import argparse
from datasets import load_dataset
import open_clip
import time
import pickle


def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def calculate_clip_score(model, prompt_list, img_list):
    model = model.cuda()

    img_subset = torch.stack([preprocess(i) for i in img_list], 0).cuda()
    prompts = tokenizer(prompt_list).cuda()

    image_features = model.encode_image(img_subset)
    text_features = model.encode_text(prompts)

    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    score = 100.0 * (image_features * text_features).sum(axis=-1)
    score = torch.max(score, torch.zeros_like(score))
    return score.cpu().squeeze().item()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str)

    # For choosing baselines. If these two are not set, then it will use DiPGO.
    parser.add_argument("--original", action="store_true")
    parser.add_argument("--prune", action="store_true")

    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="sd1.5")
    parser.add_argument("--fid", action="store_true")
    parser.add_argument("--bk", type=str, default=None)

    parser.add_argument(
        "--scheduler",
        type=str,
        default="ddim",
        choices=["dpm", "ddim", "lcm"],
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

    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=4)

    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_random_seed(args.seed)

    dataset = load_dataset("phiyodr/coco2017")
    prompts = [{"Prompt": sample["captions"][0]}
               for sample in dataset["validation"]]

    pretrained_path = "runwayml/stable-diffusion-v1-5"
    # args.pruner_model_path = "save/pruner_sd15_prune0.8.pth"

    # noise_scheduler = PNDMScheduler.from_pretrained(pretrained_path, subfolder="scheduler")
    noise_scheduler = DDIMScheduler.from_pretrained(
        pretrained_path, subfolder="scheduler"
    )
    noise_scheduler.set_timesteps(num_inference_steps=50, device="cuda:0")
    unet = UNet2DConditionModel.from_pretrained(
        pretrained_path, subfolder="unet").to("cuda:0")

    args.use_attn = True
    args.attention_type = "pre_self_step-post_mlp"
    dipgo_helper = UNetHelper(
        args, unet=unet, timesteps=noise_scheduler.timesteps)
    # dipgo_helper.load_state_dict(torch.load(args.pruner_model_path))
    dipgo_helper.eval()

    with open("save/gates/sd15_gates_0.8.pkl", "rb") as file:
        gates = pickle.load(file)
        dipgo_helper.all_timestep_gates = gates

    # gates = dipgo_helper.prune(0.8)
    dipgo_helper.enable()
    dipgo_helper = dipgo_helper.cuda()
    dipgo_helper.get_average_flops()

    pipe = StableDiffusionPipeline.from_pretrained(
        pretrained_path,
        unet=unet,
        scheduler=noise_scheduler,
        safety_checker=None)
    pipe = pipe.to("cuda")

    print(dipgo_helper.timesteps)

    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-g-14", pretrained="laion2b_s34b_b88k")
    tokenizer = open_clip.get_tokenizer("ViT-g-14")

    print("Warmup GPU...")
    for _ in range(1):
        set_random_seed(args.seed)
        prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
        _ = pipe(prompt)

    for i in range(len(prompts)):
        prompt = prompts[i]["Prompt"]

        dipgo_helper.disable()

        set_random_seed(args.seed)
        start_time = time.time()
        image1 = pipe(prompt, num_inference_step=args.steps).images[0]
        origin_time = time.time() - start_time

        print("Enable DiP-GO...")
        dipgo_helper.enable()
        dipgo_helper.all_timestep_gates = gates
        set_random_seed(args.seed)

        start_time = time.time()
        image2 = pipe(prompt, num_inference_step=args.steps).images[0]
        dipgo_time = time.time() - start_time

        origin_score = calculate_clip_score(model, [prompt], [image1])
        dipgo_score = calculate_clip_score(model, [prompt], [image2])

        os.makedirs("vis_cases/", exist_ok=True)
        image1.save(f"vis_cases/{i}_sd_{prompt}.png")
        image2.save(f"vis_cases/{i}_sd_prune0.8_{prompt}.png")

        print(
            "Done!\n"
            "Original Pipeline: {:.2f} seconds, CLIP-Score = {}.\n"
            "DiP-GO: {:.2f} seconds, Speedup Ratio = {:.2f}, CLIP-Score = {}.".format(
                origin_time,
                origin_score,
                dipgo_time,
                origin_time /
                dipgo_time,
                dipgo_score,
            ))

        # exit()
