import argparse
import numpy as np
import torch
from ldm.util import instantiate_from_config
import gradio_app.utils as utils
from omegaconf import OmegaConf

MODEL, DEVICE = None, None


def run(image, save_path, instruction, steps=50, center_crop=False):
    # 读取图片
    cropped_image, image = utils.preprocess_image(image, center_crop=center_crop)
    output_image = MODEL.inpaint(cropped_image, instruction, num_steps=steps, device=DEVICE, return_pil=True, seed=0)
    # 保存图片
    utils.save_image(save_path, output_image)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/latent-diffusion/gqa-inpaint-ldm-vq-f8-256x256.yaml",
        help="Path of the model config file",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="models/gqa_inpaint/ldm/model.ckpt",
        help="Path of the model checkpoint file",
    )
    parser.add_argument(
        "--on_cpu",
        action="store_true",
        help="Running the inference code on CPU",
    )
    args = parser.parse_args()

    DEVICE = "cpu" if args.on_cpu else "cuda"

    parsed_config = OmegaConf.load(args.config)
    MODEL = instantiate_from_config(parsed_config["model"])
    model_state_dict = torch.load(args.checkpoint, map_location="cpu")["state_dict"]
    MODEL.load_state_dict(model_state_dict)
    MODEL.eval()
    MODEL.to(DEVICE)
