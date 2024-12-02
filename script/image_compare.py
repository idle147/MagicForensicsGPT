import tkinter as tk
from tkinter import messagebox, ttk
from pathlib import Path
from PIL import Image, ImageTk, ImageDraw
import json

import cv2
import numpy as np
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from processor.utils import compare_different, create_blending_image, process_mask


def generate_image(real_img_dir: Path, mask_img_dir: Path, modify_dir: Path):
    sub_img_dir = modify_dir / "sub_images"
    sub_img_dir.mkdir(parents=True, exist_ok=True)

    sub_mask_dir = modify_dir / "sub_masks"
    sub_mask_dir.mkdir(parents=True, exist_ok=True)

    fake_img_dir = modify_dir / "images"
    for fake_img_path in fake_img_dir.glob("*.png"):
        file_name = fake_img_path.stem
        
        print(f"Generate {file_name} ...")
        real_img_path = real_img_dir / f"{file_name}.jpg"
        assert real_img_path.exists(), f"{real_img_path} not found"
        real_mask_img_path = mask_img_dir / f"mask_{file_name}.jpg"
        assert real_mask_img_path.exists(), f"{real_mask_img_path} not found"

        real_img = Image.open(str(real_img_path)).convert("RGB")
        fake_img = Image.open(str(fake_img_path)).convert("RGB")
        fake_mask_img_path = modify_dir / "masks" / f"mask_{fake_img_path.stem}.png"
        if fake_mask_img_path.exists():
            ref_mask = process_mask(real_mask_img_path, fake_mask_img_path)
        else:
            ref_mask = Image.open(str(real_mask_img_path)).convert("L")

        # 如果图片尺寸不一致，将fake_img缩放到real_img尺寸
        if real_img.size != fake_img.size:
            fake_img = fake_img.resize(real_img.size)

        mask = compare_different(real_img, fake_img, ref_mask)
        blending_img = create_blending_image(real_img, fake_img, mask)

        img_save_path = sub_img_dir / f"{fake_img_path.stem}.png"
        mask_save_path = sub_mask_dir / f"mask_{fake_img_path.stem}.png"
        blending_img.save(img_save_path)
        mask.save(mask_save_path)


if __name__ == "__main__":
    real_img_dir = Path(r"/home/yuyangxin/data/experiment/examples/images")
    mask_img_dir = Path(r"/home/yuyangxin/data/experiment/examples/masks")
    fake_img_dir = Path(r"/home/yuyangxin/data/experiment/examples/content_dragging")
    generate_image(real_img_dir, mask_img_dir, fake_img_dir)
