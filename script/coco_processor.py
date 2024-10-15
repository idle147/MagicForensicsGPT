import json
from pathlib import Path
import random
import shutil
import numpy as np
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
from PIL import Image
import cv2

# import dlib
from facenet_pytorch import MTCNN
import torch


def detect_frontal_faces(image):
    # Initialize MTCNN for face detection
    mtcnn = MTCNN(keep_all=True, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # Load the image
    if isinstance(image, str) or isinstance(image, Path):
        image = cv2.imread(image)
        if image is None:
            print("Could not read the image.")
            return False
        # Convert the image to RGB (MTCNN expects RGB images)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        # Convert the image to RGB (MTCNN expects RGB images)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Convert the image to a PIL Image
    image = Image.fromarray(image)

    # Detect faces
    boxes, _ = mtcnn.detect(image)

    # Check if any faces are detected
    if boxes is not None and len(boxes) > 0:
        return True
    else:
        return False


# Initialize COCO api for instance annotations
data_dir = Path("/home/yuyangxin/data/dataset/MSCOCO")
data_type = "train2017"
ann_file = data_dir / "annotations" / f"instances_{data_type}.json"
caption_file = data_dir / "annotations" / f"captions_{data_type}.json"

save_path = Path("/home/yuyangxin/data/experiment/examples")
# 如果文件夹存在，则删除
if save_path.exists() and save_path.is_dir():
    shutil.rmtree(save_path)

target_mask_dir = save_path / "masks"
target_img_dir = save_path / "images"
save_path.mkdir(parents=True, exist_ok=True)
target_mask_dir.mkdir(parents=True, exist_ok=True)
target_img_dir.mkdir(parents=True, exist_ok=True)

coco, coco_caps = COCO(ann_file), COCO(caption_file)

# Get all image ids and sort them
img_ids = coco.getImgIds()
random.shuffle(img_ids)

# Dictionary to store image paths and corresponding segmentation info
output_dict = {}
count = 0

max_value, min_value = 128, 32

for img_id in img_ids:
    if count > 500:
        break
    ann_ids = coco.getAnnIds(imgIds=img_id)
    anns = coco.loadAnns(ann_ids)
    # 打乱顺序
    np.random.shuffle(anns)

    # Load the image
    img = coco.loadImgs(img_id)[0]
    image_path = data_dir / data_type / img["file_name"]
    if detect_frontal_faces(image_path) == True:
        print(f"{image_path}: 发现人脸,跳过")
        continue

    image = plt.imread(image_path)
    # Get captions for the image
    cap_ids = coco_caps.getAnnIds(imgIds=img_id)
    caps = coco_caps.loadAnns(cap_ids)
    captions = [cap["caption"] for cap in caps]

    for ann in anns:
        if "bbox" in ann:
            x, y, width, height = ann["bbox"]

            if min(width, height) >= min_value and max(width, height) <= max_value and "segmentation" in ann:
                rle = maskUtils.frPyObjects(ann["segmentation"], img["height"], img["width"])
                mask = maskUtils.decode(rle)

                # Ensure the mask is a 2D binary array
                if mask.ndim == 3:
                    mask = mask[:, :, 0]

                # Save the mask image
                mask_image_path = target_mask_dir / f"mask_{image_path.name}"
                mask_image = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
                mask_image.save(mask_image_path)

                # Copy the real image
                shutil.copy(image_path, target_img_dir / img["file_name"])

                # Add the image path and segmentation info to the dictionary
                category_id = ann["category_id"]
                category_name = coco.loadCats(category_id)[0]["name"]

                output_dict[str(image_path)] = {
                    "mask_path": str(mask_image_path),
                    "category": category_name,
                    "ann": ann,
                    "captions": captions,
                }

                count += 1
                break

# 保存结果为Json内容
with open("result.json", "w") as f:
    json.dump(output_dict, f, indent=4)
