# evaluate similarity between images before and after dragging
from torchvision import transforms
import cv2
import numpy as np
from PIL import Image
import torch
import lpips
import clip


class SimilarityEvaluator:
    def __init__(self, device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 初始化LPIPS和CLIP模型
        # LPIPS的距离值越小，表示两幅图像在感知上越相似；距离值越大，表示感知差异越大。
        # CLIP相似度，可以实现更自然和直观的人机交互，因为它能够理解并关联视觉和语言信息
        self.loss_fn_alex = lpips.LPIPS(net="alex").to(self.device)
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device, jit=False)

    def preprocess_images(self, image_path1, image_path2, for_clip=False):
        # Helper function to open and preprocess an image
        def open_and_preprocess(image_path):
            # Open the image
            image = Image.open(image_path)
            
            # If for_clip is True, resize the image before processing
            if for_clip:
                # Resize both images to a fixed size, e.g., 224x224, or any other size you prefer
                target_size = (224, 224)
                image = image.resize(target_size)
                return self.clip_preprocess(image).unsqueeze(0).to(self.device)
            
            # For non-CLIP processing, use OpenCV
            image = cv2.imread(image_path)
            if image is None:
                raise FileNotFoundError(f"图像未找到：{image_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = image.astype(np.float32) / 127.5 - 1  # 归一化到[-1, 1]
            return image

        # Open and preprocess both images
        image1 = open_and_preprocess(image_path1)
        image2 = open_and_preprocess(image_path2)

        if not for_clip:
            # Ensure both images have the same size
            height1, width1, _ = image1.shape
            height2, width2, _ = image2.shape

            # Determine the target size (can be the minimum or maximum of both)
            target_height = min(height1, height2)
            target_width = min(width1, width2)

            # Resize images to the target size
            image1 = cv2.resize(image1, (target_width, target_height))
            image2 = cv2.resize(image2, (target_width, target_height))

            # Convert images to C,H,W format and to torch tensors
            image1 = np.transpose(image1, (2, 0, 1))
            image2 = np.transpose(image2, (2, 0, 1))

            image1 = torch.from_numpy(image1).unsqueeze(0).to(self.device)
            image2 = torch.from_numpy(image2).unsqueeze(0).to(self.device)

        return image1, image2


    def evaluate(self, src_path, edited_path):
        source_image, target_image = self.preprocess_images(src_path, edited_path)
        source_image_clip, target_image_clip = self.preprocess_images(src_path, edited_path, for_clip=True)

        # 计算 LPIPS
        with torch.no_grad():
            lpips_distance = self.loss_fn_alex(source_image, target_image)
            lpips_value = lpips_distance.item()

        # 计算 CLIP 相似度
        with torch.no_grad():
            source_feature = self.clip_model.encode_image(source_image_clip)
            target_feature = self.clip_model.encode_image(target_image_clip)
            source_feature /= source_feature.norm(dim=-1, keepdim=True)
            target_feature /= target_feature.norm(dim=-1, keepdim=True)
            clip_similarity = torch.sum(source_feature * target_feature).item()

        # 返回结果
        return lpips_value, clip_similarity


if __name__ == "__main__":
    evaluator = SimilarityEvaluator()
    evaluator.evaluate(
        "/home/yuyangxin/data/experiment/examples/images/000000000673.jpg",
        "/home/yuyangxin/data/experiment/examples/content_dragging/images/000000000673.png",
    )
