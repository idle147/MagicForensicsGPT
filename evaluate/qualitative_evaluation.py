# evaluate similarity between images before and after dragging
from torchvision import transforms
import cv2
import numpy as np
from PIL import Image
import torch
import lpips
import clip


class ImageSimilarityEvaluator:
    def __init__(self, device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 初始化LPIPS和CLIP模型
        # LPIPS的距离值越小，表示两幅图像在感知上越相似；距离值越大，表示感知差异越大。
        # CLIP相似度，可以实现更自然和直观的人机交互，因为它能够理解并关联视觉和语言信息
        self.loss_fn_alex = lpips.LPIPS(net="alex").to(self.device)
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device, jit=False)

    def preprocess_image(self, image_path, for_clip=False):
        if for_clip:
            return self.clip_preprocess(Image.open(image_path)).unsqueeze(0).to(self.device)
        # 读取并预处理图像
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"图像未找到：{image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 127.5 - 1  # 归一化到[-1, 1]
        image = np.transpose(image, (2, 0, 1))  # 转换为C,H,W格式
        image = torch.from_numpy(image).unsqueeze(0).to(self.device)
        return image

    def evaluate(self, src_path, edited_path):
        source_image = self.preprocess_image(src_path)
        source_image_clip = self.preprocess_image(src_path, for_clip=True)

        target_image = self.preprocess_image(edited_path)
        target_image_clip = self.preprocess_image(edited_path, for_clip=True)

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

        # 输出结果
        print(f"两张图片之间的 LPIPS 距离：{lpips_value}")
        print(f"两张图片之间的 CLIP 相似度：{clip_similarity}")

        # 返回结果
        return lpips_value, clip_similarity


if __name__ == "__main__":
    evaluator = ImageSimilarityEvaluator()
    evaluator.evaluate(
        "/home/yuyangxin/data/experiment/examples/images/000000000673.jpg",
        "/home/yuyangxin/data/experiment/examples/content_dragging/images/000000000673.png",
    )
