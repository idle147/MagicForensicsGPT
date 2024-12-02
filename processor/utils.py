from pathlib import Path
from typing import List
from PIL import Image, ImageDraw, ImageChops
import cv2
import numpy as np


def compare_different(src_img, target_img, ref_mask=None):
    # 检查图像是否加载成功
    if src_img is None or target_img is None:
        raise ValueError("One or both images could not be loaded.")

    if src_img.size != target_img.size:
        target_img = target_img.resize(src_img.size)

    # 将PIL图像转换为NumPy数组
    src_array = np.array(src_img)
    target_array = np.array(target_img)

    # 如果是彩色图像，转换为灰度图像
    if len(src_array.shape) == 3:
        src_gray = cv2.cvtColor(src_array, cv2.COLOR_BGR2GRAY)
        target_gray = cv2.cvtColor(target_array, cv2.COLOR_BGR2GRAY)
    else:
        src_gray = src_array
        target_gray = target_array

    diff = cv2.absdiff(src_gray, target_gray)

    # # 分析差异图像的直方图以确定最佳阈值
    hist = cv2.calcHist([diff], [0], None, [256], [0, 256])
    total_pixels = diff.shape[0] * diff.shape[1]
    cumulative_sum = np.cumsum(hist)
    threshold_index = np.argmax(cumulative_sum > total_pixels * 0.95)

    # 应用动态阈值以创建二值掩码
    _, mask = cv2.threshold(diff, threshold_index, 255, cv2.THRESH_BINARY)

    # 使用形态学操作去除噪点
    kernel_size = max(3, int(min(src_img.size) / 100))  # 根据图像大小动态调整
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # 如果提供了参考mask，将其与计算的mask结合
    if ref_mask is not None:
        ref_mask_array = np.array(ref_mask)
        mask = cv2.add(mask, ref_mask_array)

    return Image.fromarray(mask)


def get_scaled_coordinates(position, scale_factor):
    """
    获取缩放后的坐标点。

    参数:
    position (list): 原始坐标点列表，长度应为偶数。
    original_size (tuple): 原始图片尺寸 (宽, 高)。
    scaled_size (tuple): 缩放后图片尺寸 (宽, 高)。

    返回:
    list: 缩放后的坐标点列表。
    """
    if scale_factor == 1:
        return position

    if len(position) % 2 != 0:
        raise ValueError("Position must have an even number of elements")

    return [coord * ratio for coord, ratio in zip(position, [scale_factor, scale_factor] * (len(position) // 2))]


def get_original_coordinates(position, scale_factor):
    """
    获取缩放前的坐标点。

    参数:
    position (list): 缩放后的坐标点列表，长度应为偶数。
    original_size (tuple): 原始图片尺寸 (宽, 高)。
    scaled_size (tuple): 缩放后图片尺寸 (宽, 高)。

    返回:
    list: 缩放前的坐标点列表。
    """
    if scale_factor == 1:
        return position

    if len(position) % 2 != 0:
        raise ValueError("Position must have an even number of elements")

    return [coord / ratio for coord, ratio in zip(position, [scale_factor, scale_factor] * (len(position) // 2))]


def calculate_bbox_center(bbox):
    """
    根据COCO的bbox计算分割区域的中心点。

    参数:
    bbox (tuple): 边界框 (x, y, width, height)。

    返回:
    tuple: 中心点坐标 (center_x, center_y)。
    """
    x, y, width, height = bbox

    center_x = x + width / 2
    center_y = y + height / 2

    return (center_x, center_y)


def draw_arrow(image, start_point, end_point, arrow_color=(255, 0, 0), arrow_width=3, point_radius=5):
    """
    在图像上绘制箭头及起始点和终止点。

    参数:
    image (PIL.Image.Image): 要绘制的图像。
    start_point (tuple): 起始点坐标 (x, y)。
    end_point (tuple): 终止点坐标 (x, y)。
    arrow_color (tuple): 箭头颜色 (R, G, B)。
    arrow_width (int): 箭头宽度。
    point_radius (int): 起始点和终止点的半径。
    """
    start_point = tuple(start_point)
    end_point = tuple(end_point)
    draw = ImageDraw.Draw(image)

    # 画起始点
    draw.ellipse(
        (start_point[0] - point_radius, start_point[1] - point_radius, start_point[0] + point_radius, start_point[1] + point_radius),
        fill=arrow_color,
    )

    # 画终止点
    draw.ellipse(
        (end_point[0] - point_radius, end_point[1] - point_radius, end_point[0] + point_radius, end_point[1] + point_radius),
        fill=arrow_color,
    )

    # 画箭头线
    draw.line(start_point + end_point, fill=arrow_color, width=arrow_width)

    # 计算箭头头部的方向
    arrow_head_length = 10  # 箭头头部的长度
    arrow_head_angle = 30  # 箭头头部的角度

    # 计算箭头头部的两个点
    import math

    angle = math.atan2(end_point[1] - start_point[1], end_point[0] - start_point[0])
    angle1 = angle + math.radians(arrow_head_angle)
    angle2 = angle - math.radians(arrow_head_angle)

    arrow_head_point1 = (end_point[0] - arrow_head_length * math.cos(angle1), end_point[1] - arrow_head_length * math.sin(angle1))
    arrow_head_point2 = (end_point[0] - arrow_head_length * math.cos(angle2), end_point[1] - arrow_head_length * math.sin(angle2))

    # 画箭头头部
    draw.line(end_point + arrow_head_point1, fill=arrow_color, width=arrow_width)
    draw.line(end_point + arrow_head_point2, fill=arrow_color, width=arrow_width)

    return image


def draw_rectangle_from_diagonal(image, points, line_color=(255, 0, 0), line_width=3):
    """
    在图像上绘制一个方框，给定对角线上的两个点。

    参数:
    image (PIL.Image.Image): 要绘制的图像。
    start_point (tuple): 对角线起始点坐标 (x, y)。
    end_point (tuple): 对角线终止点坐标 (x, y)。
    line_color (tuple): 方框线条颜色 (R, G, B)。
    line_width (int): 方框线条宽度。
    """
    draw = ImageDraw.Draw(image)
    start_point_x, start_point_y, end_point_x, end_point_y = points
    # 计算方框的四个角点
    top_left = (min(start_point_x, end_point_x), min(start_point_y, end_point_y))
    bottom_right = (max(start_point_x, end_point_x), max(start_point_y, end_point_y))
    top_right = (bottom_right[0], top_left[1])
    bottom_left = (top_left[0], bottom_right[1])

    # 画方框的四条边
    draw.line(top_left + top_right, fill=line_color, width=line_width)
    draw.line(top_right + bottom_right, fill=line_color, width=line_width)
    draw.line(bottom_right + bottom_left, fill=line_color, width=line_width)
    draw.line(bottom_left + top_left, fill=line_color, width=line_width)

    return image


def mask_moving(mask: Image, start_point: List[float], end_point: List[float], is_moving=True) -> Image:
    """
    # mask是一个PIL.Image对象 像素值255的表示mask需要移动的位置
    # start_point是起始点的(x, y)坐标，end_point是终止点的(x, y)坐标
    # 将mask从start_point, 移动到end_point
    # 如果mask移动后超出图片边界, 则只移动到图片边界
    # 如果is_moving为True, 则删除移动前的位置
    # 如果is_moving为False, 则保留移动前的位置
    """

    # Calculate desired offset
    desired_dx = int(end_point[0] - start_point[0])
    desired_dy = int(end_point[1] - start_point[1])

    # Get mask bounding box
    bbox = mask.getbbox()
    if not bbox:
        return mask  # Empty mask

    left, upper, right, lower = bbox
    width, height = mask.size

    # Calculate maximum allowed movement
    max_dx_left = -left
    max_dx_right = width - right
    max_dy_top = -upper
    max_dy_bottom = height - lower

    # Adjust dx and dy to keep mask within boundaries
    dx = max(min(desired_dx, max_dx_right), max_dx_left)
    dy = max(min(desired_dy, max_dy_bottom), max_dy_top)

    # Shift the mask
    offset_mask = ImageChops.offset(mask, dx, dy)

    if is_moving:
        # Clear original mask area
        cleared_mask = mask.copy()
        draw = ImageDraw.Draw(cleared_mask)
        draw.rectangle(bbox, fill=0)
        # Combine shifted mask with cleared mask
        new_mask = ImageChops.lighter(cleared_mask, offset_mask)
    else:
        # Combine original and shifted mask
        new_mask = ImageChops.lighter(mask, offset_mask)

    return new_mask


def process_mask(src_mask, target_mask, scale_ratio=1.0):
    def read_mask(mask):
        if isinstance(mask, (str, Path)):
            return cv2.imread(str(mask), cv2.IMREAD_GRAYSCALE)
        return mask

    # 读取mask图像
    src_mask_img = read_mask(src_mask)
    target_mask_img = read_mask(target_mask)

    # 确保图像大小相同
    if src_mask_img.shape != target_mask_img.shape:
        raise ValueError("Source and target masks must have the same dimensions.")

    # 合并target_mask和target_mask_img, 要求相同的像素保留, 不同的像素取大值
    combined_mask = np.maximum(src_mask_img, target_mask_img)

    # 如果需要缩放比例，可以在此处应用缩放
    if scale_ratio != 1.0:
        new_size = (int(combined_mask.shape[1] * scale_ratio), int(combined_mask.shape[0] * scale_ratio))
        combined_mask = cv2.resize(combined_mask, new_size, interpolation=cv2.INTER_NEAREST)

    # 转为PIL图像
    combined_mask = Image.fromarray(combined_mask)
    return combined_mask


def scale_mask(src_mask: Image, scale_factor=1.0):
    """
    缩放mask为灰度图，255的表示需要缩放的内容，按照scale_ratio将内容一起缩放。

    :param src_mask: 输入的灰度掩码图像。
    :param scale_ratio: 缩放比例，默认为1.0。
    :return: 缩放后的灰度图像。
    """
    # 创建一个二值化图像，只保留白色物体
    binary_image = src_mask.point(lambda p: p > 128 and 255)

    # 获取白色物体的边界框
    bbox = binary_image.getbbox()

    if bbox:
        # 裁剪出白色物体
        white_object = src_mask.crop(bbox)

        # 计算新的尺寸
        new_size = (int(white_object.width * scale_factor), int(white_object.height * scale_factor))

        # 缩放白色物体
        scaled_object = white_object.resize(new_size, Image.LANCZOS)

        # 创建一个新的空白图像
        new_image = Image.new("L", src_mask.size, 0)

        # 计算放置缩放后物体的位置
        new_bbox = (bbox[0], bbox[1], bbox[0] + scaled_object.width, bbox[1] + scaled_object.height)

        # 将缩放后的物体粘贴到新图像上
        new_image.paste(scaled_object, new_bbox)
        return new_image
    else:
        print("No white object found.")
        return src_mask


def create_blending_image(src_img: Image, target_img: Image, mask):
    # 将target_img中mask的内容合并到src_img中,并返回结果
    src_img = src_img.convert("RGBA")
    target_img = target_img.convert("RGBA")
    mask = mask.convert("L")
    result = Image.composite(target_img, src_img, mask)
    return result


if __name__ == "__main__":
    # 示例用法
    mask = Image.new("1", (10, 10))
    mask_data = mask.load()
    for i in range(3):
        for j in range(3):
            mask_data[2 + i, 2 + j] = 1
    start_point = (3, 3)
    end_point = (6, 6)
    mask = mask_moving(mask, start_point, end_point, is_moving="moving")
    mask.show()
