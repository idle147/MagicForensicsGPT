from PIL import Image, ImageDraw


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
    if len(position) % 2 != 0:
        raise ValueError("Position must have an even number of elements")

    return [coord / ratio for coord, ratio in zip(position, [scale_factor, scale_factor] * (len(position) // 2))]


def resize_back_to_original(scaled_img, original_size):
    """
    将缩放后的图片缩放回原来的大小。

    参数:
    scaled_img (PIL.Image): 缩放后的图片。
    original_size (tuple): 原始图片尺寸 (宽, 高)。

    返回:
    PIL.Image: 缩放回原来大小的图片。
    """
    return scaled_img.resize(original_size, Image.BICUBIC)


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


def mask_change(mask, start_point, end_point, type="moving") -> Image:
    # Unpack start and end points
    start_x, start_y = start_point
    end_x, end_y = end_point

    # Get mask dimensions
    width, height = mask.size

    # Load mask data
    mask_data = mask.load()

    # Find all pixels with value 255
    region = [(x, y) for y in range(height) for x in range(width) if mask_data[x, y] == 255]

    # Calculate offset
    offset_x = end_x - start_x
    offset_y = end_y - start_y

    # Determine the bounding box of the region
    if region:
        min_x = min(x for x, y in region)
        max_x = max(x for x, y in region)
        min_y = min(y for x, y in region)
        max_y = max(y for x, y in region)

        # Adjust offset to ensure the region stays within bounds
        offset_x = max(-min_x, min(offset_x, width - 1 - max_x))
        offset_y = max(-min_y, min(offset_y, height - 1 - max_y))

        if type == "moving":
            # Remove the region from the original position
            for x, y in region:
                mask_data[x, y] = 0

        # Move the region to the new position
        for x, y in region:
            new_x = x + offset_x
            new_y = y + offset_y
            if 0 <= new_x < width and 0 <= new_y < height:
                mask_data[new_x, new_y] = 255

    return mask


if __name__ == "__main__":
    # 示例用法
    mask = Image.new("1", (10, 10))
    mask_data = mask.load()
    for i in range(3):
        for j in range(3):
            mask_data[2 + i, 2 + j] = 1
    start_point = (3, 3)
    end_point = (6, 6)
    mask = mask_change(mask, start_point, end_point, type="moving")
    mask.show()
