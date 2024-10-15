import tkinter as tk
from tkinter import messagebox, ttk
from pathlib import Path
from PIL import Image, ImageTk, ImageDraw
import json

import numpy as np


# 定义主应用程序类
class ImageSelectorApp:
    def __init__(self, master, img_dir):
        self.master = master
        self.master.title("图片选择器")
        self.master.geometry("1200x800")  # 设置初始窗口大小

        # 存储所有图片路径和当前显示的图片索引
        self.current_index = 0
        self.img_dir: Path = Path(img_dir)
        self.img_info = self.get_img_info()

        # 存储用户接受的图片路径
        self.accepted_images = []

        # 设置主布局为横向分割
        self.main_frame = ttk.Frame(self.master)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # 创建图片显示区域（左侧）
        self.image_frame = ttk.Frame(self.main_frame)
        self.image_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # 添加滚动条
        self.canvas = tk.Canvas(self.image_frame, borderwidth=0)
        self.scrollbar_y = ttk.Scrollbar(self.image_frame, orient="vertical", command=self.canvas.yview)
        self.scrollbar_x = ttk.Scrollbar(self.image_frame, orient="horizontal", command=self.canvas.xview)
        self.canvas.configure(yscrollcommand=self.scrollbar_y.set, xscrollcommand=self.scrollbar_x.set)

        self.scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)
        self.scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # 创建一个内部框架以放置图片标签
        self.inner_frame = ttk.Frame(self.canvas)
        self.canvas.create_window((0, 0), window=self.inner_frame, anchor="nw")

        self.inner_frame.bind("<Configure>", self.on_frame_configure)

        # 初始化图片标签和文本标签
        labels = ["Real", "Real Mask", "Real Cover", "Fake", "Fake Mask", "Fake Cover"]
        self.image_labels, self.text_labels = [], []
        for i, text in enumerate(labels):
            label = ttk.Label(self.inner_frame, text=text)
            row = i // 4  # Integer division to determine the row
            column = i % 4  # Modulo operation to determine the column
            label.grid(row=row, column=column, padx=5, pady=5, sticky="w")
            self.image_labels.append(label)

        # 创建按钮区域（右侧）
        self.button_frame = ttk.Frame(self.main_frame, width=200)
        self.button_frame.pack(side=tk.RIGHT, fill=tk.Y)

        # 添加按钮，并填充整个按钮区域的宽度
        self.prev_button = ttk.Button(self.button_frame, text="上一组", command=self.show_prev_group)
        self.prev_button.pack(pady=10, padx=10, fill=tk.X)

        self.accept_button = ttk.Button(self.button_frame, text="接受", command=self.accept_images)
        self.accept_button.pack(pady=10, padx=10, fill=tk.X)

        self.reject_button = ttk.Button(self.button_frame, text="拒绝", command=self.reject_images)
        self.reject_button.pack(pady=10, padx=10, fill=tk.X)

        self.next_button = ttk.Button(self.button_frame, text="下一组", command=self.show_next_group)
        self.next_button.pack(pady=10, padx=10, fill=tk.X)

        # 可选：添加缩放比例选择
        self.scale_var = tk.DoubleVar(value=1.0)
        self.scale_label = ttk.Label(self.button_frame, text="缩放比例：")
        self.scale_label.pack(pady=(30, 5), padx=10)
        self.scale_slider = ttk.Scale(
            self.button_frame, from_=0.5, to=2.0, orient=tk.HORIZONTAL, variable=self.scale_var, command=self.on_scale_change
        )
        self.scale_slider.pack(pady=5, padx=10, fill=tk.X)

        # 显示第一组图片
        self.show_images()

        # 绑定窗口关闭事件
        self.master.protocol("WM_DELETE_WINDOW", self.on_closing)

    def show_images(self):
        if not self.img_info:
            messagebox.showwarning("警告", "没有找到图片！")
            return

        if self.current_index >= len(self.img_info):
            self.current_index = len(self.img_info) - 1

        img_info = self.img_info[self.current_index]

        # Load images
        real_img, fake_img, real_mask_img, fake_mask_img = self.load_images(img_info)

        # Create cover images
        real_cover_img = self.create_cover_image(real_img, real_mask_img)
        fake_cover_img = self.create_cover_image(fake_img, fake_mask_img)

        # Get current scale
        scale = self.scale_var.get()

        # Display images
        left_positions = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]  # First row  # Second row
        left_images = self.resize_and_to_tkinter_images(
            scale,
            real_img,
            real_mask_img,
            real_cover_img,
            fake_img,
            fake_mask_img,
            fake_cover_img,
        )
        # Display images in the specified positions
        for i, (label, image, pos) in enumerate(zip(self.image_labels, left_images, left_positions)):
            row, col = pos
            label.grid(row=row, column=col, padx=5, pady=5)
            label.config(image=image)
            label.image = image  # Keep a reference to prevent garbage collection

    def on_frame_configure(self, event):
        # 更新滚动区域
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def on_scale_change(self, event):
        # 根据缩放比例重新显示图片
        self.show_images()

    def get_img_info(self):
        # 获取图片信息
        real_img_dir = self.img_dir / "images"
        real_mask_img_dir = self.img_dir / "masks"
        fake_img_dir: Path = self.img_dir / "moving object" / "images"
        fake_mask_img_dir = self.img_dir / "moving object" / "masks"

        ret = []
        for fake_img_path in fake_img_dir.rglob("*.png"):
            file_name = fake_img_path.stem.split("_")[-1]
            fake_mask_img_path = fake_mask_img_dir / f"mask_{file_name}.png"
            if not fake_mask_img_path.exists():
                print(f"[FAKE][MASK]{fake_mask_img_path} 不存在")
                continue
            real_img_path = real_img_dir / f"{file_name}.jpg"
            if not real_img_path.exists():
                print(f"[REAL]{real_img_path} 不存在")
                continue
            # 从real_mask_img_dir中检索包含file_name的字符串
            real_mask_img_path = real_mask_img_dir / f"mask_{file_name}.jpg"
            if not real_mask_img_path.exists():
                print(f"[REAL][MASK]:{real_mask_img_path} 不存在")
            ret.append(
                {
                    "real": real_img_path,
                    "real_mask": real_mask_img_path,
                    "fake": fake_img_path,
                    "fake_mask": fake_mask_img_path,
                }
            )
        return ret

    def load_images(self, img_info):
        real_img = Image.open(img_info["real"])
        fake_img = Image.open(img_info["fake"])
        real_mask_img = Image.open(img_info["real_mask"]).convert("L")
        fake_mask_img = Image.open(img_info["fake_mask"]).convert("L")
        return real_img, fake_img, real_mask_img, fake_mask_img

    def create_cover_image(self, base_img, mask_img):
        cover_img = base_img.copy()
        mask_rgba = Image.new("RGBA", base_img.size)
        draw = ImageDraw.Draw(mask_rgba)
        draw.bitmap((0, 0), mask_img, fill=(255, 0, 0, 180))  # 75% opacity
        return Image.alpha_composite(cover_img.convert("RGBA"), mask_rgba)

    def create_combined_mask(self, mask_img):
        mask_array = np.array(mask_img)
        combined_mask_array = (mask_array // 2).astype(np.uint8)
        return Image.fromarray(combined_mask_array, mode="L")

    @staticmethod
    def resize_and_to_tkinter_images(scale, *images):
        resize_img = [img.resize((int(img.width * scale), int(img.height * scale)), Image.Resampling.LANCZOS) for img in images]
        tk_images = []
        for img in resize_img:
            if img is not None and isinstance(img, Image.Image):
                tk_images.append(ImageTk.PhotoImage(img))
            else:
                tk_images.append(None)
        return tk_images

    def show_prev_group(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.show_images()
        else:
            messagebox.showinfo("提示", "已经是第一组图片了！")

    def show_next_group(self):
        if self.current_index < len(self.img_info) - 1:
            self.current_index += 1
            self.show_images()
        else:
            messagebox.showinfo("提示", "已经是最后一组图片了！")

    def accept_images(self):
        self.accepted_images.append(self.img_info[self.current_index])
        print(f"已接受图片：{self.img_info[self.current_index]}")
        self.show_next_group()

    def reject_images(self):
        self.show_next_group()

    def on_closing(self):
        self.master.destroy()
        print(f"您接受的图片路径列表：{self.accepted_images}")
        # 保存为 JSON 格式
        save_path = "./accept_files.json"
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(self.accepted_images, f, indent=4, ensure_ascii=False)


# 主程序入口
if __name__ == "__main__":
    # 创建主窗口
    root = tk.Tk()
    app = ImageSelectorApp(root, "./examples")
    root.mainloop()
