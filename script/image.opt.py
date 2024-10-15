import tkinter as tk
from tkinter import messagebox, ttk
from pathlib import Path
from PIL import Image, ImageTk, ImageDraw
import json


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
        self.image_labels, self.text_labels = [], []
        for i, text in enumerate(["Real", "Fake", "Mask", "Cover"]):
            label = ttk.Label(self.inner_frame)
            label.grid(row=i, column=0, padx=5, pady=5, sticky="w")
            self.image_labels.append(label)

            text_label = ttk.Label(self.inner_frame, text=text)
            text_label.grid(row=i, column=1, padx=5, pady=5, sticky="w")
            self.text_labels.append(text_label)

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

    def on_frame_configure(self, event):
        # 更新滚动区域
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def on_scale_change(self, event):
        # 根据缩放比例重新显示图片
        self.show_images()

    def get_img_info(self):
        # 获取图片信息
        real_img_dir = self.img_dir / "images"
        fake_img_dir: Path = self.img_dir / "moving object" / "images"
        mask_img_dir = self.img_dir / "moving object" / "masks"

        ret = []
        for fake_img_path in fake_img_dir.rglob("*.png"):
            file_name = fake_img_path.stem.split("_")[-1]
            mask_img_path = mask_img_dir / f"mask_{file_name}.png"
            if not mask_img_path.exists():
                print(f"[MASK]{mask_img_path} 不存在")
                continue
            real_img_path = real_img_dir / f"{file_name}.jpg"
            if not real_img_path.exists():
                print(f"[REAL]{real_img_path} 不存在")
                continue
            ret.append({"real": real_img_path, "fake": fake_img_path, "mask": mask_img_path})
        return ret

    def show_images(self):
        if not self.img_info:
            messagebox.showwarning("警告", "没有找到图片！")
            return

        if self.current_index >= len(self.img_info):
            self.current_index = len(self.img_info) - 1

        img_info = self.img_info[self.current_index]

        # 打开图片
        real_img = Image.open(img_info["real"])
        fake_img = Image.open(img_info["fake"])
        mask_img = Image.open(img_info["mask"]).convert("L")

        # 创建覆盖图像
        cover_img = real_img.copy()
        mask_rgba = Image.new("RGBA", real_img.size)
        draw = ImageDraw.Draw(mask_rgba)
        draw.bitmap((0, 0), mask_img, fill=(255, 0, 0, 192))  # 75% 透明度
        cover_img = Image.alpha_composite(cover_img.convert("RGBA"), mask_rgba)

        # 获取当前缩放比例
        scale = self.scale_var.get()

        # 调整图片大小
        real_img = real_img.resize((int(real_img.width * scale), int(real_img.height * scale)), Image.Resampling.LANCZOS)
        fake_img = fake_img.resize((int(fake_img.width * scale), int(fake_img.height * scale)), Image.Resampling.LANCZOS)
        mask_img = mask_img.resize((int(mask_img.width * scale), int(mask_img.height * scale)), Image.Resampling.LANCZOS)
        cover_img = cover_img.resize((int(cover_img.width * scale), int(cover_img.height * scale)), Image.Resampling.LANCZOS)

        # 转换为 Tkinter 图片
        self.tk_real = ImageTk.PhotoImage(real_img)
        self.tk_fake = ImageTk.PhotoImage(fake_img)
        self.tk_mask = ImageTk.PhotoImage(mask_img)
        self.tk_cover = ImageTk.PhotoImage(cover_img)

        # 显示图片在2x2网格中
        positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
        images = [self.tk_real, self.tk_fake, self.tk_mask, self.tk_cover]

        for i, (label, image, text_label, pos) in enumerate(zip(self.image_labels, images, self.text_labels, positions)):
            row, col = pos
            label.grid(row=row * 2, column=col, padx=5, pady=5)
            label.config(image=image)
            label.image = image
            text_label.grid(row=row * 2 + 1, column=col, padx=5, pady=5, sticky="w")

        # # 显示图片
        # self.image_labels[0].config(image=self.tk_real)
        # self.image_labels[0].image = self.tk_real

        # self.image_labels[1].config(image=self.tk_fake)
        # self.image_labels[1].image = self.tk_fake

        # self.image_labels[2].config(image=self.tk_mask)
        # self.image_labels[2].image = self.tk_mask

        # self.image_labels[3].config(image=self.tk_cover)
        # self.image_labels[3].image = self.tk_cover

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
