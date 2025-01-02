import json
from pathlib import Path
from simple_lama_inpainting import SimpleLama
from PIL import Image
from tqdm import tqdm


def removing_object(save_dir, desc_dir):
    if not save_dir.exists():
        save_dir.mkdir(parents=True, exist_ok=True)

    simple_lama = SimpleLama()
    total_files = len(list(desc_dir.glob("*.json")))
    for f in tqdm(desc_dir.glob("*.json"), desc="Processing files", total=total_files):
        try:
            with open(f, "r", encoding="utf-8") as file:
                info = json.load(file)

            # 假设 info 中包含 origin 键且有 image 和 mask 路径
            original_path = Path(info["origin"]["image"])
            mask_path = Path(info["origin"]["mask"])
            if not original_path.exists() or not mask_path.exists():
                print(f"文件丢失: {original_path} 或 {mask_path}")
                continue

            save_path = save_dir / f"{original_path.stem}.png"
            if save_path.exists():
                print(f"文件已存在: {save_path}")
                continue

            original_img = Image.open(original_path)
            mask_img = Image.open(mask_path).convert("L")

            # 要求original 和mask的尺寸一致
            assert original_img.size == mask_img.size

            result = simple_lama(original_img, mask_img)
            result.save(save_path)

        except (json.JSONDecodeError, KeyError) as e:
            print(f"文件处理错误 {f}: {e}")


if __name__ == "__main__":
    save_path = Path("/home/yuyangxin/data/experiment/examples/removing_object") / "images"
    desc_dir = Path("/home/yuyangxin/data/experiment/examples/description")
    removing_object(save_path, desc_dir)
