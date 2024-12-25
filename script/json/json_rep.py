from pathlib import Path
import json
from tqdm import tqdm

target_path = Path("/home/yuyangxin/data/experiment/examples/description")

# 遍历路径下的所有json文件
for i in tqdm(target_path.glob("*.json"), desc="Processing JSON files"):
    try:
        # 加载json
        with open(i, "r", encoding="utf-8") as json_file:
            result = json.load(json_file)

        # 修改键名：将 "moving object" 重命名为 "moving_object"
        if result.get("moving object") is None:
            continue

        result["moving_object"] = result.pop("moving object")

        # 修改 "mask" 字段的值，替换字符串中的 "moving object" 为 "moving_object"
        result["moving_object"]["mask"] = result["moving_object"]["mask"].replace("moving object", "moving_object")

        # 覆写回文件
        with open(i, "w", encoding="utf-8") as json_file:
            json.dump(result, json_file, ensure_ascii=False, indent=4)

    except KeyError as e:
        print(f"文件 {i} 中缺少键：{e}")
    except Exception as e:
        print(f"处理文件 {i} 时出错：{e}")
