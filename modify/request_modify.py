import json
from pathlib import Path
import requests

url = "http://0.0.0.0:9865/inpaint"
target_path = Path("./test")

# 遍历target_path下所有的json文件
for json_file in target_path.glob("*.json"):
    # 读取json文件
    with open(json_file, "r") as f:
        json_data = json.load(f)

    object_referring = json_data["modification"]["object_info"]["object_referring"]
    # 发送请求
    data = {
        "image_path": json_data["origin_img_path"],
        "instruction": json_data["modification"]["modify"][0]["procedure"],
        "save_dir": target_path.absolute().as_posix(),
    }
    response = requests.post(url, json=data)
    print(response.text)
