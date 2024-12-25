import json
from pathlib import Path
import pandas as pd


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def main(target_path: Path):
    # 存储每个 dataset_name 和对应的 error_count 和 count
    results = []

    # 获取当前路径下的所有文件夹
    for dir_path in target_path.iterdir():
        mismatch_count, error_count = 0, 0
        fake_mismatch, real_mismatch = 0, 0  # 初始化错误匹配为 fake 和 real 的计数
        total_count = 0
        dataset_name = dir_path.stem
        for i in dir_path.glob("*.json"):
            # 获取冲突的部分
            total_count += 1
            data = read_json(i)
            with_ref, without_ref = data["with_ref"], data["without_ref"]
            if "error" in without_ref or "error" in with_ref:
                error_count += 1
            elif with_ref["real_or_fake"] != without_ref["real_or_fake"]:
                mismatch_count += 1
                # 根据 mismatched 的具体值统计 fake 或 real
                if without_ref["real_or_fake"] == "fake":
                    fake_mismatch += 1
                elif without_ref["real_or_fake"] == "real":
                    real_mismatch += 1

        # 计算准确率
        print("计算总数: ", total_count)
        accuracy = (total_count - error_count - mismatch_count) / total_count

        # 将结果存储到列表中
        results.append(
            {
                "dataset_name": dataset_name,
                "error_count": error_count,
                "count": mismatch_count,
                "fake_mismatch": fake_mismatch,
                "real_mismatch": real_mismatch,
                "accuracy": accuracy,
            }
        )

    # 创建 DataFrame
    df = pd.DataFrame(results)

    # 将结果转换为 Markdown 格式并打印
    markdown_table = df.to_markdown(index=False)
    print(markdown_table)


if __name__ == "__main__":
    main(Path("/home/yuyangxin/data/experiment/document_example"))
