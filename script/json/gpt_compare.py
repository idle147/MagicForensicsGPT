import json
from pathlib import Path
import pandas as pd


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def main(target_path: Path, is_ref=False):
    # 存储每个 dataset_name 和对应的 error_count 和 count
    results = []

    # 获取当前路径下的所有文件夹
    for dir_path in target_path.iterdir():
        error_count, rejected_count = 0, 0
        fp, fn, tp, tn = 0, 0, 0, 0  # 初始化分类结果计数
        total_count = 0
        positive_count, negative_count = 0, 0  # 初始化正负样本计数
        dataset_name = dir_path.stem
        for i in dir_path.glob("*.json"):
            # 获取冲突的部分
            total_count += 1
            data = read_json(i)
            compare_data = data["with_ref"] if is_ref else data["without_ref"]
            if "error" in compare_data:
                rejected_count += 1
            else:
                ground_truth = data["ground_truth"]
                prediction = compare_data["real_or_fake"]

                # 计算正负样本数
                if ground_truth == "real":
                    positive_count += 1
                elif ground_truth == "fake":
                    negative_count += 1

                if ground_truth == "real":
                    if prediction == "real":
                        tp += 1
                    elif prediction == "fake":
                        fn += 1
                    else:
                        raise ValueError(f"Unknown prediction: {prediction} in {i}")
                elif ground_truth == "fake":
                    if prediction == "fake":
                        tn += 1
                    elif prediction == "real":
                        fp += 1
                    else:
                        raise ValueError(f"Unknown prediction: {prediction} in {i}")
                else:
                    raise ValueError(f"Unknown ground_truth: {ground_truth} in {i}")

        # 计算 FPR 和 FNR 的比率
        actual_negatives = fp + tn
        actual_positives = fn + tp
        fpr_rate = fp / actual_negatives if actual_negatives > 0 else 0
        fnr_rate = fn / actual_positives if actual_positives > 0 else 0

        # 计算准确率（仅考虑已分类的样本）
        classified = tp + tn + fp + fn
        accuracy = (tp + tn) / classified if classified > 0 else 0

        # 将结果存储到列表中
        results.append(
            {
                "dataset_name": dataset_name,
                "positive_count": positive_count,
                "negative_count": negative_count,
                "rejected_count": rejected_count,
                "error_count": fp + fn,
                "FPR": fpr_rate,
                "FNR": fnr_rate,
                "ACC": accuracy,
            }
        )

    # 创建 DataFrame
    df = pd.DataFrame(results)

    # 将结果转换为 Markdown 格式并打印
    markdown_table = df.to_markdown(index=False)
    print(markdown_table)


if __name__ == "__main__":
    main(Path("/home/yuyangxin/data/experiment/document_example"), is_ref=False)
