import json
from pathlib import Path


def read_json(json_path):
    with open(json_path, "r") as f:
        return json.load(f, strict=False)


def main():
    datasets = ["RealTextManipulation", "SACP_Train", "TextTamper_Test", "TextTamper_Train", "TTL_Train"]
    base_path = Path("/home/yuyangxin/data/experiment/document_example")
    for dataset in datasets:
        json_file_1 = base_path / f"chatgpt_{dataset}.json"
        json_file_2 = base_path / f"new_chatgpt_{dataset}.json"
        json_content = read_json(json_file_1)
        old_len = len(json_content)
        json_content2 = read_json(json_file_2)

        json_content.update(json_content2)
        with open(json_file_1, "w") as f:
            json.dump(json_content, f, indent=4)
        print(f"Merge {json_file_1} (len{old_len})and {json_file_2} (len{len(json_content2)}) to {len(json_content)}")


if __name__ == "__main__":
    main()
