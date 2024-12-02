# 1. 读取文件夹下的所有内容
# 2. 读取相关的内容

from pathlib import Path
import json

content_dragging = Path("/home/yuyangxin/data/experiment/examples/description")


def get_modify_instruction(editing_procedure, modify):
    start_point, end_point = modify["start_point"], modify["end_point"]
    return f"{editing_procedure} from position {start_point} to {end_point}"


def get_resizing_instruction(editing_procedure, resizing_scale):
    return f"{editing_procedure} with resizing scale {resizing_scale}"


def get_content_dragging(editing_procedure, modify):
    start_point, end_point = modify["start_point"], modify["end_point"]
    return f"{editing_procedure} from position {start_point} to {end_point}"


def get_res(info_func, type):
    # 读取目录下的所有文件
    res = []
    target_dir = Path(f"/home/yuyangxin/data/experiment/examples/{type}")
    for file in content_dragging.iterdir():
        # 加载为json的内容
        with open(file) as f:
            content = json.load(f)

        edited = target_dir / "sub_images" / f"{file.stem}.png"
        # assert edited.exists(), f"{edited} does not exist"
        if not edited.exists():
            continue

        mask = target_dir / "sub_masks" / f"mask_{file.stem}.png"
        # assert mask.exists(), f"{mask} does not exist"
        if not mask.exists():
            continue

        modify = content["moving_object"]
        editing_procedure = modify["editing_procedure"]

        res.append(
            {
                "origin": content["origin"]["image"],
                "edited": str(edited.absolute().as_posix()),
                "mask": str(mask.absolute().as_posix()),
                "instruction": str(
                    {
                        "Editing Type": type,
                        "Editing Target": content["origin"]["mask_object"],
                        "Editing Procedure": info_func(editing_procedure, modify),
                    }
                ),
            }
        )
    return res


if __name__ == "__main__":
    result = []
    result.extend(get_res(get_modify_instruction, type="moving_object"))
    result.extend(get_res(get_resizing_instruction, type="resizing_object"))
    result.extend(get_res(get_content_dragging, type="content_dragging"))
    # 保存为json文件
    with open("./experiment.json", "w") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)
