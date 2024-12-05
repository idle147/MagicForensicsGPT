from pathlib import Path
from tqdm import tqdm

from evaluate.gscore_evaluation import GScoreEvaluation
from evaluate.similarity_evaluation import SimilarityEvaluator

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from tqdm import tqdm


def process_image(original_img, f):
    s_res = similarity_evaluator.evaluate(original_img, f)
    g_res = g_score.evaluate(original_img, f)
    if g_res is None:
        return None
    return {
        "stem": f.stem,
        "g_score": g_res.score,
        "lipis": s_res[0],
        "clip": s_res[1],
    }


def main():
    target_dir = Path("/home/yuyangxin/data/experiment/examples")
    modify_types = ["content_dragging", "moving_object", "removing_object", "resizing_object"]
    result_dict = {key: {} for key in modify_types}

    for tmp in modify_types:
        modify_path = target_dir / tmp / "images"
        total_files = len(list(modify_path.glob("*.png")))

        total_g_score = 0
        total_lipis = 0
        total_clip = 0
        total_count = 0  # 记录处理的图片数量

        futures = []
        with ThreadPoolExecutor() as executor:
            for edited_path in modify_path.glob("*.png"):
                original_img = target_dir / "images" / f"{edited_path.stem}.jpg"
                if not original_img.exists():
                    print(f"Original Image not found: {original_img}")
                    continue
                futures.append(executor.submit(process_image, original_img, edited_path))

            for future in tqdm(as_completed(futures), desc=f"Processing modify type: {tmp}", total=total_files):
                result = future.result()
                if result is None:
                    continue

                result_dict[tmp][result["stem"]] = {
                    "g_score": result["g_score"],
                    "lipis": result["lipis"],
                    "clip": result["clip"],
                }
                # 累加得分
                total_g_score += result["g_score"]
                total_lipis += result["lipis"]
                total_clip += result["clip"]
                total_count += 1

        # 计算平均得分
        average_g_score = total_g_score / total_count if total_count > 0 else 0
        average_lipis = total_lipis / total_count if total_count > 0 else 0
        average_clip = total_clip / total_count if total_count > 0 else 0
        print(f"Type[{tmp}] | Total[{total_count}]: g_score[{average_g_score}]; lipis[{average_lipis}]; clip[{average_clip}]")


if __name__ == "__main__":
    g_score = GScoreEvaluation()
    similarity_evaluator = SimilarityEvaluator()
    main()
