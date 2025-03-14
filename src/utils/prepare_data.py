import json
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

# 添加推理路径
import sys

sys.path.append("src")
from modules.generator.client.openai_client import OpenAIClient
from modules.generator.generators import StraightSafetyRequirementGenerator
from modules.generator.prompt.prompt_factory import create_prompt


def read_json_file(json_path):
    """
    读取JSON文件并返回数据

    :param json_path: JSON文件路径
    :return: JSON数据列表
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def main():
    # 配置参数
    refine_datas = read_json_file("requirements_dsr1_refine.json")
    filter_datas = read_json_file("requirements_qwen_filter.json")
    classify_datas = read_json_file("requirements_qwq_classify.json")

    refine_train = []
    filter_train = []
    classify_train = []

    for refine_data in refine_datas:
        gt = {
            "need_refine": refine_data["need_refine"],
            "sub_func_requirements": refine_data["sub_func_requirements"],
        }
        item = {
            "messages": [
                {
                    "role": "user",
                    "content": create_prompt(
                        [{"func_requirement": refine_data["func_requirement"]}],
                        "src/modules/generator/prompt/prompt_template/refine.txt",
                    ),
                },
                {"role": "assistant", "content": f"```json\n{gt}\n```"},
            ]
        }
        refine_train.append(item)

    for filter_data in filter_datas:
        gt = {
            "need_filter": filter_data["need_filter"],
            "safety_criterions": filter_data["safety_criterions"],
        }
        item = {
            "messages": [
                {
                    "role": "user",
                    "content": create_prompt(
                        [{"safety_criterions": filter_data["input"]}],
                        "src/modules/generator/prompt/prompt_template/filter.txt",
                    ),
                },
                {"role": "assistant", "content": f"```json\n{gt}\n```"},
            ]
        }
        filter_train.append(item)

    for classify_data in classify_datas:
        gt = ({"class_id": classify_data["class_id"], "class": classify_data["class"]},)
        item = {
            "messages": [
                {
                    "role": "user",
                    "content": create_prompt(
                        [{"func_requirement": classify_data["func_requirement"]}],
                        "src/modules/generator/prompt/prompt_template/classify.txt",
                    ),
                },
                {"role": "assistant", "content": f"```json\n{gt}\n```"},
            ]
        }
        classify_train.append(item)


    # 保存最终合并的结果
    with open("refine_output.json", "w", encoding="utf-8") as f:
        json.dump(refine_train, f, ensure_ascii=False, indent=4)
        # 保存最终合并的结果
    with open("filter_output.json", "w", encoding="utf-8") as f:
        json.dump(filter_train, f, ensure_ascii=False, indent=4)
        # 保存最终合并的结果
    with open("classify_output.json", "w", encoding="utf-8") as f:
        json.dump(classify_train, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()
