import sys

sys.path.append(f"src")

import json
from tqdm import tqdm


def process_single_requirement(req: dict) -> dict:
    """处理单个需求的安全性需求

    Args:
        req: 需求数据字典

    Returns:
        dict: 处理后的需求数据
    """
    try:
        if req["status"] == "failed":
            return None

        new_req = {}
        new_req["requirement"] = req["requirement"]
        new_req["original_safety_criterions"] = req["original_safety_criterions"]
        new_req["safety_requirements"] = req["safety_requirements"]
        new_req["全面性"] = 0
        new_req["一致性"] = 0
        new_req["可执行性"] = 0
        new_req["可读性"] = 0

        return new_req

    except Exception as e:
        print(f"处理需求 {req.get('id', '未知')} 时出错: {str(e)}")
        return req


def update_safety_requirements(input_path: str, output_path: str):
    """更新JSON文件中的安全性需求

    Args:
        input_path: 输入JSON文件路径
        output_path: 输出JSON文件路径
    """
    try:
        # 初始化推理引擎
        new_data = []

        # 读取JSON文件
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for req in tqdm(data["processed_requirements"]):
            new_req = process_single_requirement(req)
            if req:
                new_data.append(new_req)

        # 保存更新后的JSON文件
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(new_data, f, ensure_ascii=False, indent=2)

        print(f"已成功更新安全性需求并保存到: {output_path}")

    except Exception as e:
        print(f"更新安全性需求时出错: {str(e)}")


if __name__ == "__main__":
    input_path = (
        "experiments/requirements_output_summary_No_Classify.json"
    )
    output_path = "experiments/requirements_output_summary_No_Classify_extract.json"
    update_safety_requirements(input_path, output_path)
