import sys

sys.path.append(f"src")

import json
from modules.inference import InferenceEngine
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

def process_single_requirement(engine: InferenceEngine, req: dict) -> dict:
    """处理单个需求的安全性需求
    
    Args:
        engine: 推理引擎实例
        req: 需求数据字典
    
    Returns:
        dict: 处理后的需求数据
    """
    try:
        if req["status"]=="failed":
            return req
        # 准备输入数据
        safety_criterions = req.get("processed_safety_criterions", [])
        for item in req["processed_results"]:
            item.pop("filtered_safety_criterions", None)
            item.pop("status", None)
        requirements = req["processed_results"]
        
        # 生成新的安全性需求
        new_safety_requirements = engine.generate_safety_requirements(
            safety_criterions,
            requirements
        )
        
        # 更新安全性需求
        req["safety_requirements"] = new_safety_requirements
        return req
        
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
        engine = InferenceEngine("datasets/table/安全性分析准则_all.json")
        
        # 读取JSON文件
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        
        # 设置最大线程数
        max_workers = min(10, len(data["processed_requirements"]))
        
        # 使用线程池处理需求
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_req = {
                executor.submit(process_single_requirement, engine, req): req 
                for req in data["processed_requirements"]
            }

            # 使用tqdm显示进度
            processed_requirements = []
            with tqdm(total=len(future_to_req), desc="处理需求") as pbar:
                for future in as_completed(future_to_req):
                    try:
                        processed_req = future.result()
                        processed_requirements.append(processed_req)
                    except Exception as e:
                        req = future_to_req[future]
                        print(f"处理需求 {req.get('id', '未知')} 时出错: {str(e)}")
                    finally:
                        pbar.update(1)
        
        # 按ID排序处理后的需求
        processed_requirements.sort(key=lambda x: x.get("id", 0))
        
        # 更新数据
        data["processed_requirements"] = processed_requirements
        
        # 保存更新后的JSON文件
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            
        print(f"已成功更新安全性需求并保存到: {output_path}")
        
    except Exception as e:
        print(f"更新安全性需求时出错: {str(e)}")

if __name__ == "__main__":
    input_path = "experiments/requirements_output_summary.json"
    output_path = "experiments/results/SR-RAG/requirements_output_summary_updated_r1.json"
    update_safety_requirements(input_path, output_path) 