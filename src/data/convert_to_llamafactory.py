import json
import os
from typing import List, Dict, Any
import sys
sys.path.append(".")

from config import get_prompt_template_path


def load_template(template_name: str) -> str:
    """加载提示模板"""
    try:
        template_path = get_prompt_template_path(template_name)
        with open(template_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"Error loading template {template_name}: {e}")
        return ""


def convert_refinement_data(data: List[Dict]) -> List[Dict]:
    """转换需求细化数据为LLaMA Factory格式"""
    template = load_template("refine")
    converted_data = []
    
    for item in data:
        input_data = item["input"]
        output_data = item["output"]
        
        # 构造输入文本（填充模板）
        instruction = template.replace("!<INPUT 0>!", json.dumps(input_data, ensure_ascii=False))
        
        # 构造输出文本
        if isinstance(output_data, dict):
            output_text = json.dumps(output_data, ensure_ascii=False, indent=2)
        else:
            output_text = str(output_data)
        
        converted_data.append({
            "instruction": instruction,
            "input": "",
            "output": output_text
        })
    
    return converted_data


def convert_classification_data(data: List[Dict]) -> List[Dict]:
    """转换需求分类数据为LLaMA Factory格式"""
    template = load_template("classify")
    converted_data = []
    
    for item in data:
        input_data = item["input"]
        output_data = item["output"]
        
        instruction = template.replace("!<INPUT 0>!", json.dumps(input_data, ensure_ascii=False))
        
        if isinstance(output_data, dict):
            output_text = json.dumps(output_data, ensure_ascii=False, indent=2)
        else:
            output_text = str(output_data)
        
        converted_data.append({
            "instruction": instruction,
            "input": "",
            "output": output_text
        })
    
    return converted_data


def convert_rewriting_data(data: List[Dict]) -> List[Dict]:
    """转换准则重写数据为LLaMA Factory格式"""
    template = load_template("criterion_rewrite")
    converted_data = []
    
    for item in data:
        input_data = item["input"]
        output_data = item["output"]
        
        instruction = template.replace("!<INPUT 0>!", json.dumps(input_data, ensure_ascii=False))
        
        if isinstance(output_data, dict):
            output_text = json.dumps(output_data, ensure_ascii=False, indent=2)
        else:
            output_text = str(output_data)
        
        converted_data.append({
            "instruction": instruction,
            "input": "",
            "output": output_text
        })
    
    return converted_data


def convert_all_datasets():
    """转换所有合成的数据集为LLaMA Factory格式"""
    base_path = "datasets/training_data"
    output_path = "datasets/llamafactory_data"
    
    # 确保输出目录存在
    os.makedirs(output_path, exist_ok=True)
    
    conversions = [
        ("refinement_data.json", "refine.json", convert_refinement_data),
        ("classification_data.json", "classify.json", convert_classification_data),
        ("rewriting_data.json", "filter.json", convert_rewriting_data)  # 注意：重写任务对应filter
    ]
    
    for input_filename, output_filename, convert_func in conversions:
        input_path = os.path.join(base_path, input_filename)
        output_file = os.path.join(output_path, output_filename)
        
        if os.path.exists(input_path):
            try:
                print(f"转换 {input_filename} -> {output_filename}")
                
                # 加载原始数据
                with open(input_path, 'r', encoding='utf-8') as f:
                    raw_data = json.load(f)
                
                # 转换数据
                converted_data = convert_func(raw_data)
                
                # 保存转换后的数据
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(converted_data, f, ensure_ascii=False, indent=2)
                
                print(f"成功转换 {len(converted_data)} 个样本到 {output_filename}")
                
            except Exception as e:
                print(f"转换 {input_filename} 时出错: {e}")
        else:
            print(f"文件不存在: {input_path}")


def create_dataset_info():
    """创建数据集信息文件供LLaMA Factory使用"""
    dataset_info = {
        "refine": {
            "file_name": "refine.json",
            "formatting": "sharegpt",
            "columns": {
                "messages": "messages"
            }
        },
        "classify": {
            "file_name": "classify.json", 
            "formatting": "sharegpt",
            "columns": {
                "messages": "messages"
            }
        },
        "filter": {
            "file_name": "filter.json",
            "formatting": "sharegpt", 
            "columns": {
                "messages": "messages"
            }
        }
    }
    
    output_path = "datasets/llamafactory_data/dataset_info.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dataset_info, f, ensure_ascii=False, indent=2)
    
    print(f"数据集信息已保存到: {output_path}")


if __name__ == "__main__":
    print("开始转换训练数据为LLaMA Factory格式...")
    convert_all_datasets()
    create_dataset_info()
    print("转换完成!") 