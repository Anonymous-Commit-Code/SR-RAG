import json
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

# 添加推理路径
import sys
sys.path.append("src")
sys.path.append(".")
from modules.generator.client.openai_client import OpenAIClient
from modules.generator.generators import StraightSafetyRequirementGenerator
from config import get_data_path, OUTPUT_CONFIG, CONCURRENCY_CONFIG

def read_json_file(json_path):
    """
    读取JSON文件并返回数据
    
    :param json_path: JSON文件路径
    :return: JSON数据列表
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        raise FileNotFoundError(f"文件未找到: {json_path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON格式错误: {e}")

def process_chunk(chunk, chunk_id, output_dir):
    """处理单个数据块"""
    try:
        generator = StraightSafetyRequirementGenerator(model="qwq")
        # 将chunk转换为期望的格式
        input_data = {
            "func_requirement": chunk["requirement"]
        }
        output_json = generator.generate([input_data])
        output_json = {"func_requirement": chunk["requirement"], **output_json}
        save_chunk_result(output_json, chunk_id, output_dir)
        return output_json
    except Exception as e:
        print(f"错误：{e}，数据块ID {chunk_id}")
        return None

def save_chunk_result(result, chunk_id, output_dir):
    """
    保存单个数据块的处理结果
    
    :param result: 处理结果
    :param chunk_id: 数据块ID
    :param output_dir: 输出目录
    """
    try:
        if result is not None:
            output_file = f"{output_dir}/chunk_{chunk_id}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=4)
    except Exception as e:
        print(f"保存文件失败 {output_file}: {e}")

def merge_results(output_dir):
    """
    合并所有chunk的结果
    
    :param output_dir: 输出目录
    :return: 合并后的结果列表
    """
    results = []
    try:
        chunk_files = sorted([f for f in os.listdir(output_dir) if f.startswith('chunk_')])
        
        for chunk_file in chunk_files:
            try:
                with open(f"{output_dir}/{chunk_file}", 'r', encoding='utf-8') as f:
                    chunk_result = json.load(f)
                    results.append(chunk_result)
            except Exception as e:
                print(f"读取文件失败 {chunk_file}: {e}")
                
    except Exception as e:
        print(f"合并结果时出错: {e}")
            
    return results

def main():

    # 配置参数
    json_path = get_data_path("test_data")  # JSON输入文件路径
    output_dir = f"{OUTPUT_CONFIG['output_dir']}_qwq"  # 存放各个chunk结果的目录
    final_output = "requirements_processed_qwq.json"  # 最终合并的输出文件
    max_workers = CONCURRENCY_CONFIG["max_workers"]  # 同时运行的线程数

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 读取JSON数据
    chunks = read_json_file(json_path)
    
    # 并行处理每个数据块
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for chunk in chunks:
            chunk_id = chunk["id"]
            futures[executor.submit(process_chunk, chunk, chunk_id, output_dir)] = chunk_id
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="处理数据块"):
            try:
                chunk_id = futures[future]
                result = future.result()
                if result is not None:
                    results.append(result)
            except Exception as e:
                print(f"数据块处理失败：{e}")

    # 合并所有结果
    final_results = merge_results(output_dir)

    # 保存最终合并的结果
    try:
        with open(final_output, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, ensure_ascii=False, indent=4)
        print(f"最终结果已保存至：{final_output}")
    except Exception as e:
        print(f"保存最终结果失败: {e}")

    # 统计处理结果
    total_requirements = len(final_results)
    print(f"总共处理的需求数量：{total_requirements}")

if __name__ == "__main__":
    main() 