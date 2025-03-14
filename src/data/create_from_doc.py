import json
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
import docx
import math
import os

# 添加推理路径
import sys
sys.path.append("src")
from modules.generator.client.openai_client import OpenAIClient
from modules.generator.prompt.prompt_factory import create_prompt
from modules.generator.generators import ClassificationGenerator

def read_doc_by_chunks(doc_path, chunk_size=500):
    """
    从Word文档中读取文本，并按指定大小分块
    
    :param doc_path: Word文档路径
    :param chunk_size: 每个文本块的大小（字符数）
    :return: 文本块列表
    """
    doc = docx.Document(doc_path)
    full_text = ""
    for para in doc.paragraphs:
        full_text += para.text + "\n"
    
    # 分割文本为指定大小的块
    chunks = []
    for i in range(0, len(full_text), chunk_size):
        chunk = full_text[i:i + chunk_size]
        if chunk.strip():  # 只添加非空的文本块
            chunks.append({"document": chunk.strip()})
    return chunks

def extract_content(response):
    """
    提取 API 响应内容
    """
    if isinstance(response, dict) and "choices" in response:
        return response["choices"][0]["message"]["content"].strip()
    elif isinstance(response, dict) and "content" in response:
        return response["content"][0]["text"].strip()
    else:
        return response.choices[0].message.content.strip()

def clean_response(response):
    """
    清理 API 响应内容，移除非 JSON 格式的包裹标记
    """
    return re.sub(r'^```json|```$', '', response.strip(), flags=re.MULTILINE).strip()

def save_chunk_result(result, chunk_index, output_dir):
    """
    保存单个文本块的处理结果
    
    :param result: 处理结果
    :param chunk_index: 文本块索引
    :param output_dir: 输出目录
    """
    if result is not None:
        output_file = f"{output_dir}/chunk_{chunk_index}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=4)

def process_batch(batch, batch_index, output_dir):
    """处理单个文本块"""
    try:
        generator = ClassificationGenerator()
        output_json = generator.generate([batch])
        save_chunk_result(output_json, batch_index, output_dir)
        return output_json
    except Exception as e:
        print(f"错误：{e}，文本块索引 {batch_index}")
        return None

def merge_results(output_dir):
    """
    合并所有chunk的结果
    
    :param output_dir: 输出目录
    :return: 合并后的结果列表
    """
    results = []
    chunk_files = sorted([f for f in os.listdir(output_dir) if f.startswith('chunk_')])
    
    for chunk_file in chunk_files:
        with open(f"{output_dir}/{chunk_file}", 'r', encoding='utf-8') as f:
            chunk_result = json.load(f)
            results.extend(chunk_result)
            
    return results

def main():
    # 初始化 OpenAI 客户端
    global client
    client = OpenAIClient("deepseek")

    # 配置参数
    doc_path = "/Users/renyuming/Desktop/SR-RAG/datasets/table/飞行控制.docx"  # 请替换为实际的文档路径
    output_dir = "output_chunks"  # 存放各个chunk结果的目录
    final_output = "requirements_output.json"  # 最终合并的输出文件
    max_workers = 1  # 同时运行的线程数

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 读取文档内容
    text_chunks = read_doc_by_chunks(doc_path)
    
    # 并行处理每个500字的文本块
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for i, chunk in enumerate(text_chunks):
            futures[executor.submit(process_batch, chunk, i, output_dir)] = i
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="处理文本块"):
            try:
                chunk_index = futures[future]
                result = future.result()
                if result is not None:
                    results.extend(result)
            except Exception as e:
                print(f"文本块处理失败：{e}")

    # 合并所有结果
    final_results = merge_results(output_dir)

    # 保存最终合并的结果
    with open(final_output, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, ensure_ascii=False, indent=4)
    print(f"最终结果已保存至：{final_output}")

    # 统计处理结果
    total_requirements = len(final_results)
    print(f"总共提取的需求数量：{total_requirements}")

if __name__ == "__main__":
    main() 