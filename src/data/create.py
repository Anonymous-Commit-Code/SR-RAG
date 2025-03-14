import json
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import re


# 添加推理路径
import sys
sys.path.append("src/inference")
sys.path.append("src/inference/prompt")
from modules.generator.client.openai_client import OpenAIClient
from modules.generator.prompt.prompt_factory import create_prompt


def extract_content(response):
    """
    提取 API 响应内容。
    """
    if isinstance(response, dict) and "choices" in response:
        return response["choices"][0]["message"]["content"].strip()
    elif isinstance(response, dict) and "content" in response:
        return response["content"][0]["text"].strip()
    else:
        return response.choices[0].message.content.strip()


def clean_response(response):
    """
    清理 API 响应内容，移除非 JSON 格式的包裹标记。
    """
    return re.sub(r'^```json|```$', '', response.strip(), flags=re.MULTILINE).strip()


def process_batch(batch, batch_index):
    """
    处理一批数据，生成 API 响应并提取处理结果。
    """
    # 将整个 batch 作为 prompt 输入
    prompt = create_prompt(
        [batch],
        "src/modules/generator/prompt/prompt_template/straight_generate_requirement.txt",
    )
    try:
        # 获取 API 响应
        response = extract_content(client.get_response(prompt))
        # 清理响应内容
        cleaned_response = clean_response(response)
        # 解析 JSON
        output_json = json.loads(cleaned_response)
        # 如果返回的响应与输入 batch 不匹配，则记录警告
        if len(output_json) != len(batch):
            print(f"警告：批次索引 {batch_index} 的响应条目数与输入数据不一致")
        return output_json
    except json.JSONDecodeError as e:
        print(f"JSONDecodeError: {e}，批次索引 {batch_index}，响应内容：{response}")
        return [None] * len(batch)
    except Exception as e:
        print(f"错误：{e}，批次索引 {batch_index}")
        return [None] * len(batch)


# 初始化 OpenAI 客户端
client = OpenAIClient("deepseek")

# 批量大小
batch_size = 5
start=1200
end=1400
input_json = "/Users/renyuming/Desktop/SR-RAG/src/data/table/table_all.json"
output_json = f"output_{start}_{end}.json"
with open(input_json, 'r') as f:
    json_data = json.load(f)

# 确保 JSON 文件是一个列表
if not isinstance(json_data, list):
    raise ValueError("输入的 JSON 文件内容必须是一个列表。")


json_data=json_data[start:end]

# 按批次处理数据
results = []
with ThreadPoolExecutor(max_workers=1) as executor:
    futures = {}
    for i in range(0, len(json_data), batch_size):
        batch = json_data[i:i + batch_size]
        futures[executor.submit(process_batch, batch, i // batch_size)] = i
    
    for future in tqdm(as_completed(futures), total=len(futures), desc="处理批次"):
        try:
            batch_index = futures[future]
            result = future.result()
            # 合并每个批次的结果
            results.extend(result)
        except Exception as e:
            print(f"批次处理失败：{e}")

# 保存处理结果到新的 JSON 文件
with open(output_json, 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=4)
print(f"更新后的 JSON 文件已保存至：{output_json}")

# 统计处理失败的条目数量
failed_count = sum(1 for item in results if item is None)
print(f"处理失败的条目数：{failed_count}")