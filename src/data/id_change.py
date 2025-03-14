import json

def assign_ids_to_json(json_file):
    # 读取 JSON 文件
    with open(json_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    # 确保 data 是一个列表
    if not isinstance(data, list):
        raise ValueError("JSON 文件内容应为列表格式")
    
    new_data = []
    for idx, item in enumerate(data, start=1):
        # item.pop("id")
        new_data.append({"id": idx, **item})
    
    # 将更新后的数据写回 JSON 文件
    with open(json_file, 'w', encoding='utf-8') as file:
        json.dump(new_data, file, ensure_ascii=False, indent=4)
    
    print(f"已成功更新 {json_file}，所有元素的 id 已重新分配。")

# 示例使用
if __name__ == "__main__":
    json_filename = "C.json"  # 请替换为你的 JSON 文件名
    assign_ids_to_json(json_filename)
