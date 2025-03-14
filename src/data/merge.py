import os
import json

# 定义存放所有数据的列表
all_data = []

# 目录路径
directory_path = 'table'

# 遍历目录下的所有文件
for filename in sorted(os.listdir(directory_path)):  # 使用 sorted 按照文件名顺序遍历
    if filename.endswith('.json'):
        file_path = os.path.join(directory_path, filename)

        # 打开并读取JSON文件内容
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            print(filename)
            for item in data:
                item["filename"] = filename
            # 将内容拼接到总数组中
            all_data.extend(data)

# 按顺序编号
for index, item in enumerate(all_data, start=1):
    item['序号'] = index

# 将合并后的数据写入一个新的 JSON 文件
output_file_path = 'table/table_all.json'
with open(output_file_path, 'w', encoding='utf-8') as output_file:
    json.dump(all_data, output_file, ensure_ascii=False, indent=4)

print(f"数据已成功写入 {output_file_path}")
