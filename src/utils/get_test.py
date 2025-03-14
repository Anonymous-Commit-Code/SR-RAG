import json

# 读取两个JSON文件
with open('experiments/requirements_output_summary_No_Classify_extract.json', 'r', encoding='utf-8') as f:
    A = json.load(f)
    
with open('datasets/tests/gt.json', 'r', encoding='utf-8') as f:
    B = json.load(f)

# 创建requirement到元素的集合
B_requirements = {item["requirement"] for item in B}
A_requirements = {item["requirement"] for item in A}

# 创建requirement到B中出现顺序的映射
B_requirement_order = {}
for i, item in enumerate(B):
    if item["requirement"] not in B_requirement_order:
        B_requirement_order[item["requirement"]] = i

# 找出A中与B有相同requirement的元素 (C文件) - 按照B中的requirement顺序排列
C = [item for item in A if item["requirement"] in B_requirements]
C.sort(key=lambda item: B_requirement_order[item["requirement"]])

# 找出B中A没有的元素 (D文件) - 保持B中的顺序
D = [item for item in B if item["requirement"] not in A_requirements]

# 写入结果到新文件
with open('C.json', 'w', encoding='utf-8') as f:
    json.dump(C, f, indent=4, ensure_ascii=False)
    
with open('D.json', 'w', encoding='utf-8') as f:
    json.dump(D, f, indent=4, ensure_ascii=False)

print(f"C.json包含{len(C)}个元素，D.json包含{len(D)}个元素")


