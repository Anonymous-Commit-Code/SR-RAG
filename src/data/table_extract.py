from docx import Document
import json

# 打开Word文档
doc = Document("table/rym.docx")
dic = {"序号": "id", "类别": "", "子类别": "", "数据内容": "", "备注": "", "标识": "", "编号": "", "准则编号": "",
       "编码准则": "", "故障模式": "", "典型应用场景": "", }
# 遍历所有表格
for tid, table in enumerate(doc.tables):
    data = []
    # 遍历表格中的每一行
    json_temp = {}
    head_id = {}
    id_head = {}
    for i, row in enumerate(table.rows):
        if i == 0:
            for cid, cell in enumerate(row.cells):
                json_temp[cell.text] = ""
                head_id[cell.text] = cid
                id_head[cid] = cell.text
            continue

        # 获取行中每个单元格的内容
        row_data = [cell.text for cell in row.cells]
        for cid, cell in enumerate(row.cells):
            if cell.text != "":
                json_temp[id_head[cid]] = cell.text
        json_temp["序号"] = i
        data.append(json_temp.copy())
        print(json_temp)

    with open(f"table_{tid}.json", "w", encoding="utf-8") as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=4)
