import json
import numpy as np
from typing import Dict, Any
from sentence_transformers import SentenceTransformer, util
from scipy.optimize import linear_sum_assignment
from rouge_score import rouge_scorer
import torch
import sys
sys.path.append(".")
from config import EVALUATION_CONFIG

def load_data_as_dict(json_file: str) -> Dict[Any, Dict[str, Any]]:
    """
    从给定 JSON 文件加载数据，并转换成 {id: {"requirement": str, "safety_requirements": [...]} } 的结构
    """
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data_list = json.load(f)
        data_dict = {}
        for item in data_list:
            data_dict[item["id"]] = {
                "requirement": item["requirement"],
                "safety_requirements": item.get("safety_requirements", [])
            }
        return data_dict
    except FileNotFoundError:
        raise FileNotFoundError(f"文件未找到: {json_file}")
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON格式错误: {e}")
    except Exception as e:
        raise Exception(f"加载数据时出错: {e}")

def evaluate_concatenated_text(
    gt_json_path: str,
    model_json_path: str,
    bert_model_name: str = None
):
    """
    针对每个需求（requirement），将其所有安全需求拼接成文本，
    计算GT拼接文本和模型输出拼接文本之间的BERT相似度与ROUGE分数。
    
    :param gt_json_path: Ground Truth 文件路径
    :param model_json_path: Model Output 文件路径
    :param bert_model_name: 用于BERT embedding的模型名称
    :return: 拼接文本评估结果
    """
    if bert_model_name is None:
        bert_model_name = EVALUATION_CONFIG["bert_model_name"]
    
    # 1. 加载数据
    gt_data = load_data_as_dict(gt_json_path)
    model_data = load_data_as_dict(model_json_path)
    
    # 2. 初始化 BERT 模型 & ROUGE scorer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 确定设备
    try:
        bert_model = SentenceTransformer(bert_model_name)
        bert_model.to(device)  # 将模型移动到指定设备
    except Exception as e:
        print(f"加载BERT模型失败: {e}")
        raise
    
    rouge_scorer_obj = rouge_scorer.RougeScorer(
        EVALUATION_CONFIG["rouge_types"], 
        use_stemmer=EVALUATION_CONFIG["use_stemmer"]
    )
    
    # 找出两边都有的id
    common_ids = set(gt_data.keys()).intersection(set(model_data.keys()))
    
    # 按需求ID分别评估
    concat_results = []
    all_bert_sim = []
    all_rouge1_f = []
    all_rouge2_f = []
    all_rougeL_f = []
    
    for req_id in common_ids:
        try:
            gt_item = gt_data[req_id]
            model_item = model_data[req_id]
            
            ref_list = gt_item["safety_requirements"]
            pred_list = model_item["safety_requirements"]
            
            # 如果任一方为空，则跳过
            if not ref_list or not pred_list:
                continue
                
            # 拼接文本 (用换行符连接各项)
            ref_concat = "\n".join(ref_list)
            pred_concat = "\n".join(pred_list)
            
            # 计算BERT相似度
            ref_emb = bert_model.encode([ref_concat], convert_to_tensor=True, device=device)
            pred_emb = bert_model.encode([pred_concat], convert_to_tensor=True, device=device)
            bert_sim = float(util.cos_sim(ref_emb, pred_emb)[0][0].item())
            
            # 计算ROUGE分数
            rouge_scores = rouge_scorer_obj.score(ref_concat, pred_concat)
            rouge1_f = rouge_scores["rouge1"].fmeasure
            rouge2_f = rouge_scores["rouge2"].fmeasure
            rougeL_f = rouge_scores["rougeL"].fmeasure
            
            # 记录结果
            concat_results.append({
                "id": req_id,
                "requirement": gt_item["requirement"],
                "bert_similarity": bert_sim,
                "rouge1_f": rouge1_f,
                "rouge2_f": rouge2_f,
                "rougeL_f": rougeL_f
            })
            
            # 累加统计
            all_bert_sim.append(bert_sim)
            all_rouge1_f.append(rouge1_f)
            all_rouge2_f.append(rouge2_f) 
            all_rougeL_f.append(rougeL_f)
            
        except Exception as e:
            print(f"处理需求 {req_id} 时出错: {e}")
            continue
    
    # 计算平均值
    avg_bert_sim = sum(all_bert_sim) / len(all_bert_sim) if all_bert_sim else 0.0
    avg_rouge1_f = sum(all_rouge1_f) / len(all_rouge1_f) if all_rouge1_f else 0.0
    avg_rouge2_f = sum(all_rouge2_f) / len(all_rouge2_f) if all_rouge2_f else 0.0
    avg_rougeL_f = sum(all_rougeL_f) / len(all_rougeL_f) if all_rougeL_f else 0.0
    
    result = {
        "avg_metrics": {
            "avg_bert_similarity": avg_bert_sim,
            "avg_rouge1_f": avg_rouge1_f,
            "avg_rouge2_f": avg_rouge2_f,
            "avg_rougeL_f": avg_rougeL_f
        },
        "details": concat_results
    }
    
    return result


def evaluate_two_files(
    gt_json_path: str,
    model_json_path: str,
    bert_model_name: str = None,
    bert_threshold: float = None,
    rouge_threshold: float = None
):
    """
    进行两套匹配并输出相应指标：
    1) BERT相似度 -> 匈牙利算法匹配 + 相似度阈值过滤 -> BERT-based全局PRF1
    2) ROUGE(L) -> 匈牙利算法匹配 + ROUGE阈值过滤 -> ROUGE-based全局PRF1
    并统计平均 BERT similarity、平均 ROUGE1/2/L 等。

    :param gt_json_path: Ground Truth 文件路径
    :param model_json_path: Model Output 文件路径
    :param bert_model_name: 用于BERT embedding的模型名称
    :param bert_threshold: BERT相似度阈值，低于此值的匹配对将被过滤掉
    :param rouge_threshold: ROUGE阈值（这里默认对 ROUGE-L_f），低于此值的匹配对将被过滤掉
    :return: 结果字典
    """
    # 使用配置文件的默认值
    if bert_model_name is None:
        bert_model_name = EVALUATION_CONFIG["bert_model_name"]
    if bert_threshold is None:
        bert_threshold = EVALUATION_CONFIG.get("bert_threshold", 0.7)
    if rouge_threshold is None:
        rouge_threshold = EVALUATION_CONFIG.get("rouge_threshold", 0.0)
    
    # 1. 加载数据
    gt_data = load_data_as_dict(gt_json_path)
    model_data = load_data_as_dict(model_json_path)

    # 2. 初始化 BERT 模型 & ROUGE scorer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 确定设备
    try:
        bert_model = SentenceTransformer(bert_model_name)
        bert_model.to(device)  # 将模型移动到指定设备
    except Exception as e:
        print(f"加载BERT模型失败: {e}")
        raise
    
    rouge_scorer_obj = rouge_scorer.RougeScorer(
        EVALUATION_CONFIG["rouge_types"], 
        use_stemmer=EVALUATION_CONFIG["use_stemmer"]
    )

    # ============= 准备统计 (BERT) =============
    total_matched_bert = 0
    total_ref_bert = 0
    total_model_bert = 0
    bert_sim_sum_global = 0.0  # 用于计算全局平均 BERT sim
    bert_pair_count_global = 0  # 有效匹配对数量

    # ============= 准备统计 (ROUGE) =============
    total_matched_rouge = 0
    total_ref_rouge = 0
    total_model_rouge = 0
    rouge1_sum_global = 0.0
    rouge2_sum_global = 0.0
    rougel_sum_global = 0.0
    rouge_pair_count_global = 0  # 有效匹配对数量

    # 找出两边都有的id
    common_ids = set(gt_data.keys()).intersection(set(model_data.keys()))

    # 如果需要的话，也可以记录逐条需求的详情
    details_list = []

    for req_id in common_ids:
        try:
            gt_item = gt_data[req_id]
            model_item = model_data[req_id]

            ref_list = gt_item["safety_requirements"]
            pred_list = model_item["safety_requirements"]

            n = len(ref_list)
            m = len(pred_list)

            if n == 0 and m == 0:
                continue

            # =====================================
            #  Part 1: BERT-based Matching + 阈值过滤
            # =====================================
            ref_emb = bert_model.encode(ref_list, convert_to_tensor=True, device=device)
            pred_emb = bert_model.encode(pred_list, convert_to_tensor=True, device=device)

            sim_matrix = util.cos_sim(pred_emb, ref_emb).cpu().numpy()  # shape=(m,n)
            cost_matrix = 1 - sim_matrix

            # 匈牙利算法
            row_ind, col_ind = linear_sum_assignment(cost_matrix)  # (pred_idx数组, ref_idx数组)
            matched_pairs_bert = list(zip(row_ind, col_ind))

            # 对BERT匹配对应用相似度阈值过滤
            valid_pairs_bert = []
            for (pi, ri) in matched_pairs_bert:
                bert_sim = sim_matrix[pi, ri]
                if bert_sim >= bert_threshold:
                    valid_pairs_bert.append((pi, ri))
                    bert_sim_sum_global += bert_sim
                    bert_pair_count_global += 1

            # 统计BERT匹配结果
            total_matched_bert += len(valid_pairs_bert)
            total_ref_bert += n
            total_model_bert += m

            # =====================================
            #  Part 2: ROUGE-based Matching + 阈值过滤
            #  - 这里示例用 ROUGE-L_f 作为匹配度
            # =====================================
            # 先计算所有 (pred, ref) 的 rouge_l_f
            # 以 cost = 1 - rouge_l_f
            m_rouge_matrix = np.zeros((m, n), dtype=np.float32)
            for i in range(m):
                for j in range(n):
                    pred_text = pred_list[i]
                    ref_text = ref_list[j]
                    scores = rouge_scorer_obj.score(ref_text, pred_text)
                    rouge_l_f = scores["rougeL"].fmeasure  # 也可换成 scores["rouge1"], ...
                    m_rouge_matrix[i, j] = rouge_l_f

            cost_matrix_rouge = 1 - m_rouge_matrix
            row_ind_r, col_ind_r = linear_sum_assignment(cost_matrix_rouge)
            matched_pairs_rouge = list(zip(row_ind_r, col_ind_r))

            # 对ROUGE匹配对应用阈值过滤 (这里默认对 ROUGE-L_f)
            valid_pairs_rouge = []
            for (pi, ri) in matched_pairs_rouge:
                r_l_f = m_rouge_matrix[pi, ri]  # 该pair的 rouge_l_f
                if r_l_f >= rouge_threshold:
                    valid_pairs_rouge.append((pi, ri, r_l_f))

            k_rouge = len(valid_pairs_rouge)
            total_matched_rouge += k_rouge
            total_ref_rouge += n
            total_model_rouge += m

            # 同时统计 ROUGE1/2/L 的累加值（只对有效pair）
            for (pi, ri, _) in valid_pairs_rouge:
                pred_text = pred_list[pi]
                ref_text = ref_list[ri]
                full_scores = rouge_scorer_obj.score(ref_text, pred_text)
                rouge1_sum_global += full_scores["rouge1"].fmeasure
                rouge2_sum_global += full_scores["rouge2"].fmeasure
                rougel_sum_global += full_scores["rougeL"].fmeasure

            rouge_pair_count_global += k_rouge

            # 计算本条需求的 P/R/F1 (ROUGE)
            precision_rouge = k_rouge / m if m > 0 else 0.0
            recall_rouge = k_rouge / n if n > 0 else 0.0
            f1_rouge = (
                2 * precision_rouge * recall_rouge / (precision_rouge + recall_rouge)
                if (precision_rouge + recall_rouge) > 0
                else 0.0
            )

            # （可选）记录详情
            details_list.append({
                "id": req_id,
                "requirement": gt_item["requirement"],
                "ref_count": n,
                "model_count": m,
                # -- BERT
                "bert_matched_pairs": len(valid_pairs_bert),
                "bert_precision": len(valid_pairs_bert) / m if m > 0 else 0.0,
                "bert_recall": len(valid_pairs_bert) / n if n > 0 else 0.0,
                "bert_f1": 2 * (len(valid_pairs_bert) / m) * (len(valid_pairs_bert) / n) / ((len(valid_pairs_bert) / m) + (len(valid_pairs_bert) / n)) if ((len(valid_pairs_bert) / m) + (len(valid_pairs_bert) / n)) > 0 else 0.0,
                # -- ROUGE
                "rouge_matched_pairs": k_rouge,
                "rouge_precision": precision_rouge,
                "rouge_recall": recall_rouge,
                "rouge_f1": f1_rouge
            })

        except Exception as e:
            print(f"处理需求 {req_id} 时出错: {e}")
            continue

    # ========== 全局指标(BERT) ==========
    global_precision_bert = (
        total_matched_bert / total_model_bert if total_model_bert > 0 else 0.0
    )
    global_recall_bert = (
        total_matched_bert / total_ref_bert if total_ref_bert > 0 else 0.0
    )
    global_f1_bert = (
        2 * global_precision_bert * global_recall_bert / (global_precision_bert + global_recall_bert)
        if (global_precision_bert + global_recall_bert) > 0
        else 0.0
    )
    avg_bert_sim = (
        bert_sim_sum_global / bert_pair_count_global if bert_pair_count_global > 0 else 0.0
    )

    # ========== 全局指标(ROUGE) ==========
    global_precision_rouge = (
        total_matched_rouge / total_model_rouge if total_model_rouge > 0 else 0.0
    )
    global_recall_rouge = (
        total_matched_rouge / total_ref_rouge if total_ref_rouge > 0 else 0.0
    )
    global_f1_rouge = (
        2 * global_precision_rouge * global_recall_rouge / (global_precision_rouge + global_recall_rouge)
        if (global_precision_rouge + global_recall_rouge) > 0
        else 0.0
    )

    # 计算平均 ROUGE1/2/L（只统计通过阈值过滤后的匹配对）
    avg_rouge1_f = (
        rouge1_sum_global / rouge_pair_count_global if rouge_pair_count_global > 0 else 0.0
    )
    avg_rouge2_f = (
        rouge2_sum_global / rouge_pair_count_global if rouge_pair_count_global > 0 else 0.0
    )
    avg_rougel_f = (
        rougel_sum_global / rouge_pair_count_global if rouge_pair_count_global > 0 else 0.0
    )

    # 封装结果
    result = {
        "bert_based": {
            "threshold": bert_threshold,
            "precision": global_precision_bert,
            "recall": global_recall_bert,
            "f1_score": global_f1_bert,
            "avg_bert_similarity": avg_bert_sim
        },
        "rouge_based": {
            "threshold": rouge_threshold,
            "precision": global_precision_rouge,
            "recall": global_recall_rouge,
            "f1_score": global_f1_rouge,
            "avg_rouge1_f": avg_rouge1_f,
            "avg_rouge2_f": avg_rouge2_f,
            "avg_rougeL_f": avg_rougel_f
        },
        "details": details_list
    }
    return result


if __name__ == "__main__":
    # 下面是一个示例用法，可根据需要自行修改文件名或阈值
    gt_file = "datasets/tests/gt.json"
    model_file = "experiments/results/No_Classify/v1/C.json"
    
    # 设置 BERT 相似度阈值
    bert_threshold_value = 0.70
    # ROUGE 阈值 (这里默认对 ROUGE-L_f)
    rouge_threshold_value = 0.50
    
    # 原有的评测方法
    eval_result = evaluate_two_files(
        gt_json_path=gt_file,
        model_json_path=model_file,
        bert_model_name="models/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        bert_threshold=bert_threshold_value,
        rouge_threshold=rouge_threshold_value
    )
    
    # 新增的拼接文本评测方法
    concat_result = evaluate_concatenated_text(
        gt_json_path=gt_file,
        model_json_path=model_file,
        bert_model_name="models/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    
    # 写入文件(可改成print)
    output_file = "evaluation_result_No_Classify.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("==== BERT-based Evaluation ====\n")
        f.write(f"Threshold = {eval_result['bert_based']['threshold']}\n")
        f.write(f"Precision: {eval_result['bert_based']['precision']:.4f}\n")
        f.write(f"Recall:    {eval_result['bert_based']['recall']:.4f}\n")
        f.write(f"F1 Score:  {eval_result['bert_based']['f1_score']:.4f}\n")
        f.write(f"Avg BERT Similarity: {eval_result['bert_based']['avg_bert_similarity']:.4f}\n\n")

        f.write("==== ROUGE-based Evaluation ====\n")
        f.write(f"Threshold (L_f): {eval_result['rouge_based']['threshold']}\n")
        f.write(f"Precision: {eval_result['rouge_based']['precision']:.4f}\n")
        f.write(f"Recall:    {eval_result['rouge_based']['recall']:.4f}\n")
        f.write(f"F1 Score:  {eval_result['rouge_based']['f1_score']:.4f}\n")
        f.write(f"Avg ROUGE1_f: {eval_result['rouge_based']['avg_rouge1_f']:.4f}\n")
        f.write(f"Avg ROUGE2_f: {eval_result['rouge_based']['avg_rouge2_f']:.4f}\n")
        f.write(f"Avg ROUGEL_f: {eval_result['rouge_based']['avg_rougeL_f']:.4f}\n\n")

        # 新增拼接文本评估结果
        f.write("==== Concatenated Text Evaluation ====\n")
        f.write(f"Avg BERT Similarity: {concat_result['avg_metrics']['avg_bert_similarity']:.4f}\n")
        f.write(f"Avg ROUGE1_f: {concat_result['avg_metrics']['avg_rouge1_f']:.4f}\n")
        f.write(f"Avg ROUGE2_f: {concat_result['avg_metrics']['avg_rouge2_f']:.4f}\n")
        f.write(f"Avg ROUGEL_f: {concat_result['avg_metrics']['avg_rougeL_f']:.4f}\n\n")

        # 详情部分
        f.write("==== Detailed Evaluation per Requirement ====\n")
        for item in eval_result["details"]:
            f.write(f"[ID={item['id']}] requirement: {item['requirement']}\n")
            f.write(f"  -> ref_count = {item['ref_count']}, model_count = {item['model_count']}\n")
            f.write("  -> BERT matched_pairs = {}, precision = {:.4f}, recall = {:.4f}, f1 = {:.4f}\n".format(
                item["bert_matched_pairs"], item["bert_precision"], item["bert_recall"], item["bert_f1"]
            ))
            f.write("  -> ROUGE matched_pairs = {}, precision = {:.4f}, recall = {:.4f}, f1 = {:.4f}\n".format(
                item["rouge_matched_pairs"], item["rouge_precision"], item["rouge_recall"], item["rouge_f1"]
            ))
            
            # 添加拼接文本的评估结果
            for concat_item in concat_result["details"]:
                if concat_item["id"] == item["id"]:
                    f.write("  -> Concat BERT similarity = {:.4f}, ROUGE1_f = {:.4f}, ROUGE2_f = {:.4f}, ROUGEL_f = {:.4f}\n".format(
                        concat_item["bert_similarity"], concat_item["rouge1_f"], 
                        concat_item["rouge2_f"], concat_item["rougeL_f"]
                    ))
                    break
                    
            f.write("-----------------------------------------\n\n")

    print(f"评测结果已写入文件: {output_file}")
