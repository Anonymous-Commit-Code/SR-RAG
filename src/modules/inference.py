import json
from typing import List, Dict, Any, Tuple, Optional
from modules.generator.generators import (
    RefinementGenerator,
    ClassificationGenerator,
    FilterGenerator,
    CriterionRewriteGenerator,
    ConsistencyGenerator,
    SafetyRequirementGenerator,
    QueryRewriteGenerator,
)
from modules.retriever.multi_retriever import MultiRetriever
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from tqdm import tqdm
import numpy as np
from config import CONCURRENCY_CONFIG, RETRIEVAL_CONFIG


class InferenceEngine:
    def __init__(self, knowledge_base_path: str):
        # 初始化各个生成器
        self.refinement_agent = RefinementGenerator()
        self.classification_agent = ClassificationGenerator()
        self.filter_agent = FilterGenerator()
        self.criterion_rewrite_agent = CriterionRewriteGenerator()
        self.consistency_agent = ConsistencyGenerator()
        self.safety_requirement_agent = SafetyRequirementGenerator()
        self.query_rewrite_agent = QueryRewriteGenerator()

        # 初始化检索器
        self.retriever = MultiRetriever(knowledge_base_path)

        # 回溯和重试次数
        self.max_shallow_backtrack = 3
        self.max_deep_backtrack = 3 # 假设深度回溯也进行3次
        
        # 并发设置
        self.max_workers = CONCURRENCY_CONFIG["max_workers"]
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)

    def __del__(self):
        if self.executor:
            self.executor.shutdown(wait=True)

    def _rewrite_and_verify_criterion(self, criterion: str, sub_requirement: str) -> Tuple[bool, str, Optional[str]]:
        """
        执行浅回溯（重写和验证循环）。
        返回 (是否成功, 最终准则, 失败原因)
        """
        current_criterion = criterion
        feedback = None # 用于存储一致性检查失败的原因
        
        for _ in range(self.max_shallow_backtrack):
            # 1. 重写准则
            rewrite_result = self.criterion_rewrite_agent.generate(
                [{"safety_criterion": current_criterion, "func_requirement": sub_requirement, "feedback": feedback}]
            )
            rewritten_criterion = rewrite_result.get("safety_criterion", current_criterion)

            # 2. 一致性检查
            consistency_result = self.consistency_agent.generate(
                [{"safety_criterions": rewritten_criterion, "requirements": [sub_requirement]}]
            )

            if consistency_result.get("is_consistent", False):
                return True, rewritten_criterion, None # 成功

            # 准备下一次浅回溯
            current_criterion = rewritten_criterion
            feedback = consistency_result.get("reason", "The rewritten criterion is not consistent with the requirement.")
        
        return False, current_criterion, feedback # 浅回溯失败

    def _process_sub_requirement(self, sub_req: str) -> List[str]:
        """
        为单个子需求执行完整的处理流程（分类、检索、回溯、生成）
        """
        valid_criterions = []
        original_query = sub_req

        for deep_attempt in range(self.max_deep_backtrack):
            try:
                # 1. 分类 (仅在第一次深回溯时进行)
                if deep_attempt == 0:
                    classification_result = self.classification_agent.generate([{"func_requirement": original_query}])
                    query_for_retrieval = classification_result.get("class", "") + " " + original_query
                else:
                    # 深回溯：重写查询
                    rewrite_query_result = self.query_rewrite_agent.generate([{"query": original_query}])
                    query_for_retrieval = rewrite_query_result.get("new_query", original_query)

                # 2. 检索
                retrieved_criterions = self.retriever.retrieve(
                    query_for_retrieval, k_final=RETRIEVAL_CONFIG["k_retrieval"]
                )

                # 3. 过滤 (可选，但论文中未明确提及在回溯循环中，此处为保持逻辑完整性)
                # filtered_criterions = self.filter_agent.generate([{"safety_criterions": retrieved_criterions}])
                # criterions_to_process = filtered_criterions.get("filtered_safety_criterions", [])
                criterions_to_process = retrieved_criterions

                # 4. 并行进行浅回溯（重写与验证）
                criterions_after_shallow_backtrack = []
                futures = {self.executor.submit(self._rewrite_and_verify_criterion, c, sub_req): c for c in criterions_to_process}
                
                for future in as_completed(futures):
                    is_success, final_criterion, reason = future.result()
                    if is_success:
                        criterions_after_shallow_backtrack.append(final_criterion)

                # 5. 检查是否成功
                if criterions_after_shallow_backtrack:
                    valid_criterions.extend(criterions_after_shallow_backtrack)
                    return list(set(valid_criterions)) # 成功找到有效准则，退出深回溯

                # 如果没有成功，深回溯将继续
                original_query = query_for_retrieval # 更新查询以备下次重写

            except Exception as e:
                print(f"Error processing sub-requirement '{sub_req}' on deep attempt {deep_attempt+1}: {e}")
                continue # 继续下一次深回溯尝试
        
        print(f"Failed to process sub-requirement '{sub_req}' after all deep backtracking attempts.")
        return []

    def process_requirement(self, requirement: str) -> Dict[str, Any]:
        """
        处理单个功能需求的主流程，遵循论文描述。
        """
        # 1. 需求细化
        refinement_result = self.refinement_agent.generate([{"requirement": requirement}])
        if refinement_result.get("need_refine", False):
            sub_requirements = [item['sub_func_requirement'] for item in refinement_result.get("sub_func_requirements", [])]
        else:
            sub_requirements = [requirement]

        # 2. 并行处理所有子需求
        all_processed_criterions = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_sub_req = {executor.submit(self._process_sub_requirement, sub_req): sub_req for sub_req in sub_requirements}
            for future in as_completed(future_to_sub_req):
                sub_req = future_to_sub_req[future]
                try:
                    criterions = future.result()
                    all_processed_criterions.extend(criterions)
                except Exception as e:
                    print(f"An exception occurred while processing sub-requirement '{sub_req}': {e}")
        
        # 去重
        final_criterions = sorted(list(set(all_processed_criterions)))

        # 3. 最后统一生成安全需求
        if not final_criterions:
             return {
                "original_requirement": requirement,
                "refined_requirements": sub_requirements,
                "final_safety_requirements": []
            }

        safety_requirements_result = self.safety_requirement_agent.generate(
            [{"safety_criterions": final_criterions, "requirements": sub_requirements}]
        )
        
        return {
            "original_requirement": requirement,
            "refined_requirements": sub_requirements,
            "final_safety_requirements": safety_requirements_result.get("safety_requirements", []),
        }

    def process_requirements_from_json(self, input_json_path: str) -> Dict[str, Any]:
        """
        从JSON文件读取需求列表并处理。
        """
        try:
            with open(input_json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error reading or parsing JSON file: {e}")
            return {}

        results = {}
        for item in tqdm(data, desc="Processing Requirements"):
            req_id = item.get("id")
            req_text = item.get("requirement")
            if not req_id or not req_text:
                continue
            
            result = self.process_requirement(req_text)
            results[req_id] = result
        
        return results

    def _save_chunk(self, chunk_data, chunk_index):
        output_dir = "output_chunks"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        file_path = os.path.join(output_dir, f"chunk_{chunk_index}.json")
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(chunk_data, f, ensure_ascii=False, indent=4)

    def _process_single_requirement(self, req: Dict, output_dir: str) -> Dict:
        req_id = req["id"]
        try:
            result = self.process_requirement(req["requirement"])
            # 保存结果到单独的文件
            file_path = os.path.join(output_dir, f"{req_id}.json")
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=4)
            return {"id": req_id, "status": "success"}
        except Exception as e:
            return {"id": req_id, "status": "failed", "reason": str(e)}

def main():
    # 使用示例
    # 请确保你的知识库路径是正确的
    knowledge_base_path = "datasets/table/安全性分析准则_书.json"
    engine = InferenceEngine(knowledge_base_path)

    # 处理单个需求
    test_requirement = "系统应处理油门杆信号"
    result = engine.process_requirement(test_requirement)
    print(json.dumps(result, indent=4, ensure_ascii=False))

    # # 从文件批量处理
    # input_file = "datasets/tests/gt.json" # 你的输入文件
    # all_results = engine.process_requirements_from_json(input_file)
    # with open("output.json", 'w', encoding='utf-8') as f:
    #      json.dump(all_results, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()
