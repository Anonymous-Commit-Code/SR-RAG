import sys

sys.path.append(f"src")

import json
from typing import List, Dict, Any, Tuple
from modules.generator.generators import (
    RefinementGenerator,
    ClassificationGenerator,
    FilterGenerator,
    CriterionRewriteGenerator,
    ConsistencyGenerator,
    SafetyRequirementGenerator,
)
from modules.retriever.multi_retriever import MultiRetriever
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from tqdm import tqdm
import numpy as np

class InferenceEngine:
    def __init__(self, knowledge_base_path: str):
        # 初始化各个生成器
        self.refinement_agent = RefinementGenerator()
        self.classification_agent = ClassificationGenerator()
        self.filter_agent = FilterGenerator()
        self.criterion_rewrite_agent = CriterionRewriteGenerator()
        self.consistency_agent = ConsistencyGenerator()
        self.safety_requirement_agent = SafetyRequirementGenerator()

        # 初始化检索器
        self.retriever = MultiRetriever(knowledge_base_path)

        # 最大重试次数
        self.max_retries = 3

        # 设置最大线程数
        self.max_workers = min(10, os.cpu_count() or 1)  # 使用CPU核心数但不超过3个线程
        # 添加线程池
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)

    def refine_requirement(self, requirement: str) -> List[Dict]:
        """细化功能需求"""
        # return self.refinement_agent.generate(requirement)
        return {"need_refine": False, "sub_func_requirements": [requirement]}

    def classify_requirement(self, refined_requirements: List[Dict]) -> List[Dict]:
        """对细化后的需求进行分类"""
        return self.classification_agent.generate(refined_requirements)

    def retrieve_safety_criterions(
        self, classified_requirements: List[Dict], k: int = 5
    ) -> List[str]:
        """检索相关的安全性分析准则"""
        # 构建检索查询
        query = " ".join(classified_requirements)
        # Multi
        return self.retriever.retrieve(query, k_final=k)

        #BM25
        # return self.retriever.bm25.retrieve(query, k_final=k)

        #HNSW
        # return self.retriever.hnsw.retrieve(query, k_final=k)
        # return ["工作状态发生转移时，对功能接口数据的取值进行检查，分析'取值未发生变化'或'取值变化'等情况下输出的正确性，并确保与当前飞行状态和目标状态相关的舵偏角δz、δx和δy计算结果的准确性", "系统应能够根据当前飞行状态和目标状态计算出舵偏角δz、δx和δy"]

    def filter_safety_criterions(self, safety_criterions: List[str]) -> List[str]:
        """过滤安全性分析准则"""
        return self.filter_agent.generate([{"safety_criterions": safety_criterions}])

    def __cal_consistent_by_vec(self, requirements: List[str], criterion: str):

        criterion_vec =self.retriever.hnsw.tokenize(criterion)
        requirements_vec = self.retriever.hnsw.tokenize(requirements)
        scores = (criterion_vec @ requirements_vec.T)[0]

        score = np.mean(scores)
        if score > 0.4:
            return {"is_consistent": True}
        else:
            return {"is_consistent": False}

    def post_process_safety_criterions(
        self, filtered_safety_criterions: List[str], requirements: List[Dict]
    ) -> List[str]:
        """对安全性分析准则进行后处理(重写和一致性检查)"""
        processed_criterions = []

        def process_single_criterion(criterion: str) -> Dict:
            max_retries = 3  # 最大重试次数
            current_criterion = criterion

            for attempt in range(max_retries):
                try:
                    # 进行重写
                    rewrite_result = self.criterion_rewrite_agent.generate(
                        [
                            {
                                "safety_criterion": criterion,
                                "func_requirements": requirements,
                            }
                        ]
                    )

                    # 如果没有被重写，直接返回原始准则，不进行一致性检查
                    if not rewrite_result or not rewrite_result.get(
                        "is_rewrited", False
                    ):
                        return {"is_valid": True, "safety_criterion": criterion}

                    rewritten_criterion = rewrite_result["safety_criterion"]
                    requirements_content=[item["sub_func_requirement"] for item in requirements]

                    # 进行一致性检查

                    # LLMs
                    consistency_result = self.consistency_agent.generate(
                        [
                            {
                                "safety_criterions": rewritten_criterion,
                                "requirements": requirements_content,
                            }
                        ]
                    )
                    
                    # Embedding
                    # consistency_result = self.__cal_consistent_by_vec(requirements_content, rewritten_criterion)
                    # No
                    # consistency_result={"is_consistent": True}


                    # 如果通过一致性检查，返回结果
                    if consistency_result.get("is_consistent", False):
                        return {
                            "is_valid": True,
                            "safety_criterion": rewritten_criterion,
                        }

                    # 如果不通过一致性检查且还有重试机会，使用重写后的准则继续下一轮重试
                    current_criterion = rewritten_criterion

                except Exception as e:
                    print(f"准则后处理时出错 (尝试 {attempt + 1}/{max_retries}): {str(e)}")
                    return {"is_valid": False, "safety_criterion": criterion}

            # 如果所有重试都失败，返回原始准则
            return {"is_valid": False, "safety_criterion": criterion}

        # 使用线程池并行处理每个准则
        futures = []
        for criterion in filtered_safety_criterions:
            futures.append(self.executor.submit(process_single_criterion, criterion))

        # 收集结果
        for future in futures:
            try:
                result = future.result()
                if result["is_valid"]:
                    processed_criterions.append(result["safety_criterion"])
            except Exception as e:
                print(f"处理后处理结果时出错: {str(e)}")

        return list(set(processed_criterions))  # 去重返回

    def process_single_requirement(self, requirement: Dict) -> Dict[str, Any]:
        """处理单个细化后的需求"""
        try:
            # 分类需求
            classified_requirement = self.classify_requirement(
                [{"func_requirement": requirement}]
            )

            # No 分类
            # classified_requirement={}
            # classified_requirement["class"]=""

            # 检索阶段
            safety_criterions = self.retrieve_safety_criterions(
                [classified_requirement["class"] + " " + requirement]
            )

            # 过滤准则
            filtered_safety_criterions = self.filter_safety_criterions(
                safety_criterions
            )

            # No 过滤
            # filtered_safety_criterions={  
            #     "need_filter": False,
            #     "safety_criterions": safety_criterions
            # }

            # 返回中间结果
            return {
                "requirement": requirement,
                "classified_requirement": classified_requirement,
                "filtered_safety_criterions": filtered_safety_criterions,
                "status": "success",
            }

        except Exception as e:
            return {
                "requirement": requirement,
                "status": "failed",
                "message": f"处理过程出错: {str(e)}",
            }

    def process_requirement(self, requirement: str) -> Dict[str, Any]:
        """处理单个功能需求的主流程"""
        # 细化需求
        refined_requirements = self.refine_requirement(requirement)
        if not refined_requirements["need_refine"]:
            refined_requirements["sub_func_requirements"] = [requirement]
        
        # No 细化
        # refined_requirements = {}
        # refined_requirements["sub_func_requirements"] = [requirement]

        # 限制并发处理的需求数量
        batch_size = self.max_workers
        results = []

        # 分批处理细化需求
        for i in range(
            0, len(refined_requirements["sub_func_requirements"]), batch_size
        ):
            batch = refined_requirements["sub_func_requirements"][i : i + batch_size]

            # 使用线程池并行处理当前批次的需求
            future_to_req = {
                self.executor.submit(self.process_single_requirement, req): req
                for req in batch
            }

            # 处理当前批次的结果
            for future in future_to_req:
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    results.append(
                        {
                            "requirement": future_to_req[future],
                            "status": "failed",
                            "message": f"线程执行出错: {str(e)}",
                        }
                    )

        # 汇总结果
        success_results = [r for r in results if r["status"] == "success"]
        if not success_results:
            return {"status": "failed", "message": "所有子需求处理均失败", "details": results}

        # 汇总所有过滤后的准则
        all_filtered_criterions = []
        all_requirements = []
        for result in success_results:
            all_filtered_criterions.extend(
                result["filtered_safety_criterions"]["safety_criterions"]
            )
            all_requirements.append(
                {
                    "sub_func_requirement": result["requirement"],
                    "classified": result["classified_requirement"],
                }
            )
        all_filtered_criterions = list(set(all_filtered_criterions))

        # 使用汇总后的准则和需求进行后处理
        processed_safety_criterions = self.post_process_safety_criterions(
            all_filtered_criterions, all_requirements
        )

        safety_requirements = self.generate_safety_requirements(
            processed_safety_criterions, all_requirements  # 使用后处理后的准则
        )

        # 保存结果到JSON文件
        result = {
            "requirement": requirement,
            "refined_requirements": refined_requirements,
            "processed_results": results,
            "safety_requirements": safety_requirements,
            "original_safety_criterions": all_filtered_criterions,
            "processed_safety_criterions": processed_safety_criterions,
            "status": "success",
        }

        try:
            output_path = "experiments/requirements_output_0309.json"
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存结果到JSON文件时出错: {str(e)}")

        return result

    def generate_safety_requirements(
        self, safety_criterions: List[str], requirements: List[Dict]
    ) -> List[Dict]:
        """生成安全性需求"""
        try:
            # 构造输入数据
            input_data = {
                "func_requirements": requirements,
                "safety_criterions": safety_criterions,
            }

            # 生成安全性需求
            result = self.safety_requirement_agent.generate([input_data])

            return result["safety_requirements"]

        except Exception as e:
            print(f"生成安全性需求时出错: {str(e)}")
            return []

    def __del__(self):
        """确保线程池正确关闭"""
        self.executor.shutdown(wait=True)

    def process_requirements_from_json(self, input_json_path: str) -> Dict[str, Any]:
        """从JSON文件读取并处理多个需求"""
        try:
            # 读取输入JSON文件
            with open(input_json_path, "r", encoding="utf-8") as f:
                requirements = json.load(f)

            # 创建输出目录
            output_dir = "experiments/requirements_results_0309"
            os.makedirs(output_dir, exist_ok=True)

            all_results = []

            # 创建线程池
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # 提交所有任务
                future_to_req = {
                    executor.submit(
                        self._process_single_requirement, req, output_dir
                    ): req
                    for req in requirements
                }

                # 使用tqdm显示进度
                with tqdm(total=len(requirements), desc="处理需求") as pbar:
                    for future in as_completed(future_to_req):
                        req = future_to_req[future]
                        try:
                            result = future.result()
                            all_results.append(result)
                        except Exception as e:
                            print(f"处理需求 {req['id']} 时出错: {str(e)}")
                            all_results.append(
                                {
                                    "id": req["id"],
                                    "status": "failed",
                                    "message": f"处理出错: {str(e)}",
                                    "original_requirement": req["requirement"],
                                    "analysis": req.get("analysis", ""),
                                }
                            )
                        finally:
                            pbar.update(1)

            # 按ID排序结果
            all_results.sort(key=lambda x: x["id"])

            # 保存汇总结果
            summary_result = {
                "total_requirements": len(requirements),
                "successful_requirements": len(
                    [r for r in all_results if r.get("status") != "failed"]
                ),
                "failed_requirements": len(
                    [r for r in all_results if r.get("status") == "failed"]
                ),
                "processed_requirements": all_results,
            }

            summary_path = "experiments/requirements_output_summary_0309.json"
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary_result, f, ensure_ascii=False, indent=2)

            return summary_result

        except Exception as e:
            print(f"处理需求时出错: {str(e)}")
            return {"status": "failed", "message": f"处理需求时出错: {str(e)}"}

    def _process_single_requirement(self, req: Dict, output_dir: str) -> Dict:
        """处理单个需求并保存结果"""
        try:
            requirement_id = req["id"]
            requirement_text = req["requirement"]

            # 处理单个需求
            result = self.process_requirement(requirement_text)

            # 添加原始需求信息
            result.update(
                {
                    "id": requirement_id,
                    "original_requirement": requirement_text,
                    "analysis": req.get("analysis", ""),
                }
            )

            # 保存单个需求的结果
            output_path = os.path.join(output_dir, f"requirement_{requirement_id}.json")
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

            return result

        except Exception as e:
            raise Exception(f"处理需求 {req['id']} 时出错: {str(e)}")


def main():
    # 使用示例
    engine = InferenceEngine("datasets/database.json")

    # 从JSON文件读取需求并处理
    input_json_path = "datasets/testset/gt.json"
    result = engine.process_requirements_from_json(input_json_path)

    if isinstance(result, dict) and result.get("status") != "failed":
        print("\n处理成功:")
        print(f"总共处理了 {result['total_requirements']} 个需求")
        print(f"成功: {result['successful_requirements']} 个")
        print(f"失败: {result['failed_requirements']} 个")
        print(f"结果已保存到 experiments/requirements_results/ 目录")
        print(f"汇总结果已保存到 experiments/requirements_output_summary.json")
    else:
        print("处理失败:", result.get("message", "未知错误"))


if __name__ == "__main__":
    main()
