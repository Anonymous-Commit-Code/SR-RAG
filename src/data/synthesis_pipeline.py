import json
import random
import asyncio
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import os
from tqdm import tqdm
import sys
sys.path.append(".")

from modules.generator.client.openai_client import OpenAIClient
from modules.generator.generators import (
    RefinementGenerator,
    ClassificationGenerator,
    CriterionRewriteGenerator,
    ConsistencyGenerator,
)
from config import MODEL_CONFIG, get_data_path, ensure_directories


@dataclass
class SynthesisConfig:
    """数据合成配置"""
    generator_model: str = "deepseek-r1"  # 生成器模型
    jury_models: List[str] = None  # 评审团模型列表
    num_jury_members: int = 3  # 评审团成员数量
    agreement_threshold: float = 0.6  # 评审通过阈值
    max_synthesis_attempts: int = 3  # 单个样本最大合成尝试次数
    batch_size: int = 10  # 批次大小
    
    def __post_init__(self):
        if self.jury_models is None:
            self.jury_models = ["qwen", "llama", "deepseek-v3"]


class DataSynthesizer:
    """数据合成器"""
    
    def __init__(self, config: SynthesisConfig = None):
        self.config = config or SynthesisConfig()
        
        # 使用与inference.py相同的生成器，传入模型参数
        self.refinement_generator = RefinementGenerator(model=self.config.generator_model)
        self.classification_generator = ClassificationGenerator(model=self.config.generator_model)
        self.criterion_rewrite_generator = CriterionRewriteGenerator(model=self.config.generator_model)
        self.consistency_generator = ConsistencyGenerator(model=self.config.generator_model)
        
        # 评审团客户端
        self.jury_clients = [OpenAIClient(model=model) for model in self.config.jury_models[:self.config.num_jury_members]]
        
        # 确保输出目录存在
        ensure_directories()
        
    def _load_safety_criteria(self) -> List[Dict]:
        """加载安全准则数据库"""
        try:
            with open(get_data_path("knowledge_base"), 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading safety criteria: {e}")
            return []
    
    def _load_existing_requirements(self) -> List[Dict]:
        """加载现有的功能需求"""
        try:
            with open(get_data_path("test_data"), 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading existing requirements: {e}")
            return []

    async def _generate_refinement_data(self, requirement: str) -> Optional[Dict]:
        """生成需求细化训练数据"""
        try:
            # 使用与inference.py相同的生成器和prompt模板
            response = self.refinement_generator.generate([{"requirement": requirement}])
            return {
                "input": {"requirement": requirement},
                "output": response
            }
        except Exception as e:
            print(f"Error generating refinement data: {e}")
            return None

    async def _generate_classification_data(self, requirement: str) -> Optional[Dict]:
        """生成需求分类训练数据"""
        try:
            # 使用与inference.py相同的生成器和prompt模板
            response = self.classification_generator.generate([{"func_requirement": requirement}])
            return {
                "input": {"func_requirement": requirement},
                "output": response
            }
        except Exception as e:
            print(f"Error generating classification data: {e}")
            return None

    async def _generate_rewriting_data(self, safety_criterion: str, requirement: str) -> Optional[Dict]:
        """生成准则重写训练数据"""
        # 向安全准则中注入噪声
        noise_phrases = [
            "这是一个重要的安全考虑。",
            "需要特别注意系统的稳定性。",
            "此外，还要考虑用户体验。",
            "根据行业最佳实践，",
            "在实际应用中，"
        ]
        
        # 随机选择1-2个噪声短语插入
        noisy_criterion = safety_criterion
        num_noise = random.randint(1, 2)
        for _ in range(num_noise):
            noise = random.choice(noise_phrases)
            # 在随机位置插入噪声
            insert_pos = random.randint(0, len(noisy_criterion))
            noisy_criterion = noisy_criterion[:insert_pos] + noise + noisy_criterion[insert_pos:]
        
        try:
            # 使用与inference.py相同的生成器和prompt模板
            response = self.criterion_rewrite_generator.generate([{
                "safety_criterion": noisy_criterion,
                "func_requirement": requirement
            }])
            return {
                "input": {
                    "safety_criterion": noisy_criterion,
                    "func_requirement": requirement
                },
                "output": response
            }
        except Exception as e:
            print(f"Error generating rewriting data: {e}")
            return None

    async def _jury_review(self, data_sample: Dict, task_type: str) -> bool:
        """LLM评审团评审数据样本质量"""
        reviews = []
        
        for jury_client in self.jury_clients:
            review_prompt = self._get_review_prompt(data_sample, task_type)
            try:
                review = await jury_client.generate_async(review_prompt)
                # 解析评审结果
                if isinstance(review, dict) and review.get("is_valid", False):
                    reviews.append(True)
                elif isinstance(review, str) and ("valid" in review.lower() or "正确" in review):
                    reviews.append(True)
                else:
                    reviews.append(False)
            except Exception as e:
                print(f"Error in jury review: {e}")
                reviews.append(False)
        
        # 计算同意比例
        agreement_rate = sum(reviews) / len(reviews) if reviews else 0
        return agreement_rate >= self.config.agreement_threshold

    def _get_review_prompt(self, data_sample: Dict, task_type: str) -> str:
        """获取评审提示词"""
        input_data = data_sample["input"]
        output_data = data_sample["output"]
        
        if task_type == "refinement":
            return f"""
请评审以下需求细化的合理性：

原始需求：{input_data.get('requirement', '')}
细化结果：{output_data}

评判标准：
1. 如果需要细化，分解是否合理、全面
2. 子需求是否具体明确
3. 是否保持了原需求的核心意图

请返回JSON格式：
{{"is_valid": true/false, "reason": "评审理由"}}
"""
        
        elif task_type == "classification":
            return f"""
请评审以下需求分类的准确性：

功能需求：{input_data.get('func_requirement', '')}
分类结果：{output_data}

评判标准：
1. 分类是否准确
2. 是否选择了最适合的类别

请返回JSON格式：
{{"is_valid": true/false, "reason": "评审理由"}}
"""
        
        elif task_type == "rewriting":
            return f"""
请评审以下安全准则重写的质量：

原始带噪声准则：{input_data.get('safety_criterion', '')}
功能需求：{input_data.get('func_requirement', '')}
重写结果：{output_data}

评判标准：
1. 是否成功去除了噪声
2. 重写后的准则是否与功能需求相关
3. 是否保持了核心安全约束

请返回JSON格式：
{{"is_valid": true/false, "reason": "评审理由"}}
"""
        
        return ""

    async def synthesize_refinement_dataset(self, num_samples: int = 1000) -> List[Dict]:
        """合成需求细化数据集"""
        print(f"开始合成需求细化数据集，目标样本数：{num_samples}")
        
        # 加载现有需求作为种子
        existing_requirements = self._load_existing_requirements()
        if not existing_requirements:
            print("警告：无法加载现有需求数据")
            return []
        
        synthesized_data = []
        
        with tqdm(total=num_samples, desc="合成细化数据") as pbar:
            while len(synthesized_data) < num_samples:
                # 随机选择一个需求作为种子
                seed_req = random.choice(existing_requirements)
                requirement = seed_req.get("requirement", "")
                
                if not requirement:
                    continue
                
                # 尝试生成数据
                for attempt in range(self.config.max_synthesis_attempts):
                    data_sample = await self._generate_refinement_data(requirement)
                    if data_sample:
                        # 评审团评审
                        if await self._jury_review(data_sample, "refinement"):
                            synthesized_data.append(data_sample)
                            pbar.update(1)
                            break
                
                if len(synthesized_data) >= num_samples:
                    break
        
        return synthesized_data

    async def synthesize_classification_dataset(self, num_samples: int = 1000) -> List[Dict]:
        """合成需求分类数据集"""
        print(f"开始合成需求分类数据集，目标样本数：{num_samples}")
        
        existing_requirements = self._load_existing_requirements()
        if not existing_requirements:
            print("警告：无法加载现有需求数据")
            return []
        
        synthesized_data = []
        
        with tqdm(total=num_samples, desc="合成分类数据") as pbar:
            while len(synthesized_data) < num_samples:
                seed_req = random.choice(existing_requirements)
                requirement = seed_req.get("requirement", "")
                
                if not requirement:
                    continue
                
                for attempt in range(self.config.max_synthesis_attempts):
                    data_sample = await self._generate_classification_data(requirement)
                    if data_sample:
                        if await self._jury_review(data_sample, "classification"):
                            synthesized_data.append(data_sample)
                            pbar.update(1)
                            break
                
                if len(synthesized_data) >= num_samples:
                    break
        
        return synthesized_data

    async def synthesize_rewriting_dataset(self, num_samples: int = 1000) -> List[Dict]:
        """合成准则重写数据集"""
        print(f"开始合成准则重写数据集，目标样本数：{num_samples}")
        
        safety_criteria = self._load_safety_criteria()
        existing_requirements = self._load_existing_requirements()
        
        if not safety_criteria or not existing_requirements:
            print("警告：无法加载安全准则或需求数据")
            return []
        
        synthesized_data = []
        
        with tqdm(total=num_samples, desc="合成重写数据") as pbar:
            while len(synthesized_data) < num_samples:
                # 随机选择安全准则和需求
                criterion = random.choice(safety_criteria)
                requirement = random.choice(existing_requirements)
                
                safety_criterion_text = criterion.get("safety_criterion", "")
                requirement_text = requirement.get("requirement", "")
                
                if not safety_criterion_text or not requirement_text:
                    continue
                
                for attempt in range(self.config.max_synthesis_attempts):
                    data_sample = await self._generate_rewriting_data(
                        safety_criterion_text, requirement_text
                    )
                    if data_sample:
                        if await self._jury_review(data_sample, "rewriting"):
                            synthesized_data.append(data_sample)
                            pbar.update(1)
                            break
                
                if len(synthesized_data) >= num_samples:
                    break
        
        return synthesized_data

    def save_dataset(self, dataset: List[Dict], filename: str):
        """保存数据集到文件"""
        output_path = f"datasets/training_data/{filename}"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)
        
        print(f"数据集已保存到：{output_path}")

    async def synthesize_all_datasets(self, num_samples_per_task: int = 1000):
        """合成所有任务的数据集"""
        print("开始合成所有训练数据集...")
        
        # 并行合成所有数据集
        tasks = [
            self.synthesize_refinement_dataset(num_samples_per_task),
            self.synthesize_classification_dataset(num_samples_per_task),
            self.synthesize_rewriting_dataset(num_samples_per_task)
        ]
        
        datasets = await asyncio.gather(*tasks)
        
        # 保存数据集
        dataset_names = ["refinement_data.json", "classification_data.json", "rewriting_data.json"]
        for dataset, name in zip(datasets, dataset_names):
            if dataset:
                self.save_dataset(dataset, name)
                print(f"成功合成 {len(dataset)} 个样本用于 {name}")
            else:
                print(f"合成 {name} 失败")


async def main():
    """主函数"""
    # 配置合成参数
    config = SynthesisConfig(
        generator_model="deepseek-r1",
        jury_models=["qwen", "llama", "deepseek-v3"],
        num_jury_members=3,
        agreement_threshold=0.6,
        max_synthesis_attempts=3,
        batch_size=10
    )
    
    # 创建合成器
    synthesizer = DataSynthesizer(config)
    
    # 合成所有数据集
    await synthesizer.synthesize_all_datasets(num_samples_per_task=1000)


if __name__ == "__main__":
    asyncio.run(main()) 