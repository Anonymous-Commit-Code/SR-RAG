import sys
sys.path.append("src")
from modules.generator.base_generator import BaseGenerator
from modules.generator.client.openai_client import OpenAIClient

class RefinementGenerator(BaseGenerator):
    """细化生成器"""
    def __init__(self, model="refine"):
        super().__init__()
        self.client=OpenAIClient(model)

    def _get_template_path(self):
        return "src/modules/generator/prompt/prompt_template/refine.txt"

class ClassificationGenerator(BaseGenerator):
    """分类生成器"""
    def __init__(self, model="classify"):
        super().__init__()
        self.client=OpenAIClient(model)

    def _get_template_path(self):
        return "src/modules/generator/prompt/prompt_template/classify.txt"

class QueryRewriteGenerator(BaseGenerator):
    """查询重写生成器"""
    def _get_template_path(self):
        return "src/modules/generator/prompt/prompt_template/query_rewrite.txt"

class FilterGenerator(BaseGenerator):
    """过滤生成器"""
    def __init__(self, model="filter"):
        super().__init__()
        self.client=OpenAIClient(model)

    def _get_template_path(self):
        return "src/modules/generator/prompt/prompt_template/filter.txt"

class CriterionRewriteGenerator(BaseGenerator):
    """需求改写生成器"""
    def _get_template_path(self):
        return "src/modules/generator/prompt/prompt_template/criterion_rewrite.txt"

class SafetyRequirementGenerator(BaseGenerator):
    """安全性需求生成器"""
    def _get_template_path(self):
        return "src/modules/generator/prompt/prompt_template/safety.txt"

class ConsistencyGenerator(BaseGenerator):
    """一致性判断生成器"""
    def _get_template_path(self):
        return "src/modules/generator/prompt/prompt_template/consistent.txt" 

class StraightSafetyRequirementGenerator(BaseGenerator):
    """直接生成安全性需求生成器"""
    def _get_template_path(self):
        return "src/modules/generator/prompt/prompt_template/straight_generate_requirement.txt"

class StraightWithRetrievalGenerator(BaseGenerator):
    """直接生成安全性需求生成器"""
    def _get_template_path(self):
        return "src/modules/generator/prompt/prompt_template/straight_with_retrieval.txt"

class HypoGenerator(BaseGenerator):
    """直接生成安全性需求生成器"""
    def _get_template_path(self):
        return "src/modules/generator/prompt/prompt_template/hypo.txt"


if __name__ == "__main__":
    # 使用示例
    refinement_agent = RefinementGenerator()
    print(refinement_agent.generate(["你好","hello"]))
