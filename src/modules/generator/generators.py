from modules.generator.base_generator import BaseGenerator
from modules.generator.client.openai_client import OpenAIClient
from config import get_prompt_template_path

class RefinementGenerator(BaseGenerator):
    """需求细化生成器"""
    def __init__(self, **kwargs):
        super().__init__(prompt_template_name="refine", **kwargs)

class ClassificationGenerator(BaseGenerator):
    """需求分类生成器"""
    def __init__(self, **kwargs):
        super().__init__(prompt_template_name="classify", **kwargs)

class QueryRewriteGenerator(BaseGenerator):
    """查询重写生成器"""
    def __init__(self, **kwargs):
        super().__init__(prompt_template_name="query_rewrite", **kwargs)

class FilterGenerator(BaseGenerator):
    """准则过滤生成器"""
    def __init__(self, **kwargs):
        super().__init__(prompt_template_name="filter", **kwargs)

class CriterionRewriteGenerator(BaseGenerator):
    """准则重写生成器"""
    def __init__(self, **kwargs):
        super().__init__(prompt_template_name="criterion_rewrite", **kwargs)

class SafetyRequirementGenerator(BaseGenerator):
    """安全需求生成器"""
    def __init__(self, **kwargs):
        super().__init__(prompt_template_name="safety", **kwargs)

class ConsistencyGenerator(BaseGenerator):
    """一致性检查生成器"""
    def __init__(self, **kwargs):
        super().__init__(prompt_template_name="consistent", **kwargs)

class StraightSafetyRequirementGenerator(BaseGenerator):
    """直接生成安全性需求生成器"""
    def __init__(self, model="qwen"):
        super().__init__(model)
    
    def _get_template_path(self):
        return get_prompt_template_path("straight_generate_requirement")

class StraightWithRetrievalGenerator(BaseGenerator):
    """直接生成安全性需求生成器"""
    def __init__(self, model="qwen"):
        super().__init__(model)
    
    def _get_template_path(self):
        return get_prompt_template_path("straight_with_retrieval")

class HypoGenerator(BaseGenerator):
    """直接生成安全性需求生成器"""
    def __init__(self, model="qwen"):
        super().__init__(model)
    
    def _get_template_path(self):
        return get_prompt_template_path("hypo")


if __name__ == "__main__":
    # 使用示例
    refinement_agent = RefinementGenerator()
    print(refinement_agent.generate([{"requirement":"你好"}]))
