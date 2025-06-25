from openai import OpenAI
from config import MODEL_CONFIG


class OpenAIClientBase(object):
    def __init__(self, model_name, cfg=None):
        if cfg is None:
            cfg = MODEL_CONFIG
        self.model_name = model_name
        self.base_url = cfg["endpoints"].get(model_name)
        self.api_key = cfg["api_keys"].get(model_name, "dummy-token")
        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)

    def chat(self, param):
        """统一的参数列表:
        param = {
            'model': str,            # 模型名称
            'messages': list,        # 消息列表
            'max_tokens': int,       # 最大token数
            'temperature': float,    # 温度
            'top_p': float,         # top_p采样
        }
        """
        raise NotImplementedError()


class QwenAPIClient(OpenAIClientBase):
    def __init__(self, cfg=None):
        super().__init__("qwen", cfg)

    def chat(self, param):
        try:
            completion = self.client.chat.completions.create(
                model="Qwen2.5-72B-Instruct",  # 默认模型
                messages=param.get("messages", []),
                stream=False,
                max_tokens=param.get("max_tokens", MODEL_CONFIG["max_tokens"]),
                temperature=param.get("temperature", MODEL_CONFIG["temperature"]),
                top_p=param.get("top_p", MODEL_CONFIG["top_p"]),
            )
            return "succeed", completion
        except Exception as e:
            return "failed", str(e)


class DeepSeekR1APIClient(OpenAIClientBase):
    def __init__(self, cfg=None):
        super().__init__("deepseek-r1", cfg)

    def chat(self, param):
        try:
            completion = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=param.get("messages", []),
                stream=False,
                max_tokens=param.get("max_tokens", MODEL_CONFIG["max_tokens"]),
                temperature=param.get("temperature", MODEL_CONFIG["temperature"]),
                top_p=param.get("top_p", MODEL_CONFIG["top_p"]),
            )
            return "succeed", completion
        except Exception as e:
            return "failed", str(e)


class QwQAPIClient(OpenAIClientBase):
    def __init__(self, cfg=None):
        super().__init__("qwq", cfg)

    def chat(self, param):
        try:
            completion = self.client.chat.completions.create(
                model="qwq",  # 默认模型
                messages=param.get("messages", []),
                stream=False,
                max_tokens=param.get("max_tokens", MODEL_CONFIG["max_tokens"]),
                temperature=param.get("temperature", MODEL_CONFIG["temperature"]),
                top_p=param.get("top_p", MODEL_CONFIG["top_p"]),
            )
            return "succeed", completion
        except Exception as e:
            return "failed", str(e)


class DeepSeekV3APIClient(OpenAIClientBase):
    def __init__(self, cfg=None):
        super().__init__("deepseek-v3", cfg)

    def chat(self, param):
        try:
            completion = self.client.chat.completions.create(
                model="deepseek-coder",
                messages=param.get("messages", []),
                stream=False,
                max_tokens=param.get("max_tokens", MODEL_CONFIG["max_tokens"]),
                temperature=param.get("temperature", MODEL_CONFIG["temperature"]),
                top_p=param.get("top_p", MODEL_CONFIG["top_p"]),
            )
            return "succeed", completion
        except Exception as e:
            return "failed", str(e)


class LlaMA3_3APIClient(OpenAIClientBase):
    def __init__(self, cfg=None):
        super().__init__("llama", cfg)

    def chat(self, param):
        try:
            completion = self.client.chat.completions.create(
                model="llama",  # 默认模型
                messages=param.get("messages", []),
                stream=False,
                max_tokens=param.get("max_tokens", MODEL_CONFIG["max_tokens"]),
                temperature=param.get("temperature", MODEL_CONFIG["temperature"]),
                top_p=param.get("top_p", MODEL_CONFIG["top_p"]),
            )
            return "succeed", completion
        except Exception as e:
            return "failed", str(e)


class VLLMClient(OpenAIClientBase):
    def __init__(self, cfg=None):
        super().__init__("vllm", cfg)

    def chat(self, param):
        try:
            completion = self.client.chat.completions.create(
                model="Qwen2.5-72B-Instruct",  # 默认模型
                messages=param.get("messages", []),
                stream=False,
                max_tokens=param.get("max_tokens", MODEL_CONFIG["max_tokens"]),
                temperature=param.get("temperature", MODEL_CONFIG["temperature"]),
                top_p=param.get("top_p", MODEL_CONFIG["top_p"]),
            )
            return "succeed", completion
        except Exception as e:
            return "failed", str(e)


class RerankLoRAClient(OpenAIClientBase):
    def __init__(self, cfg=None):
        super().__init__("rerank", cfg)

    def chat(self, param):
        try:
            completion = self.client.chat.completions.create(
                model="Qwen2.5-72B-Instruct-LoRA-Rerank",  # 默认模型
                messages=param.get("messages", []),
                stream=False,
                max_tokens=param.get("max_tokens", MODEL_CONFIG["max_tokens"]),
                temperature=param.get("temperature", MODEL_CONFIG["temperature"]),
                top_p=param.get("top_p", MODEL_CONFIG["top_p"]),
            )
            return "succeed", completion
        except Exception as e:
            return "failed", str(e)

class ClassifyLoRAClient(OpenAIClientBase):
    def __init__(self, cfg=None):
        super().__init__("classify", cfg)

    def chat(self, param):
        try:
            completion = self.client.chat.completions.create(
                model="Qwen2.5-72B-Instruct-LoRA-Classify",  # 默认模型
                messages=param.get("messages", []),
                stream=False,
                max_tokens=param.get("max_tokens", MODEL_CONFIG["max_tokens"]),
                temperature=param.get("temperature", MODEL_CONFIG["temperature"]),
                top_p=param.get("top_p", MODEL_CONFIG["top_p"]),
            )
            return "succeed", completion
        except Exception as e:
            return "failed", str(e)

class FilterLoRAClient(OpenAIClientBase):
    def __init__(self, cfg=None):
        super().__init__("filter", cfg)

    def chat(self, param):
        try:
            completion = self.client.chat.completions.create(
                model="Qwen2.5-72B-Instruct-LoRA-Filter",  # 默认模型
                messages=param.get("messages", []),
                stream=False,
                max_tokens=param.get("max_tokens", MODEL_CONFIG["max_tokens"]),
                temperature=param.get("temperature", MODEL_CONFIG["temperature"]),
                top_p=param.get("top_p", MODEL_CONFIG["top_p"]),
            )
            return "succeed", completion
        except Exception as e:
            return "failed", str(e)
