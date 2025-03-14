from openai import OpenAI


class OpenAIClientBase(object):
    def __init__(self, cfg=None):
        # 初始化工作
        pass

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
        super().__init__(cfg)
        from openai import OpenAI

        # 连接到本地vllm服务
        self.client = OpenAI(
            base_url="http://localhost:8001/v1",
            api_key="dummy-token",  # vllm不需要真实的api key
        )

    def chat(self, param):
        try:
            completion = self.client.chat.completions.create(
                model="Qwen2.5-72B-Instruct",  # 默认模型
                messages=param.get("messages", []),
                stream=False,
                max_tokens=param.get("max_tokens", 1024),
                temperature=param.get("temperature", 0.7),
                top_p=param.get("top_p", 1.0),
            )
            return "succeed", completion
        except Exception as e:
            return "failed", str(e)


class DeepSeekR1APIClient(OpenAIClientBase):
    def __init__(self, cfg=None):
        super().__init__(cfg)
        from openai import OpenAI

        self.client = OpenAI(
            base_url="",
            api_key="",
        )

    def chat(self, param):
        try:
            completion = self.client.chat.completions.create(
                model="",
                messages=param.get("messages", []),
                stream=False,
                max_tokens=param.get("max_tokens", 1024),
                temperature=param.get("temperature", 0.7),
                top_p=param.get("top_p", 1.0),
            )
            return "succeed", completion
        except Exception as e:
            return "failed", str(e)


class QwQAPIClient(OpenAIClientBase):
    def __init__(self, cfg=None):
        super().__init__(cfg)
        from openai import OpenAI

        # 连接到本地vllm服务
        self.client = OpenAI(
            base_url="http://localhost:8002/v1",
            api_key="dummy-token",  # vllm不需要真实的api key
        )

    def chat(self, param):
        try:
            completion = self.client.chat.completions.create(
                model="qwq",  # 默认模型
                messages=param.get("messages", []),
                stream=False,
                max_tokens=param.get("max_tokens", 1024),
                temperature=param.get("temperature", 0.7),
                top_p=param.get("top_p", 1.0),
            )
            return "succeed", completion
        except Exception as e:
            return "failed", str(e)


class DeepSeekV3APIClient(OpenAIClientBase):
    def __init__(self, cfg=None):
        super().__init__(cfg)
        from openai import OpenAI

        self.client = OpenAI(
            base_url="",
            api_key="",
        )

    def chat(self, param):
        try:
            completion = self.client.chat.completions.create(
                model="",
                messages=param.get("messages", []),
                stream=False,
                max_tokens=param.get("max_tokens", 1024),
                temperature=param.get("temperature", 0.7),
                top_p=param.get("top_p", 1.0),
            )
            return "succeed", completion
        except Exception as e:
            return "failed", str(e)


class LlaMA3_3APIClient(OpenAIClientBase):
    def __init__(self, cfg=None):
        super().__init__(cfg)
        from openai import OpenAI

        # 连接到本地vllm服务
        self.client = OpenAI(
            base_url="http://localhost:8003/v1",
            api_key="dummy-token",  # vllm不需要真实的api key
        )

    def chat(self, param):
        try:
            completion = self.client.chat.completions.create(
                model="llama",  # 默认模型
                messages=param.get("messages", []),
                stream=False,
                max_tokens=param.get("max_tokens", 1024),
                temperature=param.get("temperature", 0.7),
                top_p=param.get("top_p", 1.0),
            )
            return "succeed", completion
        except Exception as e:
            return "failed", str(e)


class VLLMClient(OpenAIClientBase):
    def __init__(self, cfg=None):
        super().__init__(cfg)
        from openai import OpenAI

        # 连接到本地vllm服务
        self.client = OpenAI(
            base_url="http://localhost:8004/v1",
            api_key="dummy-token",  # vllm不需要真实的api key
        )

    def chat(self, param):
        try:
            completion = self.client.chat.completions.create(
                model="Qwen2.5-72B-Instruct",  # 默认模型
                messages=param.get("messages", []),
                stream=False,
                max_tokens=param.get("max_tokens", 1024),
                temperature=param.get("temperature", 0.7),
                top_p=param.get("top_p", 1.0),
            )
            return "succeed", completion
        except Exception as e:
            return "failed", str(e)


class RerankLoRAClient(OpenAIClientBase):
    def __init__(self, cfg=None):
        super().__init__(cfg)
        from openai import OpenAI

        # 连接到本地vllm服务
        self.client = OpenAI(
            base_url="http://localhost:8001/v1",
            api_key="dummy-token",  # vllm不需要真实的api key
        )

    def chat(self, param):
        try:
            completion = self.client.chat.completions.create(
                model="Qwen2.5-72B-Instruct-LoRA-Rerank",  # 默认模型
                messages=param.get("messages", []),
                stream=False,
                max_tokens=param.get("max_tokens", 1024),
                temperature=param.get("temperature", 0.7),
                top_p=param.get("top_p", 1.0),
            )
            return "succeed", completion
        except Exception as e:
            return "failed", str(e)

class ClassifyLoRAClient(OpenAIClientBase):
    def __init__(self, cfg=None):
        super().__init__(cfg)
        from openai import OpenAI

        # 连接到本地vllm服务
        self.client = OpenAI(
            base_url="http://localhost:8001/v1",
            api_key="dummy-token",  # vllm不需要真实的api key
        )

    def chat(self, param):
        try:
            completion = self.client.chat.completions.create(
                model="Qwen2.5-72B-Instruct-LoRA-Classify",  # 默认模型
                messages=param.get("messages", []),
                stream=False,
                max_tokens=param.get("max_tokens", 1024),
                temperature=param.get("temperature", 0.7),
                top_p=param.get("top_p", 1.0),
            )
            return "succeed", completion
        except Exception as e:
            return "failed", str(e)

class FilterLoRAClient(OpenAIClientBase):
    def __init__(self, cfg=None):
        super().__init__(cfg)
        from openai import OpenAI

        # 连接到本地vllm服务
        self.client = OpenAI(
            base_url="http://localhost:8001/v1",
            api_key="dummy-token",  # vllm不需要真实的api key
        )

    def chat(self, param):
        try:
            completion = self.client.chat.completions.create(
                model="Qwen2.5-72B-Instruct-LoRA-Filter",  # 默认模型
                messages=param.get("messages", []),
                stream=False,
                max_tokens=param.get("max_tokens", 1024),
                temperature=param.get("temperature", 0.7),
                top_p=param.get("top_p", 1.0),
            )
            return "succeed", completion
        except Exception as e:
            return "failed", str(e)
