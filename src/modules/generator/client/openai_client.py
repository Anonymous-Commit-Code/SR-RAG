import sys

sys.path.append("src")
import time
import random
from tenacity import retry, stop_after_attempt, wait_exponential
from modules.generator.client.openai_client_base import (
    QwenAPIClient,
    VLLMClient,
    DeepSeekR1APIClient,
    DeepSeekV3APIClient,
    LlaMA3_3APIClient,
    QwQAPIClient,
    RerankLoRAClient,
    ClassifyLoRAClient,
    FilterLoRAClient

)


class OpenAIClient:
    def __init__(self, client_type="red"):
        self.client = self._initialize_client(client_type)
        self.type = client_type

    def _initialize_client(self, client_type):
        """
        根据指定的客户端类型初始化相应的 OpenAI 客户端。

        :param client_type: 客户端类型，可以是 "red"、"vllm" 或其他类型。
        :return: 初始化后的 OpenAI 客户端实例。
        """
        if client_type == "qwen":
            return QwenAPIClient()
        elif client_type == "vllm":
            return VLLMClient()
        elif client_type.startswith("deepseek-r1"):
            return DeepSeekR1APIClient()
        elif client_type.startswith("deepseek-v3"):
            return DeepSeekV3APIClient()
        elif client_type.startswith("llama"):
            return LlaMA3_3APIClient()
        elif client_type.startswith("qwq"):
            return QwQAPIClient()
        elif client_type.startswith("rerank"):
            return RerankLoRAClient
        elif client_type.startwiths("classify"):
            return ClassifyLoRAClient
        elif client_type.startwiths("filter"):
            return FilterLoRAClient
        

    @retry(
        stop=stop_after_attempt(5), wait=wait_exponential(multiplier=2, min=4, max=10)
    )
    def get_response(self, prompt):
        """
        向 OpenAI API 发送请求并获取响应。

        :param prompt: 要发送给 OpenAI 的提示信息。
        :return: OpenAI API 的响应内容。
        :raises Exception: 如果 API 请求失败则抛出异常。
        """
        try:
            time.sleep(random.uniform(0.5, 1.5))
            param = {
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": 16384,
                "temperature": 0.3,
                "top_p": 1.0,
            }

            status, response = self.client.chat(param)

            if status == "failed":
                raise Exception(response)

            return response
        except Exception as e:
            print(f"Error in get_response: {str(e)}")
            raise e
