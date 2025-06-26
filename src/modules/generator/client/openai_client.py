import time
import random
import asyncio
import json
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
    FilterLoRAClient,
)
from config import MODEL_CONFIG


class OpenAIClient:
    def __init__(self, model="qwen"):
        # 支持传入model参数作为client_type的别名
        client_type = model if isinstance(model, str) else "qwen"
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
            return RerankLoRAClient()
        elif client_type.startswith("classify"):
            return ClassifyLoRAClient()
        elif client_type.startswith("filter"):
            return FilterLoRAClient()
        else:
            # 默认返回 QwenAPIClient
            return QwenAPIClient()

    @retry(
        stop=stop_after_attempt(MODEL_CONFIG["max_retries"]),
        wait=wait_exponential(
            multiplier=MODEL_CONFIG["retry_delay"]["multiplier"],
            min=MODEL_CONFIG["retry_delay"]["min"],
            max=MODEL_CONFIG["retry_delay"]["max"],
        ),
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
                "max_tokens": MODEL_CONFIG["max_tokens"],
                "temperature": MODEL_CONFIG["temperature"],
                "top_p": MODEL_CONFIG["top_p"],
            }

            status, response = self.client.chat(param)

            if status == "failed":
                raise Exception(response)

            return response
        except Exception as e:
            print(f"Error in get_response: {str(e)}")
            raise e

    async def generate_async(self, prompt):
        """
        异步生成响应，用于数据合成流水线
        
        :param prompt: 输入提示
        :return: 生成的响应（尝试解析为JSON，失败则返回原始字符串）
        """
        try:
            # 使用asyncio在线程池中运行同步方法
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, self.get_response, prompt)
            
            # 尝试解析JSON响应
            try:
                if isinstance(response, str):
                    # 提取JSON部分（如果响应包含```json...```格式）
                    if "```json" in response:
                        json_start = response.find("```json") + 7
                        json_end = response.find("```", json_start)
                        if json_end != -1:
                            json_str = response[json_start:json_end].strip()
                            return json.loads(json_str)
                    # 尝试直接解析整个响应
                    return json.loads(response)
                return response
            except json.JSONDecodeError:
                # 如果无法解析为JSON，返回原始响应
                return response
                
        except Exception as e:
            print(f"Error in generate_async: {e}")
            return None
