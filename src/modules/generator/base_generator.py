from abc import ABC, abstractmethod
from modules.generator.client.openai_client import OpenAIClient
from modules.generator.prompt.prompt_factory import create_prompt
import json
import re

class BaseGenerator(ABC):
    def __init__(self, model="qwen"):
        self.client = OpenAIClient(model)
        self.template_path = self._get_template_path()
    
    @abstractmethod
    def _get_template_path(self):
        """返回对应的prompt模板路径"""
        pass

    def _extract_content(self, response):
        """提取 API 响应内容"""
        if isinstance(response, dict) and "choices" in response:
            return response["choices"][0]["message"]["content"].strip()
        elif isinstance(response, dict) and "content" in response:
            return response["content"][0]["text"].strip()
        return response.choices[0].message.content.strip()

    def _clean_response(self, response):
        """清理 API 响应内容"""
        return re.sub(r'^```json|```$', '', response.strip(), flags=re.MULTILINE).strip()

    def _remove_think_tags(self, text):
        """清理 think 内容"""

        # 匹配 "<think>" 开始，到 "</think>\n\n" 结束的内容
        pattern = r'<think>.*?</think>\n\n'
        # re.DOTALL 使得 . 能匹配包括换行符在内的任意字符
        cleaned_text = re.sub(pattern, '', text, flags=re.DOTALL)
        
        # 处理可能没有额外换行的情况 (如 <think>内容</think>)
        pattern2 = r'<think>.*?</think>'
        cleaned_text = re.sub(pattern2, '', cleaned_text, flags=re.DOTALL)
        
        return cleaned_text

    def generate(self, content):
        """生成响应"""
        prompt = create_prompt(content, self.template_path)
        response = self._extract_content(self.client.get_response(prompt))
        no_think_response = self._remove_think_tags(response)
        cleaned_response = self._clean_response(no_think_response)
        return json.loads(cleaned_response) 