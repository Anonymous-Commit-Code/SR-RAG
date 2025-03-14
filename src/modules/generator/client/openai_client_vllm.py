import base64
import requests
import os
from PIL import Image
from typing import Optional

# === 配置部分 ===
API_URL = "https://runway.devops.xiaohongshu.com/openai/chat/completions?api-version=2024-02-15-preview"
API_KEY = "ea2e4d0fcb494188b354153b900bc2a4"

PROMPT_TEMPLATE = """
根据图片给出一条12字以内，生动惊艳有文采的slogan。只输出slogan，不需要输出其他任何内容
"""

def encode_image(image_path: str) -> Optional[str]:
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        print(f"错误: 无法编码图片 {image_path}: {e}")
        return None

def resize_image(image: Image.Image, target_size: int = 512) -> Image.Image:
    width, height = image.size
    aspect_ratio = width / height

    if 0.9 <= aspect_ratio <= 1.1:
        return image.resize((target_size, target_size), Image.LANCZOS)
    elif width < height:
        new_width = target_size
        new_height = int(new_width / aspect_ratio)
    else:
        new_height = target_size
        new_width = int(new_height * aspect_ratio)

    return image.resize((new_width, new_height), Image.LANCZOS)

def process_image_request(prompt: str, base64_image: str) -> str:
    headers = {
        "api-key": API_KEY,
        "Content-Type": "application/json"
    }
    data = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"<image>\n{prompt}"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 16,
        "temperature": 1
    }

    try:
        response = requests.post(API_URL, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except Exception as e:
        print(f"请求错误: {e}")
        return ""

def process_single_image(image_path: str) -> str:
    # 调整图片大小
    with Image.open(image_path) as img:
        resized_img = resize_image(img)
        temp_path = "temp_resized.jpg"
        resized_img.save(temp_path)
    
    # 编码图片并发送请求
    base64_image = encode_image(temp_path)
    os.remove(temp_path)
    
    if base64_image:
        return process_image_request(PROMPT_TEMPLATE, base64_image)
    return ""

if __name__ == "__main__":
    # 示例使用
    image_path = "/Users/shijiachen/Downloads/test.png"  # 替换为实际图片路径
    result = process_single_image(image_path)
    print(f"生成的slogan: {result}") 