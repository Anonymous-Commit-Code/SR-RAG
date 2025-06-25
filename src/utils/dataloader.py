import json
import os
from typing import List, Dict, Any, Optional


class DataLoader:
    """数据加载器类，用于加载和处理各种格式的数据"""
    
    def __init__(self, data_path: str):
        """
        初始化数据加载器
        
        :param data_path: 数据文件路径
        """
        self.data_path = data_path
        self.data = None
    
    def load_json(self) -> List[Dict[str, Any]]:
        """
        加载JSON格式的数据
        
        :return: JSON数据列表
        """
        try:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
            return self.data
        except FileNotFoundError:
            raise FileNotFoundError(f"数据文件未找到: {self.data_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON格式错误: {e}")
    
    def load_txt(self) -> List[str]:
        """
        加载文本文件数据
        
        :return: 文本行列表
        """
        try:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                self.data = f.readlines()
            return [line.strip() for line in self.data]
        except FileNotFoundError:
            raise FileNotFoundError(f"数据文件未找到: {self.data_path}")
    
    def save_json(self, data: List[Dict[str, Any]], output_path: Optional[str] = None):
        """
        保存数据为JSON格式
        
        :param data: 要保存的数据
        :param output_path: 输出文件路径，如果为None则使用原路径
        """
        save_path = output_path or self.data_path
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
    
    def get_data(self) -> Optional[List]:
        """
        获取已加载的数据
        
        :return: 数据列表或None
        """
        return self.data
    
    def __len__(self) -> int:
        """
        返回数据长度
        
        :return: 数据长度
        """
        return len(self.data) if self.data else 0


def load_dataset(file_path: str, file_type: str = 'json') -> List[Dict[str, Any]]:
    """
    便捷函数：加载数据集
    
    :param file_path: 文件路径
    :param file_type: 文件类型 ('json' 或 'txt')
    :return: 加载的数据
    """
    loader = DataLoader(file_path)
    if file_type.lower() == 'json':
        return loader.load_json()
    elif file_type.lower() == 'txt':
        return loader.load_txt()
    else:
        raise ValueError(f"不支持的文件类型: {file_type}")


def batch_process(data: List[Any], batch_size: int = 32):
    """
    将数据分批处理
    
    :param data: 要分批的数据
    :param batch_size: 批次大小
    :return: 批次生成器
    """
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]


if __name__ == "__main__":
    # 使用示例
    loader = DataLoader("test.json")
    try:
        data = loader.load_json()
        print(f"成功加载 {len(data)} 条数据")
    except Exception as e:
        print(f"加载失败: {e}")
