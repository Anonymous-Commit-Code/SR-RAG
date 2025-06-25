#!/usr/bin/env python3
"""
验证脚本：检查所有修复是否成功应用
"""

import os
import sys
import importlib.util
from pathlib import Path


def check_file_exists(file_path: str, description: str) -> bool:
    """检查文件是否存在"""
    if os.path.exists(file_path):
        print(f"✅ {description}: {file_path}")
        return True
    else:
        print(f"❌ {description}: {file_path} - 文件不存在")
        return False


def check_file_content(file_path: str, search_text: str, description: str) -> bool:
    """检查文件是否包含特定内容"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            if search_text in content:
                print(f"✅ {description}: 在 {file_path} 中找到")
                return True
            else:
                print(f"❌ {description}: 在 {file_path} 中未找到")
                return False
    except Exception as e:
        print(f"❌ {description}: 读取 {file_path} 时出错 - {e}")
        return False


def check_import(module_path: str, description: str) -> bool:
    """检查模块是否可以正常导入"""
    try:
        # 添加项目根目录到路径
        sys.path.insert(0, '.')
        
        # 将文件路径转换为模块名
        module_name = module_path.replace('/', '.').replace('.py', '')
        if module_name.startswith('src.'):
            module_name = module_name[4:]  # 移除 'src.' 前缀
        
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            print(f"✅ {description}: {module_path} - 导入成功")
            return True
        else:
            print(f"❌ {description}: {module_path} - 无法加载模块")
            return False
    except Exception as e:
        print(f"❌ {description}: {module_path} - 导入失败: {e}")
        return False


def main():
    """主验证函数"""
    print("🔍 开始验证所有修复...")
    print("=" * 60)
    
    passed_checks = 0
    total_checks = 0
    
    # 1. 检查文件名修复
    total_checks += 1
    if check_file_exists("src/modules/retriever/retriever.py", "正确命名的检索器文件"):
        passed_checks += 1
    
    total_checks += 1
    if not os.path.exists("src/modules/retriever/retriver.py"):
        print("✅ 拼写错误的文件已删除: retriver.py")
        passed_checks += 1
    else:
        print("❌ 拼写错误的文件仍然存在: retriver.py")
    
    # 2. 检查 startswith 修复
    total_checks += 1
    if check_file_content(
        "src/modules/generator/client/openai_client.py",
        "startswith",
        "方法名拼写修复"
    ):
        passed_checks += 1
    
    total_checks += 1
    if not check_file_content(
        "src/modules/generator/client/openai_client.py",
        "startwiths",
        "检查是否还有拼写错误"
    ):
        print("✅ 拼写错误已修复: startwiths -> startswith")
        passed_checks += 1
    
    # 3. 检查默认返回值
    total_checks += 1
    if check_file_content(
        "src/modules/generator/client/openai_client.py",
        "else:\n            # 默认返回 QwenAPIClient\n            return QwenAPIClient()",
        "默认返回值添加"
    ):
        passed_checks += 1
    
    # 4. 检查配置文件
    total_checks += 1
    if check_file_exists("config.py", "配置文件"):
        passed_checks += 1
    
    # 5. 检查dataloader实现
    total_checks += 1
    if check_file_content(
        "src/utils/dataloader.py",
        "class DataLoader:",
        "DataLoader类实现"
    ):
        passed_checks += 1
    
    # 6. 检查生成器初始化修复
    total_checks += 1
    if check_file_content(
        "src/modules/generator/generators.py",
        "super().__init__(model)",
        "生成器初始化修复"
    ):
        passed_checks += 1
    
    # 7. 检查配置文件使用
    total_checks += 1
    if check_file_content(
        "src/modules/generator/generators.py",
        "from config import get_prompt_template_path",
        "配置文件导入"
    ):
        passed_checks += 1
    
    # 8. 检查异常处理改善
    total_checks += 1
    if check_file_content(
        "src/modules/retriever/hnsw.py",
        "except Exception as e:",
        "异常处理改善"
    ):
        passed_checks += 1
    
    # 9. 检查字段名统一
    total_checks += 1
    if check_file_content(
        "src/modules/retriever/bm25.py",
        'doc.get("safety_criterion", doc.get("分析准则", ""))',
        "字段名统一处理"
    ):
        passed_checks += 1
    
    # 10. 检查requirements.txt更新
    total_checks += 1
    if check_file_content(
        "requirements.txt",
        ">=",
        "版本号添加"
    ):
        passed_checks += 1
    
    # 11. 检查README更新
    total_checks += 1
    if check_file_content(
        "README.md",
        "## 🔧 最新修复",
        "README更新"
    ):
        passed_checks += 1
    
    # 12. 尝试导入关键模块（如果可能）
    key_modules = [
        ("config.py", "配置模块"),
        ("src/utils/dataloader.py", "数据加载模块"),
        ("src/modules/retriever/retriever.py", "检索器模块"),
    ]
    
    for module_path, description in key_modules:
        total_checks += 1
        if check_import(module_path, description):
            passed_checks += 1
    
    # 输出结果
    print("=" * 60)
    print(f"🎯 验证完成: {passed_checks}/{total_checks} 项检查通过")
    
    if passed_checks == total_checks:
        print("🎉 所有修复验证成功！")
        return True
    else:
        print(f"⚠️  还有 {total_checks - passed_checks} 项需要注意")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 