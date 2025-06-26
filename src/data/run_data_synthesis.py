#!/usr/bin/env python3
"""
SR-RAG 数据合成流水线
按照论文描述实现"generation-review"数据合成过程
"""

import asyncio
import argparse
import os
import sys

# 添加项目根目录到路径
sys.path.append(".")

from src.data.synthesis_pipeline import DataSynthesizer, SynthesisConfig
from src.data.convert_to_llamafactory import convert_all_datasets, create_dataset_info


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="SR-RAG数据合成流水线")
    
    parser.add_argument(
        "--num_samples", 
        type=int, 
        default=1000,
        help="每个任务生成的样本数量 (默认: 1000)"
    )
    
    parser.add_argument(
        "--generator_model",
        type=str,
        default="deepseek-r1",
        help="生成器模型名称 (默认: deepseek-r1)"
    )
    
    parser.add_argument(
        "--jury_models",
        nargs="+",
        default=["qwen", "llama", "deepseek-v3"],
        help="评审团模型列表 (默认: qwen llama deepseek-v3)"
    )
    
    parser.add_argument(
        "--agreement_threshold",
        type=float,
        default=0.6,
        help="评审团通过阈值 (默认: 0.6)"
    )
    
    parser.add_argument(
        "--max_attempts",
        type=int,
        default=3,
        help="单个样本最大生成尝试次数 (默认: 3)"
    )
    
    parser.add_argument(
        "--convert_only",
        action="store_true",
        help="仅转换已有数据为LLaMA Factory格式"
    )
    
    parser.add_argument(
        "--task",
        choices=["refinement", "classification", "rewriting", "all"],
        default="all",
        help="指定要合成的任务类型 (默认: all)"
    )
    
    return parser.parse_args()


async def run_synthesis(args):
    """运行数据合成流程"""
    print("=== SR-RAG 数据合成流水线 ===")
    print(f"生成器模型: {args.generator_model}")
    print(f"评审团模型: {args.jury_models}")
    print(f"每个任务样本数: {args.num_samples}")
    print(f"评审通过阈值: {args.agreement_threshold}")
    print("="*50)
    
    # 配置合成参数
    config = SynthesisConfig(
        generator_model=args.generator_model,
        jury_models=args.jury_models,
        num_jury_members=min(3, len(args.jury_models)),
        agreement_threshold=args.agreement_threshold,
        max_synthesis_attempts=args.max_attempts,
        batch_size=10
    )
    
    # 创建合成器
    synthesizer = DataSynthesizer(config)
    
    # 根据指定任务进行合成
    if args.task == "all":
        print("开始合成所有任务的数据集...")
        await synthesizer.synthesize_all_datasets(args.num_samples)
    
    elif args.task == "refinement":
        print("开始合成需求细化数据集...")
        dataset = await synthesizer.synthesize_refinement_dataset(args.num_samples)
        if dataset:
            synthesizer.save_dataset(dataset, "refinement_data.json")
            print(f"成功合成 {len(dataset)} 个需求细化样本")
    
    elif args.task == "classification":
        print("开始合成需求分类数据集...")
        dataset = await synthesizer.synthesize_classification_dataset(args.num_samples)
        if dataset:
            synthesizer.save_dataset(dataset, "classification_data.json")
            print(f"成功合成 {len(dataset)} 个需求分类样本")
    
    elif args.task == "rewriting":
        print("开始合成准则重写数据集...")
        dataset = await synthesizer.synthesize_rewriting_dataset(args.num_samples)
        if dataset:
            synthesizer.save_dataset(dataset, "rewriting_data.json")
            print(f"成功合成 {len(dataset)} 个准则重写样本")


def run_conversion():
    """运行数据格式转换"""
    print("=== 数据格式转换 ===")
    print("将合成数据转换为LLaMA Factory格式...")
    
    try:
        convert_all_datasets()
        create_dataset_info()
        print("✅ 数据格式转换完成")
    except Exception as e:
        print(f"❌ 数据格式转换失败: {e}")


def check_prerequisites():
    """检查运行前提条件"""
    required_files = [
        "datasets/database.json",
        "datasets/testset/gt.json"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ 缺少必要的数据文件:")
        for file_path in missing_files:
            print(f"  - {file_path}")
        print("\n请确保数据文件存在后重新运行。")
        return False
    
    return True


async def main():
    """主函数"""
    args = parse_arguments()
    
    # 检查前提条件
    if not check_prerequisites():
        sys.exit(1)
    
    try:
        if args.convert_only:
            # 仅运行格式转换
            run_conversion()
        else:
            # 运行完整流水线
            await run_synthesis(args)
            
            # 自动运行格式转换
            print("\n" + "="*50)
            run_conversion()
            
        print("\n🎉 数据合成流水线执行完成!")
        print("\n📁 生成的文件:")
        print("  - datasets/training_data/     # 原始合成数据")
        print("  - datasets/llamafactory_data/ # LLaMA Factory格式数据")
        print("\n💡 接下来可以使用fine-tuning配置文件进行模型训练。")
        
    except KeyboardInterrupt:
        print("\n⚠️  用户中断执行")
    except Exception as e:
        print(f"\n❌ 执行过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main()) 