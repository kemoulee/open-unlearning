#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TOFU数据集第18条数据的top-P采样生成脚本
针对已在TOFU数据集上训练的Llama-3.2-1B模型，从forget10%数据集中采样第18条数据，使用top-P sampling生成5条回复
"""

import torch
import os
import sys
import json
from datasets import load_dataset
from omegaconf import DictConfig, OmegaConf
import torch.nn.functional as F

# 添加项目路径到sys.path，以便导入src模块
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from model import get_model

# 设置缓存目录，与项目标准做法一致
HF_HOME = os.getenv('HF_HOME', os.path.expanduser('~/.cache/huggingface'))

def create_model_config():
    """创建模型配置，使用已在TOFU数据集上训练的模型"""
    model_config = {
        "model_args": {
            "pretrained_model_name_or_path": "open-unlearning/tofu_Llama-3.2-1B-Instruct_full",
            "attn_implementation": "flash_attention_2",
            "torch_dtype": "bfloat16",
            "device_map": "auto"  # 自动分配设备
        },
        "tokenizer_args": {
            "pretrained_model_name_or_path": "open-unlearning/tofu_Llama-3.2-1B-Instruct_full"
        },
        "template_args": {
            "apply_chat_template": True,
            "system_prompt": "You are a helpful assistant.",
            "system_prompt_with_special_tokens": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant.<|eot_id|>",
            "user_start_tag": "<|start_header_id|>user<|end_header_id|>\n\n",
            "user_end_tag": "<|eot_id|>",
            "asst_start_tag": "<|start_header_id|>assistant<|end_header_id|>\n\n",
            "asst_end_tag": "<|eot_id|>",
            "date_string": "10 Apr 2025"
        }
    }
    return OmegaConf.create(model_config)

def load_tofu_dataset():
    """加载TOFU forget10%数据集"""
    print("📥 正在加载TOFU数据集...")
    dataset = load_dataset("locuslab/TOFU", name="forget10", split="train")
    print(f"✅ 数据集加载完成，共{len(dataset)}条数据")
    return dataset

def load_trained_model():
    """加载已在TOFU数据集上训练的Llama-3.2-1B模型"""
    print("🤖 正在加载TOFU训练的Llama-3.2-1B模型...")
    print(f"💾 缓存目录: {HF_HOME}")
    
    model_cfg = create_model_config()
    
    # 检查缓存状态
    model_name = "open-unlearning/tofu_Llama-3.2-1B-Instruct_full"
    cache_path = os.path.join(HF_HOME, "hub", f"models--{model_name.replace('/', '--')}")
    
    if os.path.exists(cache_path):
        print(f"✅ 发现本地缓存: {cache_path}")
    else:
        print(f"📥 本地无缓存，将从HuggingFace Hub下载（约16GB）")
        print(f"🕐 请耐心等待下载完成...")
    
    model, tokenizer = get_model(model_cfg)
    print("✅ 已训练模型加载完成")
    return model, tokenizer

def format_prompt_with_system(question):
    """使用系统prompt格式化问题"""
    system_prompt = "Cutting Knowledge Date: December 2023\nToday Date: 10 Apr 2025\n\nYou are a helpful assistant."
    
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
    return prompt

def generate_responses(model, tokenizer, prompt, num_samples=5):
    """使用top-P采样生成回复"""
    print("🎯 开始生成回复...")
    
    # 编码输入
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    
    # 将输入移动到模型设备
    if hasattr(model, 'device'):
        device = model.device
    else:
        device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    
    print(f"🔧 输入长度: {input_ids.shape[1]} tokens")
    print(f"💻 使用设备: {device}")
    
    # 生成参数：使用top-P采样
    generation_kwargs = {
        "input_ids": input_ids,
        "do_sample": True,       # 启用采样
        "top_p": 0.9,           # top-P参数，保留累计概率前90%的token
        "temperature": 0.7,      # 温度参数，控制随机性
        "max_new_tokens": 200,   # 最大生成token数
        "num_return_sequences": num_samples,  # 生成5条回复
        "pad_token_id": tokenizer.eos_token_id,
        "use_cache": True,
        "early_stopping": True
    }
    
    # 生成回复
    print("⚡ 正在生成...")
    with torch.no_grad():
        outputs = model.generate(**generation_kwargs)
    
    # 解码生成的文本
    responses = []
    for i in range(num_samples):
        # 只保留新生成的部分（去掉原始prompt）
        generated_tokens = outputs[i][input_ids.shape[-1]:]
        response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        responses.append(response.strip())
    
    print("✅ 回复生成完成")
    return responses

def generate_responses_by_probability(model, tokenizer, prompt):
    """根据token概率采样不同类型的回复"""
    print("🎯 开始生成不同概率级别的回复...")
    
    # 编码输入
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    
    # 将输入移动到模型设备
    if hasattr(model, 'device'):
        device = model.device
    else:
        device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    
    print(f"🔧 输入长度: {input_ids.shape[1]} tokens")
    print(f"💻 使用设备: {device}")
    
    # 定义三种不同的采样策略
    sampling_configs = {
        "高概率": {
            "do_sample": True,
            "top_p": 0.3,           # 只考虑累计概率前30%的token（更保守）
            "temperature": 0.3,      # 低温度，倾向于高概率token
            "max_new_tokens": 200,
            "num_return_sequences": 5,
            "pad_token_id": tokenizer.eos_token_id,
            "use_cache": True,
            "early_stopping": True
        },
        "中概率": {
            "do_sample": True,
            "top_p": 0.7,           # 考虑累计概率前70%的token
            "temperature": 0.7,      # 中等温度
            "max_new_tokens": 200,
            "num_return_sequences": 5,
            "pad_token_id": tokenizer.eos_token_id,
            "use_cache": True,
            "early_stopping": True
        },
        "低概率": {
            "do_sample": True,
            "top_p": 0.95,          # 考虑更多token（包括低概率的）
            "temperature": 1.2,      # 高温度，增加随机性，倾向于低概率token
            "max_new_tokens": 200,
            "num_return_sequences": 5,
            "pad_token_id": tokenizer.eos_token_id,
            "use_cache": True,
            "early_stopping": True
        }
    }
    
    all_responses = {}
    
    # 为每种概率级别生成回复
    for prob_type, config in sampling_configs.items():
        print(f"⚡ 正在生成{prob_type}回复 (top_p={config['top_p']}, temperature={config['temperature']})...")
        
        generation_kwargs = {
            "input_ids": input_ids,
            **config
        }
        
        with torch.no_grad():
            outputs = model.generate(**generation_kwargs)
        
        # 解码生成的文本
        responses = []
        for i in range(5):
            # 只保留新生成的部分（去掉原始prompt）
            generated_tokens = outputs[i][input_ids.shape[-1]:]
            response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            responses.append(response.strip())
        
        all_responses[prob_type] = responses
    
    print("✅ 所有概率级别的回复生成完成")
    return all_responses

def generate_responses_by_direct_probability(model, tokenizer, prompt):
    """直接根据token概率采样不同类型的回复"""
    print("🎯 开始直接概率采样生成回复...")
    
    # 编码输入
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    
    # 将输入移动到模型设备
    if hasattr(model, 'device'):
        device = model.device
    else:
        device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    
    print(f"🔧 输入长度: {input_ids.shape[1]} tokens")
    print(f"💻 使用设备: {device}")
    
    def sample_from_probability_range(logits, prob_range, num_samples=5):
        """从指定概率范围采样token"""
        # 获取概率分布
        probs = F.softmax(logits, dim=-1)
        
        # 根据概率排序
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        
        # 确定概率范围的索引区间
        vocab_size = probs.shape[-1]
        if prob_range == "高概率":
            # 前10%的token
            start_idx = 0
            end_idx = max(1, int(vocab_size * 0.1))
        elif prob_range == "中概率":
            # 10%-50%的token
            start_idx = int(vocab_size * 0.1)
            end_idx = int(vocab_size * 0.5)
        else:  # 低概率
            # 50%-90%的token（避免极低概率的噪声token）
            start_idx = int(vocab_size * 0.5)
            end_idx = int(vocab_size * 0.9)
        
        # 从指定范围随机采样
        range_indices = sorted_indices[start_idx:end_idx]
        range_probs = sorted_probs[start_idx:end_idx]
        
        # 重新归一化概率
        range_probs = range_probs / range_probs.sum()
        
        # 进行采样
        sampled_tokens = []
        for _ in range(num_samples):
            # 根据重新归一化的概率分布采样
            sample_idx = torch.multinomial(range_probs, 1)
            sampled_token = range_indices[sample_idx]
            sampled_tokens.append(sampled_token.item())
        
        return sampled_tokens
    
    def generate_sequence(prob_range, max_length=200):
        """生成指定概率类型的序列"""
        sequences = []
        
        for seq_idx in range(5):  # 生成5条回复
            current_ids = input_ids.clone()
            
            for step in range(max_length):
                # 前向传播获取logits
                with torch.no_grad():
                    outputs = model(current_ids)
                    logits = outputs.logits[0, -1, :]  # 获取最后一个位置的logits
                
                # 从指定概率范围采样下一个token
                next_tokens = sample_from_probability_range(logits, prob_range, num_samples=1)
                next_token = next_tokens[0]
                
                # 添加到序列
                current_ids = torch.cat([current_ids, torch.tensor([[next_token]], device=device)], dim=1)
                
                # 检查是否生成了结束token
                if next_token == tokenizer.eos_token_id:
                    break
            
            # 解码序列（去掉原始prompt）
            generated_tokens = current_ids[0, input_ids.shape[1]:]
            response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            sequences.append(response.strip())
        
        return sequences
    
    all_responses = {}
    prob_ranges = ["高概率", "中概率", "低概率"]
    
    # 为每种概率级别生成回复
    for prob_range in prob_ranges:
        print(f"⚡ 正在生成{prob_range}回复...")
        if prob_range == "高概率":
            print("   📊 采样策略: 从前10%高概率token中选择")
        elif prob_range == "中概率":
            print("   📊 采样策略: 从10%-50%中等概率token中选择")
        else:
            print("   📊 采样策略: 从50%-90%低概率token中选择")
        
        responses = generate_sequence(prob_range)
        all_responses[prob_range] = responses
    
    print("✅ 所有概率级别的回复生成完成")
    return all_responses

def main():
    """主函数"""
    print("🚀 TOFU数据集第18条数据直接概率采样生成脚本启动")
    print("🎯 使用已在TOFU数据集上训练的Llama-3.2-1B模型")
    print("🎲 采样方式: 直接从高、中、低概率token区间采样，各5条")
    print("=" * 60)
    
    # 加载数据集
    dataset = load_tofu_dataset()
    
    # 获取第18条数据（索引17，因为从0开始）
    target_idx = 17
    if len(dataset) <= target_idx:
        print(f"❌ 错误：数据集只有{len(dataset)}条数据，无法获取第18条")
        return
    
    target_data = dataset[target_idx]
    question = target_data["question"]
    answer = target_data["answer"]
    
    print(f"📋 第18条数据内容:")
    print(f"问题: {question}")
    print(f"原始答案: {answer}")
    print("=" * 60)
    
    # 加载已训练的模型
    model, tokenizer = load_trained_model()
    
    # 格式化prompt
    prompt = format_prompt_with_system(question)
    print(f"🔧 格式化后的prompt预览:")
    print(f"{prompt[:200]}...")
    print("=" * 60)
    
    # 直接概率采样生成回复
    all_responses = generate_responses_by_direct_probability(model, tokenizer, prompt)
    
    # 输出结果
    print("📝 TOFU训练模型生成的直接概率采样回复:")
    print("=" * 60)
    
    for prob_type, responses in all_responses.items():
        print(f"🎯 {prob_type}回复 (共5条):")
        if prob_type == "高概率":
            print("   📊 采样范围: 概率排序前10%的token")
        elif prob_type == "中概率":
            print("   📊 采样范围: 概率排序10%-50%的token")
        else:  # 低概率
            print("   📊 采样范围: 概率排序50%-90%的token")
        
        for i, response in enumerate(responses, 1):
            print(f"   {i}. {response}")
        print("-" * 60)
    
    print("🎉 任务完成！")
    print("💡 说明：")
    print("   - 高概率回复：每步都选择概率最高的前10%token")
    print("   - 中概率回复：每步都选择概率中等的token (10%-50%)")
    print("   - 低概率回复：每步都选择概率较低的token (50%-90%)")
    print("   - 这种方法能更直接地控制生成内容的概率特性")

if __name__ == "__main__":
    main() 