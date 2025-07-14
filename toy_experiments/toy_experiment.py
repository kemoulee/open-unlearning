#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NPO Unlearn实验脚本
监测不同概率回复在unlearn过程中的log-probability变化
"""

import hydra
import torch
import torch.nn.functional as F
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from omegaconf import DictConfig, OmegaConf
from datasets import load_dataset
from transformers import TrainerCallback
import json
from tqdm import tqdm

# 添加项目路径到sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from model import get_model
from data import get_data, get_collators
from trainer import load_trainer
from trainer.utils import seed_everything

class LogProbMonitor(TrainerCallback):
    """监测回复log-probability的回调函数"""
    
    def __init__(self, model, tokenizer, responses_dict, template_args):
        self.model = model
        self.tokenizer = tokenizer
        self.responses_dict = responses_dict
        self.template_args = template_args
        self.log_probs_history = []
        self.steps = []
        
    def compute_log_prob(self, question, answer):
        """计算给定问题-答案对的log-probability"""
        # 格式化输入（和生成时保持一致）
        system_prompt = "Cutting Knowledge Date: December 2023\nToday Date: 10 Apr 2025\n\nYou are a helpful assistant."
        
        # 分别编码问题和完整prompt
        question_prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        
        full_prompt = question_prompt + answer
        
        # 编码
        question_ids = self.tokenizer.encode(question_prompt, return_tensors="pt", add_special_tokens=False)
        full_ids = self.tokenizer.encode(full_prompt, return_tensors="pt", add_special_tokens=False)
        
        # 移动到模型设备
        device = next(self.model.parameters()).device
        full_ids = full_ids.to(device)
        
        # 计算log-probability
        with torch.no_grad():
            outputs = self.model(full_ids)
            logits = outputs.logits[0]  # [seq_len, vocab_size]
            
            # 只计算answer部分的log-prob
            answer_start = question_ids.shape[1]
            answer_tokens = full_ids[0, answer_start:]
            answer_logits = logits[answer_start-1:-1]  # shift by 1 for next token prediction
            
            if len(answer_tokens) == 0 or len(answer_logits) == 0:
                return -10.0  # 返回一个默认的低概率值
            
            # 计算log概率
            log_probs = F.log_softmax(answer_logits, dim=-1)
            token_log_probs = log_probs.gather(1, answer_tokens.unsqueeze(1)).squeeze(1)
            
            # 返回平均log-probability
            return token_log_probs.mean().item()
    
    def on_step_end(self, args, state, control, **kwargs):
        """每个step结束时计算log-probabilities"""
        if state.global_step % 50 == 0:  # 每50步监测一次
            print(f"🔍 Step {state.global_step}: 监测log-probabilities...")
            
            current_log_probs = {}
            
            # 计算每个回复的log-prob
            for response_type, responses in self.responses_dict.items():
                if response_type in ['高概率', '中概率', '低概率']:
                    # 计算5个回复的平均
                    log_probs = []
                    for response in responses:
                        log_prob = self.compute_log_prob(self.responses_dict['question'], response)
                        log_probs.append(log_prob)
                    current_log_probs[response_type] = np.mean(log_probs)
                else:
                    # ground_truth 和 fixed_response
                    log_prob = self.compute_log_prob(self.responses_dict['question'], responses)
                    current_log_probs[response_type] = log_prob
            
            self.log_probs_history.append(current_log_probs)
            self.steps.append(state.global_step)
            
            # 输出当前结果
            print(f"   Ground Truth: {current_log_probs['ground_truth']:.4f}")
            print(f"   Fixed Response: {current_log_probs['fixed_response']:.4f}")
            print(f"   高概率平均: {current_log_probs['高概率']:.4f}")
            print(f"   中概率平均: {current_log_probs['中概率']:.4f}")
            print(f"   低概率平均: {current_log_probs['低概率']:.4f}")

def generate_responses_by_direct_probability(model, tokenizer, question):
    """生成不同概率级别的回复（简化版）"""
    print("🎯 生成不同概率级别的回复...")
    
    # 格式化prompt
    system_prompt = "Cutting Knowledge Date: December 2023\nToday Date: 10 Apr 2025\n\nYou are a helpful assistant."
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
    
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    
    def sample_from_probability_range(logits, prob_range):
        """从指定概率范围采样token"""
        probs = F.softmax(logits, dim=-1)
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        
        vocab_size = probs.shape[-1]
        if prob_range == "高概率":
            start_idx, end_idx = 0, max(1, int(vocab_size * 0.1))
        elif prob_range == "中概率":
            start_idx, end_idx = int(vocab_size * 0.1), int(vocab_size * 0.5)
        else:  # 低概率
            start_idx, end_idx = int(vocab_size * 0.5), int(vocab_size * 0.9)
        
        range_indices = sorted_indices[start_idx:end_idx]
        range_probs = sorted_probs[start_idx:end_idx]
        range_probs = range_probs / range_probs.sum()
        
        sample_idx = torch.multinomial(range_probs, 1)
        return range_indices[sample_idx].item()
    
    def generate_sequence(prob_range, max_length=50):
        """生成指定概率类型的序列"""
        sequences = []
        
        for _ in range(5):  # 生成5条回复
            current_ids = input_ids.clone()
            
            for _ in range(max_length):
                with torch.no_grad():
                    outputs = model(current_ids)
                    logits = outputs.logits[0, -1, :]
                
                next_token = sample_from_probability_range(logits, prob_range)
                current_ids = torch.cat([current_ids, torch.tensor([[next_token]], device=device)], dim=1)
                
                if next_token == tokenizer.eos_token_id:
                    break
            
            generated_tokens = current_ids[0, input_ids.shape[1]:]
            response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            sequences.append(response.strip())
        
        return sequences
    
    # 生成不同概率级别的回复
    responses = {}
    for prob_range in ["高概率", "中概率", "低概率"]:
        print(f"   生成{prob_range}回复...")
        responses[prob_range] = generate_sequence(prob_range)
    
    return responses

def plot_log_probs(steps, log_probs_history, save_path):
    """绘制log-probability变化图"""
    print("📊 绘制log-probability变化图...")
    
    plt.figure(figsize=(12, 8))
    
    # 提取每条线的数据
    ground_truth_probs = [log_probs['ground_truth'] for log_probs in log_probs_history]
    fixed_response_probs = [log_probs['fixed_response'] for log_probs in log_probs_history]
    high_prob_probs = [log_probs['高概率'] for log_probs in log_probs_history]
    mid_prob_probs = [log_probs['中概率'] for log_probs in log_probs_history]
    low_prob_probs = [log_probs['低概率'] for log_probs in log_probs_history]
    
    # 绘制曲线
    plt.plot(steps, ground_truth_probs, 'g-', linewidth=2, label='Ground Truth', marker='o')
    plt.plot(steps, fixed_response_probs, 'r-', linewidth=2, label='Fixed Response ("She mainly writes in English.")', marker='s')
    plt.plot(steps, high_prob_probs, 'b-', linewidth=2, label='高概率回复平均', marker='^')
    plt.plot(steps, mid_prob_probs, 'orange', linewidth=2, label='中概率回复平均', marker='v')
    plt.plot(steps, low_prob_probs, 'purple', linewidth=2, label='低概率回复平均', marker='d')
    
    plt.xlabel('Training Steps', fontsize=12)
    plt.ylabel('Log-Probability', fontsize=12)
    plt.title('NPO Unlearning过程中回复Log-Probability变化', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📈 图表已保存到: {save_path}")

@hydra.main(version_base=None, config_path=".", config_name="experiment_config.yaml")
def main(cfg: DictConfig):
    """主实验函数"""
    print("🚀 NPO Unlearn实验开始")
    print("=" * 60)
    
    print(f"🔧 实验配置:")
    print(f"   Trainer: {cfg.trainer.handler}")
    print(f"   Model: {cfg.model.model_args.pretrained_model_name_or_path}")
    print(f"   Forget Split: {cfg.forget_split}")
    print(f"   Retain Split: {cfg.retain_split}")
    print("=" * 60)
    
    # 设置随机种子
    seed_everything(42)
    
    # 加载模型
    print("🤖 加载模型...")
    model, tokenizer = get_model(cfg.model)
    
    # 加载TOFU数据集第18条数据
    print("📥 加载TOFU数据集...")
    dataset = load_dataset("locuslab/TOFU", name="forget10", split="train")
    target_idx = 17  # 第18条数据
    target_data = dataset[target_idx]
    question = target_data["question"]
    ground_truth = target_data["answer"]
    
    print(f"📋 实验数据:")
    print(f"   问题: {question}")
    print(f"   Ground Truth: {ground_truth}")
    print("=" * 60)
    
    # 生成不同概率级别的回复
    print("🎲 生成不同概率级别的回复...")
    generated_responses = generate_responses_by_direct_probability(model, tokenizer, question)
    
    # 准备监测的回复字典
    responses_dict = {
        'question': question,
        'ground_truth': ground_truth,
        'fixed_response': "She mainly writes in English.",
        **generated_responses
    }
    
    print("📝 生成的回复:")
    for response_type, responses in generated_responses.items():
        print(f"   {response_type}: {len(responses)}条")
        for i, response in enumerate(responses[:2], 1):  # 只显示前2条
            print(f"     {i}. {response[:50]}...")
    print("=" * 60)
    
    # 准备训练数据
    print("📊 准备训练数据...")
    data = get_data(cfg.data, mode="unlearn", tokenizer=tokenizer, template_args=cfg.model.template_args)
    collator = get_collators(cfg.collator, tokenizer=tokenizer)
    
    # 创建log-prob监测器
    monitor = LogProbMonitor(model, tokenizer, responses_dict, cfg.model.template_args)
    
    # 设置trainer
    print("🏃 设置trainer...")
    trainer, trainer_args = load_trainer(
        trainer_cfg=cfg.trainer,
        model=model,
        train_dataset=data.get("train", None),
        eval_dataset=data.get("eval", None),
        tokenizer=tokenizer,
        data_collator=collator,
        evaluators=None,
        template_args=cfg.model.template_args,
    )
    
    # 添加回调函数
    trainer.add_callback(monitor)
    
    # 执行初始监测
    print("🔍 执行初始监测...")
    monitor.on_step_end(trainer_args, trainer.state, None)
    
    # 开始训练
    print("🏋️ 开始NPO unlearning...")
    trainer.train()
    
    # 绘制结果
    save_path = "toy_experiments/npo_unlearn_log_probs.png"
    plot_log_probs(monitor.steps, monitor.log_probs_history, save_path)
    
    # 保存数据
    results = {
        'steps': monitor.steps,
        'log_probs_history': monitor.log_probs_history,
        'question': question,
        'ground_truth': ground_truth,
        'generated_responses': generated_responses
    }
    
    results_path = "toy_experiments/npo_unlearn_results.json"
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print("🎉 实验完成！")
    print(f"📈 图表保存到: {save_path}")
    print(f"💾 数据保存到: {results_path}")

if __name__ == "__main__":
    main() 