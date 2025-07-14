#!/bin/bash

# NPO Unlearn Toy实验运行脚本
echo "🚀 启动NPO Unlearn实验..."

# 设置工作目录
cd /home/yc47912/open-unlearning/toy_experiments

# 运行实验
python toy_experiment.py \
    trainer.handler=NPO \
    model.model_args.pretrained_model_name_or_path=open-unlearning/tofu_Llama-3.2-1B-Instruct_full \
    forget_split=forget10 \
    retain_split=retain90 \
    task_name=1B_10_NPO_toy

echo "✅ 实验完成！" 