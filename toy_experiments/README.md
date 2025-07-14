# NPO Unlearn Toy Experiment

这个实验监测NPO unlearning过程中不同概率回复的log-probability变化。

## 实验概述

1. **模型**: Llama-3.2-1B-Instruct (已在TOFU数据集上训练)
2. **方法**: NPO (Negative Preference Optimization)
3. **数据**: TOFU forget10% 第18条数据
4. **监测**: 17个回复的log-probability变化
   - 5个高概率回复（平均）
   - 5个中概率回复（平均） 
   - 5个低概率回复（平均）
   - 1个ground truth
   - 1个固定回复 ("She mainly writes in English.")

## 文件说明

- `toy_experiment.py`: 主实验脚本
- `experiment_config.yaml`: 实验配置文件
- `run_experiment.sh`: 运行脚本
- `tofu_sample_generation.py`: 回复生成脚本（参考）

## 使用方法

### 方法1: 使用运行脚本
```bash
cd /home/yc47912/open-unlearning/toy_experiments
chmod +x run_experiment.sh
./run_experiment.sh
```

### 方法2: 直接运行Python脚本
```bash
cd /home/yc47912/open-unlearning/toy_experiments
python toy_experiment.py
```

### 方法3: 自定义参数
```bash
cd /home/yc47912/open-unlearning/toy_experiments
python toy_experiment.py \
    trainer.args.num_train_epochs=5 \
    trainer.args.learning_rate=2e-5 \
    task_name=custom_experiment
```

## 输出文件

- `npo_unlearn_log_probs.png`: log-probability变化图表
- `npo_unlearn_results.json`: 实验数据（JSON格式）

## 实验流程

1. 🤖 加载预训练模型
2. 📥 加载TOFU数据集第18条数据
3. 🎲 生成高、中、低概率回复各5条
4. 🔍 执行初始log-probability监测
5. 🏋️ 开始NPO unlearning训练
6. 📊 每50步监测一次log-probability
7. 📈 绘制变化图表并保存结果

## 预期结果

- **Ground Truth**: 在unlearning过程中log-probability应该下降
- **固定回复**: 作为对照组，变化应该相对较小
- **高概率回复**: 可能受到较大影响
- **中/低概率回复**: 影响程度可能不同

## 注意事项

- 确保有足够的GPU内存（推荐8GB+）
- 实验时间约30-60分钟（取决于硬件配置）
- 结果图表会自动保存到当前目录 