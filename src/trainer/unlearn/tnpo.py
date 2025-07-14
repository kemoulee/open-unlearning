import torch
import torch.nn.functional as F
from trainer.utils import compute_token_nll
from trainer.unlearn.grad_diff import GradDiff


class TNPO(GradDiff):
    """
    Token-level NPO实现，使用token-level自适应权重计算weighted NLL。
    
    基于NPO的梯度分析，但在token级别应用权重：
    对每个token i: w_i = 2 * σ(β * r_π_i)
    其中 r_π_i = log(π_θ(s_i|s_{<i}) / π_ref(s_i|s_{<i}))
    
    L_forget = -∑_i w_i * NLL_i
    
    相比sequence-level的NPOW，TNPO在每个token上应用不同的权重，
    能够更精细地控制遗忘过程。
    """
    def __init__(self, beta=5.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = beta
        if self.ref_model is None:
            self.ref_model = self._prepare_ref_model(self.model)

    def compute_loss(self, model, inputs, return_outputs=False):
        forget_inputs = inputs["forget"]

        # 1. 计算当前模型在forget数据上的token-level NLL loss
        forget_token_loss, forget_outputs = compute_token_nll(model, forget_inputs)
        
        # 2. 计算参考模型在forget数据上的token-level NLL loss (no grad)
        with torch.no_grad():
            forget_ref_token_loss, _ = compute_token_nll(self.ref_model, forget_inputs)
        
        # 3. 计算token-level log概率比率 r_π_i = -(NLL_θ_i - NLL_ref_i)
        token_log_ratio = -(forget_token_loss - forget_ref_token_loss)
        
        # 4. 计算token-level自适应权重 w_i = 2 * σ(β * r_π_i)
        # 这个权重反映了当前模型相对于参考模型对每个token的"熟悉程度"
        token_adaptive_weight = 2.0 * torch.sigmoid(self.beta * token_log_ratio).detach()
        
        # 5. 创建mask来忽略padding tokens (-100)
        labels = forget_inputs["labels"]
        shifted_labels = labels[..., 1:].contiguous()
        loss_mask = shifted_labels != -100
        
        # 6. 计算加权token loss（负号实现梯度上升）
        # 只对非padding的tokens计算损失
        weighted_token_loss = token_adaptive_weight * forget_token_loss * loss_mask
        
        # 7. 计算最终的forget loss
        # 对每个序列求和，然后求平均
        forget_loss_weighted = -(weighted_token_loss.sum(dim=-1)).mean()

        # 8. 计算retain loss
        retain_inputs = inputs["retain"]
        retain_inputs = {
            "input_ids": retain_inputs["input_ids"],
            "attention_mask": retain_inputs["attention_mask"],
            "labels": retain_inputs["labels"],
        }
        retain_loss = self.compute_retain_loss(model=model, retain_inputs=retain_inputs)

        # 9. 组合total loss
        loss = self.gamma * forget_loss_weighted + self.alpha * retain_loss
        return (loss, forget_outputs) if return_outputs else loss 