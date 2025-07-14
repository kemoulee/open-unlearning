import torch
import torch.nn.functional as F
from trainer.utils import compute_batch_nll
from trainer.unlearn.grad_diff import GradDiff


class NPOW(GradDiff):
    """
    NPO的近似实现，使用自适应权重直接计算weighted NLL。
    
    基于梯度分析，NPO的梯度为：
    ∂L/∂θ = -2 * σ(β * r_π) * ∂L_θ/∂θ
    
    这启发我们使用自适应权重 w = 2 * σ(β * r_π) 来加权NLL：
    L_forget ≈ -(w * L_θ).mean()
    
    注意：这是一个近似实现，忽略了权重函数对参数的依赖性。
    在实践中，当模型参数变化不大时，这个近似是合理的。
    """
    def __init__(self, beta=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = beta
        if self.ref_model is None:
            self.ref_model = self._prepare_ref_model(self.model)

    def compute_loss(self, model, inputs, return_outputs=False):
        forget_inputs = inputs["forget"]

        # 1. 计算当前模型在forget数据上的NLL loss
        forget_loss, forget_outputs = compute_batch_nll(model, forget_inputs)
        
        # 2. 计算参考模型在forget数据上的NLL loss (no grad)
        with torch.no_grad():
            forget_ref_loss, _ = compute_batch_nll(self.ref_model, forget_inputs)
        
        # 3. 计算log概率比率 r_π = log(π_θ/π_ref) = -(L_θ - L_ref)
        log_ratio = -(forget_loss - forget_ref_loss)
        
        # 4. 计算自适应权重 w = 2 * σ(β * r_π)
        # 这个权重反映了当前模型相对于参考模型对forget数据的"熟悉程度"
        adaptive_weight = 2.0 * torch.sigmoid(self.beta * log_ratio).detach()
        
        # 5. 计算加权forget loss（负号实现梯度上升）
        # 权重越大，表示模型越"记得"这些数据，遗忘力度应该越大
        forget_loss_weighted = -(adaptive_weight * forget_loss).mean()

        # 6. 计算retain loss
        retain_inputs = inputs["retain"]
        retain_inputs = {
            "input_ids": retain_inputs["input_ids"],
            "attention_mask": retain_inputs["attention_mask"],
            "labels": retain_inputs["labels"],
        }
        retain_loss = self.compute_retain_loss(model=model, retain_inputs=retain_inputs)

        # 7. 组合total loss
        loss = self.gamma * forget_loss_weighted + self.alpha * retain_loss
        return (loss, forget_outputs) if return_outputs else loss 