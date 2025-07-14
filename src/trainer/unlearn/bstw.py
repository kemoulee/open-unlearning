import torch
import torch.nn.functional as F
from torch import nn

from trainer.unlearn.grad_diff import GradDiff
from trainer.utils import compute_batch_nll
from data.utils import IGNORE_INDEX


class BSTW(GradDiff):
    """
    BST with NPOW-style adaptive Weight (BSTW) - BST方法的NPOW风格自适应权重版本。
    
    权重计算与NPOW相同：基于概率比的对数 r_π = log(π_θ/π_ref)
    但损失函数仍使用BST loss进行遗忘
    
    w = 2 * σ(β * r_π)，其中 r_π = log(P_θ/P_ref) = -(NLL_θ - NLL_ref)
    最终损失：L_forget = -(w * BST_loss).mean()
    """

    def __init__(self, lambda_bst: float = 0.5, bst_mode: str = "token", detach_mode: str = "yes", 
                 top_k: int = 20, beta: float = 1.0, *args, **kwargs):
        """
        Initializes the BSTW trainer.

        Args:
            lambda_bst (float, optional): The interpolation coefficient for bootstrapping repulsion. Defaults to 0.5.
            bst_mode (str, optional): The bootstrapping mode. Only "token" is implemented. Defaults to "token".
            detach_mode (str, optional): Whether to detach model beliefs from computation graph. Defaults to "yes".
            top_k (int, optional): Number of top tokens to consider when computing belief probabilities. Defaults to 20.
            beta (float, optional): Beta parameter for adaptive weight calculation. Defaults to 1.0.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__(*args, **kwargs)
        self.lambda_bst = lambda_bst
        self.bst_mode = bst_mode
        self.detach_mode = detach_mode
        self.top_k = top_k
        self.beta = beta
        
        if self.ref_model is None:
            self.ref_model = self._prepare_ref_model(self.model)

        if self.bst_mode != "token":
            raise NotImplementedError(
                'Only "token" mode (Bootstrapping-Token) is currently implemented.'
            )

        if self.detach_mode not in ["yes", "no"]:
            raise ValueError(
                f'detach_mode must be either "yes" or "no", got "{self.detach_mode}"'
            )

    def compute_bst_loss(self, model, inputs, is_ref_model=False):
        """
        计算BST损失，可以用于当前模型或参考模型
        
        Args:
            model: 要计算损失的模型
            inputs: 输入数据
            is_ref_model: 是否为参考模型（如果是，则不计算梯度）
            
        Returns:
            bst_loss: 每个序列的BST损失 (batch_size,)
            outputs: 模型输出
        """
        context_manager = torch.no_grad() if is_ref_model else torch.enable_grad()
        
        with context_manager:
            # 1. Forward pass to get model's predictions (logits)
            outputs = model(**inputs)
            logits = outputs.logits

            # 2. Prepare shifted logits and labels for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            labels = inputs["labels"]
            shift_labels = labels[..., 1:].contiguous()

            # 3. Get the model's belief with topK constraint
            top_k_logits = shift_logits.clone()
            if self.top_k < shift_logits.size(-1):
                topk_values, topk_indices = torch.topk(shift_logits, self.top_k, dim=-1)
                mask = torch.full_like(shift_logits, float('-inf'))
                mask.scatter_(-1, topk_indices, topk_values)
                top_k_logits = mask
            
            # Apply softmax to get belief probabilities
            belief_probs = F.softmax(top_k_logits, dim=-1)
            if self.detach_mode == "yes" and not is_ref_model:
                belief_probs = belief_probs.detach()

            # 4. Create one-hot representation of the original labels
            shift_labels_clamped = torch.clamp(shift_labels.long(), min=0, max=logits.size(-1) - 1)
            one_hot_labels = F.one_hot(
                shift_labels_clamped, num_classes=logits.size(-1)
            ).float()
            one_hot_labels[shift_labels == IGNORE_INDEX] = 0

            # 5. Construct the soft target
            soft_target = (self.lambda_bst * belief_probs +
                           (1 - self.lambda_bst) * one_hot_labels)

            # 6. Calculate cross-entropy loss with the soft target
            log_probs = F.log_softmax(shift_logits, dim=-1)
            bst_loss_per_token = -(soft_target * log_probs).sum(dim=-1)

            # 7. Apply mask to ignore padding tokens and average per sequence
            loss_mask = shift_labels != IGNORE_INDEX
            bst_loss = (bst_loss_per_token * loss_mask).sum(dim=-1)

        return bst_loss, outputs

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        计算BSTW损失：权重基于概率比（和NPOW完全相同），损失基于BST
        """
        forget_inputs = inputs["forget"]

        # === 第1步：计算权重（和NPOW完全相同）===
        # 1.1 计算当前模型和参考模型的NLL loss (用于权重计算)
        forget_nll_loss, _ = compute_batch_nll(model, forget_inputs)
        
        with torch.no_grad():
            forget_ref_nll_loss, _ = compute_batch_nll(self.ref_model, forget_inputs)
        
        # 1.2 计算概率比的对数 r_π = log(P_θ/P_ref) = -(NLL_θ - NLL_ref)
        log_ratio = -(forget_nll_loss - forget_ref_nll_loss)
        
        # 1.3 计算自适应权重 w = 2 * σ(β * r_π) (和NPOW相同)
        adaptive_weight = 2.0 * torch.sigmoid(self.beta * log_ratio).detach()

        # === 第2步：计算BST损失（用于实际遗忘）===
        # 2.1 计算当前模型的BST loss
        forget_bst_loss, forget_outputs = self.compute_bst_loss(model, forget_inputs, is_ref_model=False)
        
        # 2.2 使用权重加权BST loss（负号实现梯度上升）
        forget_loss_weighted = -(adaptive_weight * forget_bst_loss).mean()

        # === 第3步：计算retain loss ===
        retain_inputs = inputs["retain"]
        retain_inputs = {
            "input_ids": retain_inputs["input_ids"],
            "attention_mask": retain_inputs["attention_mask"],
            "labels": retain_inputs["labels"],
        }
        retain_loss = self.compute_retain_loss(model=model, retain_inputs=retain_inputs)

        # === 第4步：组合总损失 ===
        loss = self.gamma * forget_loss_weighted + self.alpha * retain_loss
        return (loss, forget_outputs) if return_outputs else loss 