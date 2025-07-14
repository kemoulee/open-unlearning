import torch
from torch.nn import CrossEntropyLoss
from trainer.unlearn.grad_diff import GradDiff

class WGA(GradDiff):
    """
    Weighted Gradient Ascent (WGA) unlearning trainer.

    This method performs gradient ascent on the forget set, but weights each token's loss
    by raising the token probability to the beta power.
    This implementation is based on user-provided code logic.
    """
    def __init__(self, beta=5.0, *args, **kwargs):
        """
        Initialize WGA trainer.

        Args:
            beta (float): Hyperparameter for controlling token probability weights.
            *args, **kwargs: Parameters passed to the parent GradDiff trainer.
        """
        super().__init__(*args, **kwargs)
        self.beta = beta

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Compute the combined loss for WGA.
        
        This includes weighted gradient ascent loss on the forget set and standard loss on the retain set.
        """
        # 1. Perform weighted gradient ascent on the forget set
        forget_inputs = inputs["forget"]
        outputs = model(**forget_inputs)
        
        # Extract necessary parts from model outputs and inputs for loss computation
        logits = outputs.logits
        labels = forget_inputs.get("labels")


        # Compute cross-entropy loss for each token (without reduction)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = CrossEntropyLoss(ignore_index=-100, reduction='none')
        lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        # Calculate weights based on token probabilities (weights = p^beta = (exp(-loss))^beta)
        with torch.no_grad():
            weights = ((-lm_loss).exp() ** self.beta)

        # Apply weights to loss for gradient ascent
        weighted_loss = weights * lm_loss
        
        # 1. Reshape the 1D loss back to 2D (batch_size, sequence_length)
        #    shift_labels 的形状是 (batch_size, sequence_length - 1)，正好是我们需要的形状
        weighted_loss_2d = weighted_loss.view(shift_labels.shape)

        # 2. Create a 2D mask to zero out invalid tokens
        mask_2d = (shift_labels != -100).float()

        # 3. Apply the mask and sum the weighted losses for each sequence
        #    乘以mask后，padding位置的损失变为0，不影响求和
        sequence_loss_sum = (weighted_loss_2d * mask_2d).sum(dim=-1)

        # 4. Compute the final forget loss by averaging the sequence-level sums
        #    这现在与TNPO的归一化方式完全相同
        forget_loss = -sequence_loss_sum.mean()

        # 2. Compute loss on retain set (reuse parent class logic)
        retain_inputs = inputs["retain"]
        retain_loss = self.compute_retain_loss(model=model, retain_inputs=retain_inputs)

        # 3. Combined loss
        # self.gamma and self.alpha come from parent class GradDiff
        loss = self.gamma * forget_loss + self.alpha * retain_loss

        return (loss, outputs) if return_outputs else loss