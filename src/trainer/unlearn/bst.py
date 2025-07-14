# src/trainer/unlearn/bst.py

import torch
import torch.nn.functional as F
from torch import nn

from trainer.unlearn.grad_diff import GradDiff
from data.utils import IGNORE_INDEX


class BST(GradDiff):
    """
    Bootstrapping-based Unlearning Trainer.

    This trainer implements the unlearning method described in "LLM Unlearning with LLM Beliefs".
    It supports Bootstrapping-Token (BS-T) to mitigate the squeezing effect by unlearning
    not only the original data but also the model's own beliefs.
    """

    def __init__(self, lambda_bst: float = 0.5, bst_mode: str = "token", detach_mode: str = "yes", top_k: int = 20, *args, **kwargs):
        """
        Initializes the BST trainer.

        Args:
            lambda_bst (float, optional): The interpolation coefficient for bootstrapping repulsion,
                                          balancing between the model's belief and the original target. Defaults to 0.5.
            bst_mode (str, optional): The bootstrapping mode. Currently, only "token" (BS-T) is implemented.
                                      Defaults to "token".
            detach_mode (str, optional): Whether to detach model beliefs from the computation graph.
                                         "yes" - detach beliefs (default behavior), "no" - keep beliefs in graph.
                                         Defaults to "yes".
            top_k (int, optional): Number of top tokens to consider when computing belief probabilities.
                                   Only the top K logits will be kept and renormalized. Defaults to 20.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__(*args, **kwargs)
        self.lambda_bst = lambda_bst
        self.bst_mode = bst_mode
        self.detach_mode = detach_mode
        self.top_k = top_k

        if self.bst_mode != "token":
            raise NotImplementedError(
                'Only "token" mode (Bootstrapping-Token) is currently implemented.'
            )

        if self.detach_mode not in ["yes", "no"]:
            raise ValueError(
                f'detach_mode must be either "yes" or "no", got "{self.detach_mode}"'
            )

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Computes the combined loss for the unlearning task using the BST method.
        The forget loss is calculated using the Bootstrapping-Token (BS-T) objective,
        and the retain loss is inherited from the parent GradDiff trainer.
        """
        # Unpack inputs from the ForgetRetainDataset
        forget_inputs = inputs["forget"]
        retain_inputs = inputs["retain"]

        # --- Forget Loss (BS-T Objective) ---

        # 1. Forward pass to get model's predictions (logits)
        outputs = model(**forget_inputs)
        logits = outputs.logits

        # 2. Prepare shifted logits and labels for next-token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        labels = forget_inputs["labels"]
        shift_labels = labels[..., 1:].contiguous()

        # 3. Get the model's belief (current predicted probabilities) with topK constraint
        # This is pi_theta in the paper's formula and must be treated as a constant for the target.
        
        # Apply topK constraint: only keep top K logits and set others to -inf
        top_k_logits = shift_logits.clone()
        if self.top_k < shift_logits.size(-1):
            # Get topK values and indices
            topk_values, topk_indices = torch.topk(shift_logits, self.top_k, dim=-1)
            
            # Create a mask for topK positions
            mask = torch.full_like(shift_logits, float('-inf'))
            mask.scatter_(-1, topk_indices, topk_values)
            top_k_logits = mask
        
        # Apply softmax to get belief probabilities (now only topK will have non-zero probs)
        belief_probs = F.softmax(top_k_logits, dim=-1)
        if self.detach_mode == "yes":
            belief_probs = belief_probs.detach()

        # 4. Create one-hot representation of the original labels
        # Clamp shift_labels to valid range [0, num_classes) to prevent CUDA errors
        # IGNORE_INDEX (-100) values will be clamped to 0, but then zeroed out below
        shift_labels_clamped = torch.clamp(shift_labels.long(), min=0, max=logits.size(-1) - 1)
        one_hot_labels = F.one_hot(
            shift_labels_clamped, num_classes=logits.size(-1)
        ).float()
        # Ensure that ignored indices do not contribute to the one-hot target
        one_hot_labels[shift_labels == IGNORE_INDEX] = 0

        # 5. Construct the soft target by interpolating between belief and original target
        # This corresponds to t_u in Equation (6) of the paper 
        soft_target = (self.lambda_bst * belief_probs +
                       (1 - self.lambda_bst) * one_hot_labels)

        # 6. Calculate cross-entropy loss with the soft target.
        # This corresponds to Equation (7). We multiply by -1 for gradient ascent.
        log_probs = F.log_softmax(shift_logits, dim=-1)
        bst_loss_per_token = -(soft_target * log_probs).sum(dim=-1)

        # 7. Apply mask to ignore padding tokens and average the loss
        loss_mask = shift_labels != IGNORE_INDEX
        bst_loss = (bst_loss_per_token * loss_mask).sum(dim=-1) / loss_mask.sum(dim=-1)

        # Final forget loss for gradient ascent (maximize log-likelihood of soft target)
        forget_loss = -bst_loss.mean()

        # --- Retain Loss (Inherited from GradDiff) ---
        retain_loss = self.compute_retain_loss(model=model, retain_inputs=retain_inputs)

        # --- Combine Losses ---
        # The final loss is a weighted sum of the forget and retain losses.
        loss = self.gamma * forget_loss + self.alpha * retain_loss

        return (loss, outputs) if return_outputs else loss