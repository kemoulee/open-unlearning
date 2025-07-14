# src/trainer/unlearn/bss.py

import torch
from torch import nn
import torch.nn.functional as F
from typing import Dict, Any, List
import logging

from trainer.unlearn.grad_diff import GradDiff
from data.utils import IGNORE_INDEX

logger = logging.getLogger(__name__)

class BSS(GradDiff):
    """
    Bootstrapping-Sentence (BS-S) Unlearning Trainer.
    
    This trainer implements the BS-S method from "LLM Unlearning with LLM Beliefs".
    It augments the unlearning data with responses generated from the model itself
    to mitigate the squeezing effect and achieve a more thorough unlearning.
    """

    def __init__(
        self,
        lambda_bss: float = 0.5,
        n_samples: int = 1,
        temperature: float = 1.0,
        regeneration_epochs: float = 2.0,
        *args,
        **kwargs
    ):
        """
        Initializes the BSS trainer.

        Args:
            lambda_bss (float): The trade-off hyperparameter for augmented data loss.
            n_samples (int): The number of augmented responses to generate per input.
            temperature (float): The temperature for sampling during generation.
            regeneration_epochs (float): Frequency (in epochs) for regenerating augmented data. 
                                         Set to 1.0 to regenerate every epoch.
                                         Set to 0 to regenerate every step (very slow).
        """
        super().__init__(*args, **kwargs)
        self.lambda_bss = lambda_bss
        self.n_samples = n_samples
        self.temperature = temperature
        self.regeneration_epochs = regeneration_epochs
        
        # Cache for storing the generated augmented data
        self.augmented_data_cache = None
        self.last_regeneration_epoch = -1.0

    def _prepare_prompts_for_generation(self, inputs: Dict[str, torch.Tensor]) -> List[torch.Tensor]:
        """Extracts the prompt part from the input_ids for generation."""
        prompts = []
        for i in range(inputs["input_ids"].shape[0]):
            # Find the start of the response by finding the first non-IGNORE_INDEX label
            # The tokens before this point belong to the prompt
            response_start_index = (inputs["labels"][i] != IGNORE_INDEX).nonzero(as_tuple=True)[0]
            if len(response_start_index) == 0:
                # If all labels are ignored, the whole sequence is the prompt
                prompt_end_index = inputs["input_ids"].shape[1]
            else:
                prompt_end_index = response_start_index[0]
            
            prompts.append(inputs["input_ids"][i, :prompt_end_index])
        return prompts

    def _generate_and_process_augmented_data(self, model: nn.Module, forget_inputs: Dict[str, torch.Tensor]):
        """
        Generates new responses for the forget set prompts and processes them into a training-ready batch.
        This is a computationally expensive operation.
        """
        logger.info(f"Regenerating augmented data for BS-S at epoch {self.state.epoch:.2f}...")
        
        # 1. Isolate prompts from the forget data
        prompts = self._prepare_prompts_for_generation(forget_inputs)
        
        # Store newly generated samples
        new_input_ids, new_labels, new_attention_mask = [], [], []

        model.eval() # Switch to evaluation mode for generation
        with torch.no_grad():
            for prompt_ids in prompts:
                prompt_ids = prompt_ids.unsqueeze(0).to(self.accelerator.device)
                
                # Create attention mask for the prompt
                attention_mask = torch.ones_like(prompt_ids)
                
                # 2. Generate N new responses for each prompt
                generated_outputs = model.generate(
                    input_ids=prompt_ids,
                    attention_mask=attention_mask,
                    do_sample=True,
                    top_p=None,
                    temperature=self.temperature,
                    num_return_sequences=self.n_samples,
                    max_new_tokens=200,
                    use_cache=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    tokenizer=self.tokenizer,
                    stop_strings=["<|eot_id|>"],
                )

                # 3. Process each generated sequence into a new training sample
                for output in generated_outputs:
                    # Remove prompt part from the generated output
                    generated_ids = output[prompt_ids.shape[1]:]
                    
                    # Clean up generation: remove any tokens after stop token
                    generated_ids = self._clean_generated_sequence(generated_ids)

                    # Create the new sample
                    aug_input_ids = torch.cat([prompt_ids.squeeze(0), generated_ids])
                    aug_labels = torch.cat([torch.full_like(prompt_ids.squeeze(0), IGNORE_INDEX), generated_ids])
                    aug_attention_mask = torch.ones_like(aug_input_ids)

                    new_input_ids.append(aug_input_ids)
                    new_labels.append(aug_labels)
                    new_attention_mask.append(aug_attention_mask)
        
        model.train() # Switch back to training mode
        
        # 4. Collate with left padding (consistent with TOFU evaluation)
        padded_input_ids = self._pad_sequences_left(new_input_ids, padding_value=self.tokenizer.pad_token_id)
        padded_labels = self._pad_sequences_left(new_labels, padding_value=IGNORE_INDEX)
        padded_attention_mask = self._pad_sequences_left(new_attention_mask, padding_value=0)

        self.augmented_data_cache = {
            "input_ids": padded_input_ids.to(self.accelerator.device),
            "labels": padded_labels.to(self.accelerator.device),
            "attention_mask": padded_attention_mask.to(self.accelerator.device),
        }
        self.last_regeneration_epoch = self.state.epoch

    def _clean_generated_sequence(self, generated_ids: torch.Tensor) -> torch.Tensor:
        """Remove tokens after stop sequences."""
        # Convert to list for easier processing
        ids_list = generated_ids.tolist()
        
        # Look for <|eot_id|> token
        eot_token_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        if eot_token_id in ids_list:
            # Find first occurrence and truncate
            eot_idx = ids_list.index(eot_token_id)
            ids_list = ids_list[:eot_idx + 1]  # Include the eot token
        
        return torch.tensor(ids_list, device=generated_ids.device)

    def _pad_sequences_left(self, sequences, padding_value):
        """Apply left padding to sequences, consistent with generation tasks."""
        # Reverse sequences, apply right padding, then reverse back
        reversed_sequences = [torch.flip(seq, dims=[0]) for seq in sequences]
        padded_reversed = nn.utils.rnn.pad_sequence(
            reversed_sequences, batch_first=True, padding_value=padding_value
        )
        # Reverse back to get left padding
        return torch.flip(padded_reversed, dims=[1])

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Computes the combined loss for the unlearning task using the BS-S method.
        """
        forget_inputs = inputs["forget"]
        retain_inputs = inputs["retain"]

        # --- Original Forget Loss (Standard Gradient Ascent) ---
        original_forget_outputs = model(**forget_inputs)
        original_forget_loss = -original_forget_outputs.loss

        # --- Augmented Forget Loss (on Generated Data) ---
        
        # Check if we need to regenerate the augmented data
        should_regenerate = (self.augmented_data_cache is None) or \
                            (self.regeneration_epochs > 0 and (self.state.epoch - self.last_regeneration_epoch >= self.regeneration_epochs)) or \
                            (self.regeneration_epochs == 0) # Regenerate every step


        if self.is_in_train and should_regenerate:
            self._generate_and_process_augmented_data(model, forget_inputs)

        # Calculate loss on the cached augmented data
        if self.augmented_data_cache:
            augmented_forget_outputs = model(**self.augmented_data_cache)
            augmented_forget_loss = -augmented_forget_outputs.loss
        else:
            # If no augmented data is available (e.g., first step), use zero loss
            augmented_forget_loss = torch.tensor(0.0, device=model.device)

        # --- Combine Forget Losses (Equation 8 from the paper) ---
        forget_loss = (1 - self.lambda_bss) * original_forget_loss + self.lambda_bss * augmented_forget_loss

        # --- Retain Loss (Inherited from GradDiff) ---
        retain_loss = self.compute_retain_loss(model=model, retain_inputs=retain_inputs)

        # --- Final Combined Loss ---
        loss = self.gamma * forget_loss + self.alpha * retain_loss

        return (loss, original_forget_outputs) if return_outputs else loss