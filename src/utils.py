from dataclasses import dataclass
from collections import Counter
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer
import torch
from datasets import load_dataset
from transformers import BatchEncoding
import torch.nn as nn
from transformers.data.data_collator import _torch_collate_batch
import math
import random

########## UTILS FOR TRAINING ##########
@dataclass
class PasswordDataCollator(DataCollatorForLanguageModeling):
    """
    CustomDataCollator for this task. It modifies the special token mask so that the end of password token is not ignored (should also be predicted).
    """
    def torch_call(self, examples):
        # Handle dict or lists with proper padding and conversion to tensor.
        if isinstance(examples[0], (dict, BatchEncoding)):
            batch = self.tokenizer.pad(examples, return_tensors="pt", pad_to_multiple_of=self.pad_to_multiple_of)
        else:
            batch = {
                "input_ids": _torch_collate_batch(examples, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)
            }

        special_tokens_mask = batch.pop("special_tokens_mask", None) # Remove if given
        
        # Create custom special tokens mask
        special_tokens_mask = torch.where(batch['input_ids'] != self.tokenizer.pad_token_id, 0, 1)
        
        if self.mlm:
            batch["input_ids"], batch["labels"] = self.torch_mask_tokens(
                batch["input_ids"], special_tokens_mask=special_tokens_mask
            )
        else:
            labels = batch["input_ids"].clone()
            if self.tokenizer.pad_token_id is not None:
                labels[labels == self.tokenizer.pad_token_id] = -100
            batch["labels"] = labels
        return batch

    
class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__