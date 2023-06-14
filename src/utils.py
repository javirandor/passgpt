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


########## UTILS FOR EVALUATION ##########
def top1_accuracy(true, logits, device):
    """
    Number of correct characters guessed (ignore start of sentence token)
    Returns: number of correct, total number, replacement dict
    """
    predictions = logits.argmax(1)
    end_index = (true == 2).nonzero(as_tuple=False)[0]
    correct_characters = (true[:end_index+1].to(device) == predictions[:end_index+1]).sum()
    diff = (true[:end_index+1].to(device) != predictions[:end_index+1])
    replacements = dict(zip(true[:end_index+1][diff].cpu().numpy(), predictions[:end_index+1][diff].cpu().numpy()))
    return correct_characters, end_index+1, replacements

def topk_accuracy(true, logits, k, device):
    """
    Number of correct characters guessed (ignore start of sentence token) among the topk candidates
    Returns: number of correct, total number
    """
    topk_predictions = logits.topk(k, 1, True, True).indices
    end_index = (true == 2).nonzero(as_tuple=False)[0]
    
    topk = topk_predictions.t()[:, :end_index+1]
    target = true[:end_index+1].to(device)

    correct = topk.eq(target.view(1, -1).expand_as(topk))
    n_correct = correct[:k].reshape(-1).float().sum(0)
    return n_correct, end_index+1

softmax = nn.Softmax(dim=1)

def topk_passwords(logits, num_pass=3):
    passwords = []

    predictions = logits.argmax(1)
    passwords.append(predictions)
    
    try:
        end_index = (predictions == 2).nonzero(as_tuple=False)[0]
    except:
        end_index = len(predictions)-1
        #print(predictions)

    relevant_logits = logits[:end_index+1]
    probs = softmax(relevant_logits)
    topk_probs, topk_values = probs.topk(num_pass, 1, True, True)

    acc_prob = topk_probs[:, 0].squeeze(0)
    next_index = torch.ones(acc_prob.shape, dtype=int)

    for i in range(num_pass-1):
        temp = torch.clone(predictions)
        min_prob = acc_prob.argmin()
        next_char = topk_values[min_prob, next_index[min_prob]]
        temp[min_prob] = next_char
        acc_prob[min_prob] += topk_probs[min_prob, next_index[min_prob]]
        next_index[min_prob] += 1
        passwords.append(temp)
    
    return passwords

def topk_passwords_multiple(logits, num_pass=3):
    passwords = []

    predictions = logits.argmax(1)
    passwords.append(predictions)
    
    try:
        end_index = (predictions == 2).nonzero(as_tuple=False)[0]
    except:
        end_index = len(predictions)-1

    relevant_logits = logits[:end_index+1]
    probs = softmax(relevant_logits)
    topk_probs, topk_values = probs.topk(num_pass, 1, True, True)

    acc_prob = topk_probs[:, 0].squeeze(0)
    next_index = torch.ones(acc_prob.shape, dtype=int)

    for i in range(num_pass-1):
        min_prob = acc_prob.argmin()
        next_char = topk_values[min_prob, next_index[min_prob]]
        replacements = []
        
        for i in passwords:
            temp = i.clone()
            temp[min_prob] = next_char
            replacements.append(temp.clone().cpu())
        
        passwords = list(set(passwords).union(set(replacements)))
        acc_prob[min_prob] += topk_probs[min_prob, next_index[min_prob]]
        next_index[min_prob] += 1
        
    passwords = [list(i.cpu().numpy()) for i in passwords]
    passwords.sort()
    passwords = list(passwords for passwords,_ in itertools.groupby(passwords))
    passwords = [torch.tensor(i) for i in passwords]

    return passwords

def topk_password_correct(true, logits, num_pass=3):
    logits = logits.cpu()
    possible_passwords = topk_passwords(logits, num_pass)
    for passw in possible_passwords:
        end_index = (true == 2).nonzero(as_tuple=False)[0]
        correct_characters = (true[:end_index+1] == passw[:end_index+1]).sum()
        if correct_characters == end_index+1:
            return True
        
def conditional_password_guessing(true, logits, masked, num_candidates=10e5):
    # Naive implementation. Same number of candidates per character. TODO: use probabilities to get better estimates
    true = true.cuda()
    logits = logits.cuda()
    masked = masked.cuda()
    
    masked_chars = (masked==4).sum()
    candidates_per_char = min(math.floor(num_candidates/masked_chars), 210)
    
    # Find masked chars
    masked_idx = torch.nonzero(masked==4).flatten()
    true_masked_chars = true[masked_idx]
    
    # Find logits for masked chars
    topk_values, topk_idxs = torch.topk(input=logits, k=candidates_per_char, dim=1, largest=True, sorted=True) #logits.topk(candidates_per_char, 0, True, True)
    topk_values_masked = topk_idxs[masked_idx, :]
    
    # Check intersection 
    true_masked_chars = true_masked_chars.reshape(-1, 1).repeat(1, topk_values_masked.shape[1])
    intersection = (true_masked_chars == topk_values_masked)
    
    true = true.cpu()
    logits = logits.cpu()
    masked = masked.cpu()
    
    if intersection.sum() == len(masked_idx):
        return True
    return False

@dataclass
class EvalDataCollator(DataCollatorForLanguageModeling):
    """
    Required to create a custom special token mask which does not remove end of password token
    """
    def torch_mask_tokens(self, inputs, special_tokens_mask):
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        import torch

        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 100% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        return inputs, labels
    
    def torch_call(self, examples):
        # Handle dict or lists with proper padding and conversion to tensor.
        if isinstance(examples[0], (dict, BatchEncoding)):
            batch = self.tokenizer.pad(examples, return_tensors="pt", pad_to_multiple_of=self.pad_to_multiple_of)
        else:
            batch = {
                "input_ids": _torch_collate_batch(examples, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)
            }
            
        batch["clean"] = torch.clone(batch["input_ids"])

        special_tokens_mask = batch.pop("special_tokens_mask", None) # Remove if given
        
        # Create custom special tokens mask
        special_tokens_mask = torch.where((batch['input_ids'] != self.tokenizer.pad_token_id) & (batch['input_ids'] != self.tokenizer.eos_token_id) & (batch['input_ids'] != self.tokenizer.bos_token_id), 0, 1)
        
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