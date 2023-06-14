# Enable relative import in certain environments
import sys
sys.path.append("../code")

import os
import argparse
import torch
from transformers import GPT2LMHeadModel, GPT2Config
from datasets import load_dataset
from pathlib import Path

from transformers import RobertaTokenizerFast
from transformers import TrainingArguments

import numpy as np
import random

import time
from datetime import timedelta
import yaml
import shutil
from datasets import disable_caching

# Relative import from utils
from utils import *

if __name__ == "__main__":
    
    # Load config from file
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", help="path to yaml config", type=str, required=True)
    args = parser.parse_args()
    
    with open(args.config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        
    args = dotdict(config["config_args"])
    model_args = dotdict(config["model_args"])
    training_args = dotdict(config["training_args"])
    training_args["seed"] = args.seed
    
    # Init random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    assert not os.path.exists(training_args.output_dir), "The provided output path already exists, please provide a unique path."
    Path(training_args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Declare constants
    TOKENIZER_MAX_LEN = args.maxchars + 2 # Additional characters for start and end of password tokens
    
    # Load tokenizer
    print("===> Loading tokenizer")
    tokenizer = RobertaTokenizerFast.from_pretrained(args.tokenizer_path, 
                                                      max_len=TOKENIZER_MAX_LEN,
                                                      padding="max_length", 
                                                      truncation=True,
                                                      do_lower_case=False,
                                                      strip_accents=False,
                                                      mask_token="<mask>",
                                                      unk_token="<unk>",
                                                      pad_token="<pad>",
                                                      truncation_side="right")
    
    # Define dataloader
    print("===> Loading data")
    
    def preprocess_function(entries):
        """
        This function tokenizes a list of passwords. It appends the end of password token to each of them before processing.
        """
        to_tokenize = ['<s>' + p[:args.maxchars] +'</s>' for p in entries['text']]
        return tokenizer(to_tokenize, 
                         truncation=True, 
                         padding="max_length", 
                         max_length=TOKENIZER_MAX_LEN, 
                         add_special_tokens=False, 
                         return_special_tokens_mask=False)
    
    
    data_files = {'train': [args.train_data_path]}
    dataset = load_dataset('text', data_files=data_files)
    print("Dataset loaded with {} entries".format(len(dataset)))
    
    if args.subsample > 0:
        print("Subsampling dataset to {} random entries".format(args.subsample))
        dataset['train'] = dataset['train'].select([i for i in range(args.subsample)])
        
    # Process data
    print("===> Processing data")
    tokenized_datasets = dataset.map(preprocess_function, batched=True, remove_columns=dataset["train"].column_names)
    tokenized_datasets = tokenized_datasets.shuffle(seed=args.seed)
    
    # Format data
    tokenized_datasets.set_format(type="torch")
    
    print("===> Initializing model")

    config = GPT2Config(
        vocab_size=tokenizer.vocab_size,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        **model_args
    )
    
    model = GPT2LMHeadModel(config)
    print("Model initialized with {} parameters".format(sum(t.numel() for t in model.parameters())))
    
    print("===> Preparing training")
    # Define the data collator. In charge of hiding tokens to be predicted.
    data_collator = PasswordDataCollator(
        tokenizer=tokenizer, mlm=False
    )
    
    train_args = TrainingArguments(
            **training_args
        )

    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=train_args,
        train_dataset=tokenized_datasets["train"]
    )
    
    print("===> Launching training")
    start = time.time()
    trainer.train()
    end = time.time()
    
    print("===> Training completed after {}. Storing last version.".format(str(timedelta(seconds=end-start))))
    model.save_pretrained(os.path.join(training_args.output_dir, "last"))
    
    # Comment out next lines if you want to keep several checkpoints.
    print("===> Deleting previous checkpoints")
    checkpoints = [i for i in os.listdir(training_args.output_dir) if i.startswith("checkpoint")]
    for c in checkpoints: 
        shutil.rmtree(os.path.join(training_args.output_dir, c))
    
    print("===> Training finished succesfully :)")
