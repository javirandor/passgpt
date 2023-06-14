import os
import argparse
import torch
from transformers import GPT2LMHeadModel
from datasets import load_dataset
from pathlib import Path

from transformers import RobertaTokenizerFast

import numpy as np
import random
from tqdm import trange


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", help="Path to PassGPT checkpoint or Huggingface name", type=str, required=True)
    parser.add_argument("--tokenizer_path", help="Path to pre-trained tokenizer or Huggingface name. If none, it will be set to model_path", type=str, default=None)
    parser.add_argument("--train_data_path", help="path to training data", type=str, required=False)
    parser.add_argument("--eval_data_path", help="path to evaluation data", type=str, required=False)
    parser.add_argument("--out_path", help="Path to store the generations", type=str, required=True)
    parser.add_argument("--filename", help="Filename where generations will be stored", type=str, default="passwords.txt")
    parser.add_argument("--maxchars", help="Maximum length of the passwords", type=int, default=16)
    parser.add_argument("--seed_offset", help="Random seed offset for generation. Allows to parallelize generation across different executions.", type=int, default=0)
    parser.add_argument("--num_generate", help="Number of passwords to generate", type=int, default=int(1e7))
    parser.add_argument("--batch_size", help="Batch size for generation", type=int, default=1000)
    parser.add_argument("--device", help="Device to run execution", type=str, default='cuda')
    parser.add_argument("--num_beams", help="Number of beams for sampling", type=int, default=1)
    parser.add_argument("--top_p", help="Probability for nucleus sampling", type=int, default=100)
    parser.add_argument("--top_k", help="Sample only from k tokens with highes probability", type=int, default=None)
    parser.add_argument("--temperature", help="Sampling temperature", type=float, default=1.0)

    
    args = parser.parse_args()
    
    # Init random seeds
    random.seed(args.seed_offset)
    np.random.seed(args.seed_offset)
    torch.manual_seed(args.seed_offset)
    
    # Assert sizes are divisible
    assert args.num_generate%args.batch_size==0, "Number of passwords to generate should be divisible by batch size"
    
    assert not os.path.isfile(os.path.join(args.out_path, args.filename)), "The provided output path already exists, please provide a unique path."
    Path(args.out_path).mkdir(parents=True, exist_ok=True)
    
    # Load tokenizer
    if args.tokenizer_path is None:
        args.tokenizer_path = args.model_path
    
    tokenizer = RobertaTokenizerFast.from_pretrained(args.tokenizer_path, 
                                                  max_len=args.maxchars+2,
                                                  padding="max_length", 
                                                  truncation=True,
                                                  do_lower_case=False,
                                                  strip_accents=False,
                                                  mask_token="<mask>",
                                                  unk_token="<unk>",
                                                  pad_token="<pad>",
                                                  truncation_side="right")
    
    # Load models
    model = GPT2LMHeadModel.from_pretrained(args.model_path).eval().to(args.device)
    
    # Passwords generation
    generations = []
    
    print("bos_token", tokenizer.bos_token_id)

    for i in trange(int(args.num_generate/args.batch_size)):
        # Set seed for reproducibility
        torch.manual_seed(args.seed_offset + i)

        with torch.no_grad():
            # Generate tokens sampling from the distribution of codebook indices
            g = model.generate(torch.tensor([[tokenizer.bos_token_id]]).cuda(), do_sample=True, max_length=args.maxchars+2, pad_token_id=tokenizer.pad_token_id, bad_words_ids=[[tokenizer.bos_token_id]], num_return_sequences=args.batch_size, num_beams=args.num_beams, top_p=args.top_p/100, top_k=args.top_k, temperature=args.temperature)

            # Remove start of sentence token
            g = g[:, 1:]

        decoded = tokenizer.batch_decode(g.tolist())
        decoded_clean = [i.split("</s>")[0] for i in decoded] # Get content before end of password token

        generations += decoded_clean

        del g
        del decoded
        del decoded_clean
    
    # Store passwords
    with open(os.path.join(args.out_path, args.filename), 'w') as f:
        for line in generations:
            f.write("{}\n".format(line))
            
    # Log information
    num_generated = len(generations)
    num_unique = len(set(generations))
    perc_unique = num_unique/num_generated * 100
    
    data_files = {}
    
    if args.train_data_path is not None:
        data_files["train"] = [args.train_data_path]
        
    if args.eval_data_path is not None:
        data_files["eval"] = [args.eval_data_path]
        
    if len(data_files):
        dataset = load_dataset('text', data_files=data_files)
        
        if args.train_data_path:
            train_passwords = set(dataset["train"]["text"])
            inter_with_train = len(train_passwords.intersection(set(generations)))
            
        if args.eval_data_path:        
            eval_passwords = set(dataset["eval"]["text"])
            inter_with_eval = len(eval_passwords.intersection(set(generations)))

    
    # Log details
    with open(os.path.join(args.out_path, f"log_{args.filename}.txt"), 'w') as f:
        f.write("Passwords generated using model at: {0}\n".format(args.model_path))
        f.write("Number of passwords generated: {}\n".format(num_generated))
        f.write("{} unique passwords generated => {:.2f}%\n".format(num_unique, perc_unique))
        if args.eval_data_path:
            f.write("{} passwords where found in the test set. {:.2f}% of the test set guessed.\n".format(inter_with_eval, 100*inter_with_eval/len(eval_passwords)))
        if args.train_data_path:
            f.write("{} passwords where found in the training set. {:.2f}% of the train set guessed.\n".format(inter_with_train, 100*inter_with_train/len(train_passwords)))
    
    
        
    
