import argparse
import sys
import random
import os
from pathlib import Path
from tokenizers import ByteLevelBPETokenizer, trainers
from pathlib import Path

#### It is strongly recommended to use this script with a training set including unique passwords, not frequencies.

class PassTokenizer(ByteLevelBPETokenizer):
    """ByteLevelBPETokenizer
    Represents a Byte-level BPE as introduced by OpenAI with their GPT-2 model
    """

    def train_from_iterator(
        self,
        iterator,
        vocab_size: int = 30000,
        min_frequency: int = 2,
        show_progress: bool = True,
        special_tokens = [],
        length = None,
    ):
        """Train the model using the given iterator"""

        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            show_progress=show_progress,
            special_tokens=special_tokens,
            initial_alphabet=[],
        )
        self._tokenizer.train_from_iterator(
            iterator,
            trainer=trainer,
            length=length,
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", help="path to the training dataset", type=str, required=True)
    parser.add_argument("--output_path", help="path where to store the tokenizer files", type=str, required=True)
    args = parser.parse_args()
    
    print("===> Reading passwords")
    
    with open(args.train_path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()

    # Filter printable passwords
    ascii_printable = []
    for p in lines:
        if all(32 < ord(c) < 128 for c in p):
            ascii_printable.append(p)
        
    # Log information about your data
    all_chars = ''.join(ascii_printable)  # concatenate all strings into a single string
    unique_chars = set(all_chars)
    count = len(unique_chars)
    print(f"The number of distinct letters in all strings is {count}")

    # Customize training
    special_tokens=[
        "<s>",
        "<pad>",
        "</s>", # This will be used to indicate end of password
        "<unk>",
        "<mask>",
    ]
    
    # Create BPE tokenizer
    print("===> Training tokenizer")
    tokenizer = PassTokenizer()
    
    # Customize training
    tokenizer.train_from_iterator(ascii_printable, vocab_size=count+len(special_tokens), min_frequency=1, special_tokens=special_tokens)

    print("===> Tokenizer trained with vocabulary")
    vocab = tokenizer.get_vocab()
    print(sorted(vocab, key=lambda x: vocab[x]))
    
    Path(os.path.join(args.output_path, f"byte_bpe_tokenizer_{count+len(special_tokens)}")).mkdir(parents=True, exist_ok=True)
       
    # Export
    tokenizer.save_model(os.path.join(args.output_path, f"byte_bpe_tokenizer_{count+len(special_tokens)}"))
    print("===> Tokenizer exported succesfully")
    
    
        
    