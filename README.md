# PassGPT: Password Modeling and (Guided) Generation with LLMs

_Javier Rando<sup>1</sup>, Fernando Perez-Cruz<sup>1,2</sup> and Briland Hitaj<sup>3</sup>_

<sup><sup>1</sup>ETH Zurich, <sup>2</sup>Swiss Data Science Center, <sup>3</sup>SRI International</sup>

[![License for code and models](https://img.shields.io/badge/Code%20and%20Models%20License-CC%20By%20NC%204.0-yellow)](https://github.com/javirandor/passbert/blob/main/LICENSE)

-----------

This repository contains the official implementation of the PassGPT model. For details, see [the paper](https://arxiv.org/abs/2306.01545).

**Usage and License Notices**: PassGPT is intended and licensed for research use only. The model and code are CC BY NC 4.0 (allowing only non-commercial use) and should not be used outside of research purposes. This material should never be used to attack real systems.

## Overview
PassGPT is a [GPT-2 model](https://huggingface.co/docs/transformers/model_doc/gpt2) trained from scratch on password leaks.
* PassGPT outperforms existing methods based on generative adversarial networks (GAN) by guessing twice as many previously unseen passwords.
* We introduce the concept of guided password generation, where we adapt the sampling procedure to generate passwords matching arbitrary constraints, a feat lacking in current GAN-based strategies. 
* The probability that PassGPT assigns to passwords can be used to enhance existing password strength estimators.

## Dataset
In our work, we train PassGPT on the [RockYou password leak](https://wiki.skullsecurity.org/index.php/Passwords).

## Pre-trained models
* Model trained on passwords from the RockYou leak with at most 10 characters can be accessed [here](https://huggingface.co/javirandor/passgpt-10characters/). This model was used to compare with previous work in our paper.
* If you need access to our model pre-trained on passwords with up to 16 characters, please get in touch with me. Include your name, reference to previous work (e.g., Google Scholar or personal website), and a brief summary of the project where PassGPT will be used.

## Generate passwords using our pre-trained model
In order to use our model, you will need to first log into HuggingFace and accept the conditions [here](https://huggingface.co/javirandor/passgpt-10characters/). Access to the model is automatically granted. You may also need to [set up an authentication key](https://huggingface.co/docs/hub/security-tokens).

After this, you can use this simple code to generate `NUM_GENERATIONS` passwords. It can even run on CPU! To scale up your generations, you can use [`generate_passwords.py`](https://github.com/javirandor/passgpt/blob/main/src/generate_passwords.py).

```
from transformers import GPT2LMHeadModel
from transformers import RobertaTokenizerFast

tokenizer = RobertaTokenizerFast.from_pretrained("javirandor/passgpt-10characters",
                                                  use_auth_token="YOUR_ACCESS_TOKEN",
                                                  max_len=12,
                                                  padding="max_length", 
                                                  truncation=True,
                                                  do_lower_case=False,
                                                  strip_accents=False,
                                                  mask_token="<mask>",
                                                  unk_token="<unk>",
                                                  pad_token="<pad>",
                                                  truncation_side="right")

model = GPT2LMHeadModel.from_pretrained("javirandor/passgpt-10characters", use_auth_token="YOUR_ACCESS_TOKEN").eval()

NUM_GENERATIONS = 1

# Generate passwords sampling from the beginning of password token
g = model.generate(torch.tensor([[tokenizer.bos_token_id]]).cuda(),
                  do_sample=True,
                  num_return_sequences=NUM_GENERATIONS,
                  max_length=12,
                  pad_token_id=tokenizer.pad_token_id,
                  bad_words_ids=[[tokenizer.bos_token_id]])

# Remove start of sentence token
g = g[:, 1:]

decoded = tokenizer.batch_decode(g.tolist())
decoded_clean = [i.split("</s>")[0] for i in decoded] # Get content before end of password token

# Print your sampled passwords!
print(decoded_clean)
```

## Train your own model

To train your own PassGPT model on a dataset, you need to follow these steps. Optionally, you can adapt this code to finetune a pre-trained model on a different dataset.

1. Create a tokenizer for your passwords. Importantly, this tokenizer is character-level, avoiding possible concatenation of letters into one single token as usually done in NLP. This is to preserve a meaningful probability distribution under the model. Depending on your task, you might want to adapt this to be more expressive.
```
python src/create_tokenizer.py --train_path {PATH_TO_TRAINING_DATA} --output_path {PATH_TO_TOKENIZERS_FOLDER}
```
2. Customize your training configuration file and store it in `CONFIG_PATH`. You can use [this template](https://github.com/javirandor/passbert/blob/main/configs/passgpt-16chars.yaml).
3. Train PassGPT
```
python src/train_passgpt.py --config_path {CONFIG_PATH}
```
4. Generate passwords using the model you just trained
```
python src/generate_passwords.py --model_path {MODEL_PATH} --out_path {PASSWORD_OUTPUT_FOLDER} --num_generate {NUM_PASSWORDS} --train_data_path {PATH_TO_TRAINING_DATA} --eval_data_path {PATH_TO_EVAL_DATA}
```
Additionally, you can tweak further generation parameters such as `--temperature`, `--top_p` or `--top_k`

## Cite our work
```
@article{rando2023passgpt,
  title={PassGPT: Password Modeling and (Guided) Generation with Large Language Models},
  author={Rando, Javier and Perez-Cruz, Fernando and Hitaj, Briland},
  journal={arXiv preprint arXiv:2306.01545},
  year={2023}
}
```
