# PassGPT: Password Modeling and (Guided) Generation with LLMs

_Javier Rando, Fernando Perez-Cruz and Briland Hitaj_

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
Coming soon, we will provide access to the checkpoint of our curated models.

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
