# PassGPT: Password Modeling and (Guided) Generation with LLMs
_Javier Rando, Fernando Perez-Cruz and Briland Hitaj_

This repository contains the official implementation of the PassGPT model. For details, see [the paper](https://arxiv.org/abs/2306.01545).

### Execution

1. Create a tokenizer
```
python src/create_tokenizer.py --train_path data/rockyouascii/after/train_16_unique.txt --output_path models/tokenizers/
```

2. Customize your configuration file and store it in `CONFIG_PATH`

3. Train PassGPT
```
python src/train_passgpt.py --config_path {CONFIG_PATH}
```


