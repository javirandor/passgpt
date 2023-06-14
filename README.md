# PassGPT

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


