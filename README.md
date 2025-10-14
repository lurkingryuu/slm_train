# Mini Offline LLM
This is a **tiny, completely offline language model** built only with PyTorch and Transformers.

## Requirements
```bash
pip install torch transformers tokenizers
````

## Train

```bash
python train_local_llm.py
```

## Inference

```bash
python run_inference.py
```

You can replace `data.txt` with any local corpus (e.g., your company docs) to retrain it.

