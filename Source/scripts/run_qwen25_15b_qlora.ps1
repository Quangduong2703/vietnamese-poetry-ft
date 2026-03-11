$env:TOKENIZERS_PARALLELISM = "false"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"

python D:\NLP\Source\scripts\train_qwen25_15b_qlora.py `
  --config D:\NLP\Source\configs\qwen25_15b_qlora.json
