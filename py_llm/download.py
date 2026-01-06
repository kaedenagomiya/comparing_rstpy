import os
import sys
from huggingface_hub import snapshot_download

model_name = "Qwen-Coder"

if model_name == "Qwen-Coder":
    # https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct
    model_id = "Qwen/Qwen2.5-Coder-7B-Instruct"
elif model_name == "ELYZA-CodeLlama":
    # https://huggingface.co/elyza/ELYZA-japanese-CodeLlama-7b/tree/main
    model_id = "elyza/ELYZA-japanese-CodeLlama-7b"

revision = "main" 

snapshot_download(
        repo_id=model_id,
        local_dir=model_id,
        local_dir_use_symlinks=False,
        revision=revision)
