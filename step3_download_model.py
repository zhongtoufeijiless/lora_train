# step3_download_model.py: 下载LLaMA3基础大模型
from modelscope import snapshot_download
model_dir = snapshot_download('LLM-Research/Meta-Llama-3-8B-Instruct', cache_dir='/root/autodl-tmp', revision='master')
print(f"模型已下载到: {model_dir}") 