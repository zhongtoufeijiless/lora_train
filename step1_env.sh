#!/bin/bash
# step1_env.sh: LoRA微调环境准备脚本

# 创建conda环境
conda create -n lora_env python=3.9 -y

# 激活环境
source $(conda info --base)/etc/profile.d/conda.sh
conda activate lora_env

# 配置国内pip源
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 安装依赖
pip install torch transformers peft datasets accelerate pandas modelscope 