# LoRA 微调快速上手指南

本指南以 LLaMA3/Chat-嬛嬛为例，简明总结 LoRA 微调的核心流程，适合新手无脑跟做。

---

## step1_环境准备
- 安装 Anaconda/Miniconda（如已安装可跳过）
- 创建并激活 Python 虚拟环境（推荐 Python 3.9+）
```bash
conda create -n lora_env python=3.9 -y
conda activate lora_env
```
- 安装依赖（推荐国内镜像）
```bash
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip install torch transformers peft datasets accelerate pandas
```

---

## step2_准备数据集
- 准备 instruction 微调格式的数据集（如 `huanhuan.json`）：
```json
[
  {"instruction": "你是谁？", "input": "", "output": "家父是大理寺少卿甄远道。"},
  ...
]
```
- 放到指定目录（如 `./dataset/huanhuan.json`）

---

## step3_下载基础大模型
- 新建 `model_download.py`：
```python
from modelscope import snapshot_download
model_dir = snapshot_download('LLM-Research/Meta-Llama-3-8B-Instruct', cache_dir='/root/autodl-tmp', revision='master')
```
- 运行下载脚本
```bash
python model_download.py
```

---

## step4_训练 LoRA
- 新建 `train_lora.py`（见 Chat-嬛嬛示例，可参数化模型/数据/输出路径）
- 启动训练：
```bash
python train_lora.py \
  --model_path /root/autodl-tmp/LLM-Research/Meta-Llama-3-8B-Instruct \
  --data_path ./dataset/huanhuan.json \
  --output_dir ./output/llama3_1_instruct_lora \
  --batch_size 4 \
  --epochs 3 \
  --save_steps 100
```
- 训练完成后，LoRA权重在 `output/llama3_1_instruct_lora/checkpoint-xxx/` 目录下。

---

## step5_推理部署
- 新建 `infer_lora.py`（见 Chat-嬛嬛示例）
- 推理命令示例：
```bash
python infer_lora.py --lora_path ./output/llama3_1_instruct_lora/checkpoint-699 --prompt "嬛嬛你怎么了，朕替你打抱不平！"
```
- 输出示例：
```
皇上： 嬛嬛你怎么了，朕替你打抱不平！
嬛嬛： ...（模型生成的甄嬛风格回复）
```

---

## step6_常见问题
- 显存不足：减小 batch size 或用 QLoRA
- 权重找不到：确认 `adapter_config.json` 路径
- prompt 格式不符：按目标模型官方文档调整

---

## step7_适配其它模型
- 替换 `--model_path` 为其它大模型路径
- 按目标模型官方文档调整 prompt 拼接方式
- 其它流程一致

---

如需 QLoRA、API/Web 部署、数据增强等进阶内容，可随时扩展本指南。 


-----------