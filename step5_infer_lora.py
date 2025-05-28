import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import argparse

def main():
    parser = argparse.ArgumentParser(description="LLaMA3 LoRA 推理脚本")
    parser.add_argument('--model_path', type=str, default='/root/autodl-tmp/LLM-Research/Meta-Llama-3-8B-Instruct', help='基础模型路径')
    parser.add_argument('--lora_path', type=str, default='./output/llama3_1_instruct_lora/checkpoint-699', help='LoRA权重路径')
    parser.add_argument('--prompt', type=str, default=None, help='用户输入问题')
    args = parser.parse_args()

    print(f"加载基础模型: {args.model_path}")
    model = AutoModelForCausalLM.from_pretrained(args.model_path, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True).eval()
    print(f"加载LoRA权重: {args.lora_path}")
    model = PeftModel.from_pretrained(model, model_id=args.lora_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    if args.prompt is None:
        prompt = input("请输入你的问题：")
    else:
        prompt = args.prompt

    messages = [
        {"role": "system", "content": "假设你是皇帝身边的女人--甄嬛。"},
        {"role": "user", "content": prompt}
    ]
    input_ids = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([input_ids], return_tensors="pt").to(model.device)
    with torch.no_grad():
        generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=256)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print('皇上：', prompt)
    print('嬛嬛：', response)

if __name__ == "__main__":
    main() 