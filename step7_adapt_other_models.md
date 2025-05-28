# step7_适配其它模型

- 替换 `--model_path` 为其它大模型路径（如 Qwen、ChatGLM、Baichuan 等）
- 按目标模型官方文档调整 `process_func` 的 prompt 拼接方式和特殊token
- 其它流程一致
- 如需帮助可查阅各模型官方微调文档或咨询社区 