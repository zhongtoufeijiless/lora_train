# step6_常见问题与调试

- 显存不足：减小 batch size 或用 QLoRA
- 权重找不到：确认 `adapter_config.json` 路径
- prompt 格式不符：按目标模型官方文档调整
- 推理输出不理想：检查数据集格式和prompt模板是否与模型一致
- 训练慢：可先用小数据集测试流程 