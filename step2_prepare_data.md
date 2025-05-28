# step2_准备数据集

1. 本目录已自带以下数据集文件，可直接用于微调：
   - `huanhuan.json`：全量甄嬛风格对话数据
   - `huanhuan-100.json`：小样本测试用数据（建议先用它跑通流程）
   - `huanhuan.jsonl`：行分隔JSON格式数据

2. 数据集格式示例（JSON数组，每条为一个指令样本）：
```json
[
  {"instruction": "你是谁？", "input": "", "output": "家父是大理寺少卿甄远道。"},
  ...
]
```

3. 你也可以将自己的数据集整理为上述格式，替换 `./dataset/huanhuan.json` 或新建其它文件。

4. 推荐先用 `huanhuan-100.json` 跑通流程，确认无误后再用全量数据。 