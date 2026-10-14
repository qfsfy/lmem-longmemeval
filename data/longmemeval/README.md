# LongMemEval 数据集存放目录

官方数据（MIT 许可）：

- 仓库：https://github.com/xiaowu0162/LongMemEval
- HuggingFace：https://huggingface.co/datasets/xiaowu0162/LongMemEval

下载后放到本目录，例如：

```bash
# 方式一：huggingface-cli
huggingface-cli download xiaowu0162/LongMemEval --repo-type dataset --local-dir .

# 方式二：手动下载 json 放到本目录
```

预期的文件形态（官方 S/M 版）：

```json
[{
  "question_id": "...",
  "question_type": "single_session-user",
  "question": "...",
  "answer": "...",
  "question_date": "2023-05-08",
  "haystack_session_ids": [...],
  "haystack_dates": [...],
  "haystack_sessions": [[{"role": "user", "content": "..."}, ...], ...],
  "answer_session_ids": [...]
}]
```

注意：`*.json` / `*.jsonl` 已加入 .gitignore，不要提交数据。
