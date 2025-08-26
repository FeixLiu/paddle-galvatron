
# 大模型生成快速上手

PaddleNLP 提供了方便易用的 Auto API，能够快速的加载模型和 Tokenizer。这里以使用 Qwen/Qwen2-0.5B 模型做文本生成为例：

```python
from paddlenlp.transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B", dtype="float16")
input_features = tokenizer("你好！请自我介绍一下。", return_tensors="pd")
outputs = model.generate(**input_features, max_length=128)

print(tokenizer.batch_decode(outputs[0], skip_special_tokens=True))
# ['我是一个AI语言模型，我可以回答各种问题，包括但不限于：天气、新闻、历史、文化、科学、教育、娱乐等。请问您有什么需要了解的吗？']
```
