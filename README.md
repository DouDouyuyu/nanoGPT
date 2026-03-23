# nanoGPT

## 项目简介

![image-20260323085402279](C:\Users\81955\AppData\Roaming\Typora\typora-user-images\image-20260323085402279.png)

---

 基于 nanoGPT 改造的轻量级诗歌文本生成模型训练 / 微调框架，深度适配中文诗歌（唐诗 / 宋词 / 现代诗等）的文本特性。本项目代码简洁易定制，既适合诗歌创作爱好者快速搭建生成模型，也便于 NLP 入门学习者理解 GPT 核心原理与实战流程。 

---

## 特性

- 适配中文诗歌的字符 / 分词级训练，支持唐诗、宋词、现代诗等多种诗歌体裁
- 轻量化实现，核心训练逻辑仅数百行代码，易理解、易修改
- 支持 GPU/CPU/MPS（Apple Silicon）多设备训练，低配设备也能快速尝鲜
- 支持预训练模型微调（基于 GPT-2），快速提升诗歌生成效果

## 【 学习目标 】 

1. 掌握基于 PyTorch 构建 nanoGPT 模型网络的方法：理解模型架构、核心组件（如注意力层、Transformer 块）的实现原理与功能
2. 学会 nanoGPT 全流程训练：包括数据准备与预处理、模型配置、训练过程调优及优化技巧
3. 掌握模型推理逻辑与采样细节，实现自定义诗歌生成

## 诗词⽣成器 

### [**第⼀步**]训练数据预处理

1.在项目根目录的 `nanoGPT/data` 文件夹下创建 `poemtext` 子文件夹，将下载好的唐诗数据集 `tang_poet.txt` 放入该目录。

2.在 `poemtext` 文件夹中创建 `prepare.py` 文件，用于完成数据集的编码与切分。

3.编写数据预处理代码：

**prepare.py**文件代码：

```python
import os
import tiktoken
import numpy as np

# 读取数据
input_file_path = r'D:\learning\大模型\2\nanoGPT-master\data\poemtext\tang_poet.txt'
with open(input_file_path, 'r', encoding='utf-8') as f:
    data = f.read()
# 9:1划分训练/测试集
n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

# GPT-2 BPE编码
enc = tiktoken.get_encoding("gpt2")
train_ids = enc.encode_ordinary(train_data)
val_ids = enc.encode_ordinary(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# 保存为二进制文件（供训练用）
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__),'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__),'val.bin'))
```

 执行效果 

 运行 prepare.py`后，poemtext 文件夹下会生成 train.bin（训练集）和 val.bin（验证集）二进制文件，供后续模型训练使用 

![image-20260323090241260](C:\Users\81955\AppData\Roaming\Typora\typora-user-images\image-20260323090241260.png)

### [第⼆步**]**模型的训练

 数据预处理完成后，通过以下步骤启动训练  ：

**操作步骤**

1.进入项目根目录的 `nanoGPT/config` 文件夹，创建 `train_poemtext_char.py` 配置文件。

2.写入训练配置：

```python
out_dir = 'out-poemtext-char'
eval_interval = 250
eval_iters = 200
log_interval = 10
always_save_checkpoint = False
dataset = 'poemtext'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 256
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2
learning_rate = 1e-3
max_iters = 5000
lr_decay_iters = 5000
min_lr = 1e-4
beta2 = 0.99
warmup_iters = 100
compile = False
```

3.启动模型训练：

在项目根目录执行以下命令：

```
python train.py config/train_poemtext_char.py
```

**训练效果**

 训练过程中会实时打印损失值、验证集精度等日志，训练完成后，项目根目录会生成 out-poemtext-char 文件夹，其中 ckpt.pt 为训练好的模型权重文件（基于 5.8 万首唐诗训练得到）。 

![bb12e43310e449eeb2e84514cd3fbd4b](C:\Users\81955\AppData\Local\Temp\bb12e43310e449eeb2e84514cd3fbd4b.png)

![acc36583c8a84509b5752573356cbf46](C:\Users\81955\AppData\Local\Temp\acc36583c8a84509b5752573356cbf46.png) 

![image-20260323090640647](C:\Users\81955\AppData\Roaming\Typora\typora-user-images\image-20260323090640647.png)

### [第三步]模型的推理及采样： 

 训练完成后，通过采样脚本生成诗歌文本： 

**执行命令**

在项目根目录运行：

```python
python sample.py --out_dir=out-poemtext-char
```

**生成效果**

![fb941044dd7b4261b0d92b0bdaf37d24](C:\Users\81955\AppData\Local\Temp\fb941044dd7b4261b0d92b0bdaf37d24.png)

## 核心参数说明

| 参数分类 | 核心参数              | 诗歌推荐值 | 作用             |
| -------- | --------------------- | ---------- | ---------------- |
| 训练     | max_iters             | 5000       | 总训练步数       |
| 模型     | n_layer/n_head/n_embd | 6/6/384    | 模型大小与效果   |
| 训练     | batch_size            | 64/32/16   | 根据显存调整     |
| 优化     | learning_rate         | 1e-3       | 训练速度与稳定性 |
| 生成     | temperature           | 0.7        | 诗歌创作风格     |

