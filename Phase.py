#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 31 22:13:58 2025

@author: root
"""

# gpt_path_pipeline.py

import pandas as pd
import json
from collections import Counter
from transformers import GPT2Config, GPT2LMHeadModel, Trainer, TrainingArguments
from torch.optim import AdamW
from torch.utils.data import Dataset
import torch
import os
from statistics import mean
import numpy as np
import torch.nn as nn


# ---------- Step 1: Data Preprocessing (Flat, Unique Items with <PHASE> and <DAY> tags) ----------
def preprocess_for_gpt(csv_path, path_output='paths.json', vocab_output='vocab.json'):
    df = pd.read_csv(csv_path, parse_dates=['stfsj'])
    df = df.dropna(subset=['zyid', 'stfsj', 'label', 'period'])

    df = df.sort_values(by=['zyid', 'stfsj'])
    grouped = df.groupby('zyid')
    paths = []
    label_counter = Counter()
    day_lengths = []

    for _, group in grouped:
        group = group.sort_values(by='stfsj')
        group['date'] = group['stfsj'].dt.date

        flat_items = []
        current_phase = None
        current_day = None

        for _, row in group.iterrows():
            phase = row['period']
            label = row['label']
            date = row['date']

            if date != current_day:
                flat_items.append('<day>')
                current_day = date
                label_counter.update(['<day>'])

            if phase != current_phase:
                flat_items.append(f'<{phase}>')
                current_phase = phase
                label_counter.update([f'<{phase}>'])

            flat_items.append(label)
            label_counter.update([label])

        flat_items.append('<EOS>')
        paths.append(flat_items)
        day_lengths.append(len(flat_items))

    with open(path_output, 'w', encoding='utf-8') as f:
        json.dump(paths, f, ensure_ascii=False, indent=2)

    vocab = {"<PAD>": 0, "<BOS>": 1, "<EOS>": 2, "<SEP>": 3}
    for i, label in enumerate(sorted(label_counter), start=4):
        vocab[label] = i

    with open(vocab_output, 'w', encoding='utf-8') as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

    with open("day_length_stats.json", "w", encoding="utf-8") as f:
        json.dump(day_lengths, f)

    print(f"✅ 已生成 {len(paths)} 条路径")
    print(f"✅ 词表大小：{len(vocab)} 项")


# ---------- Step 2: Dataset Preparation ----------
def encode_path(path, vocab):
    tokens = [vocab['<BOS>']] + [vocab.get(it, vocab['<PAD>']) for it in path] + [vocab['<EOS>']]
    return tokens

def decode_ids(token_ids, inv_vocab):
    return [inv_vocab.get(str(i), '<UNK>') for i in token_ids if i != 0]

def decode_by_phases_and_days(token_ids, inv_vocab):
    tokens = decode_ids(token_ids, inv_vocab)
    result = []
    phase = None
    day = []
    days_in_phase = []

    for token in tokens:
        if token == '<BOS>' or token == '<PAD>':
            continue
        elif token == '<EOS>':
            if day:
                days_in_phase.append(day)
            if phase:
                result.append((phase, days_in_phase))
            break
        elif token.startswith('<') and token.endswith('>'):
            if token == '<day>':
                if day:
                    days_in_phase.append(day)
                    day = []
            else:
                if day:
                    days_in_phase.append(day)
                    day = []
                if phase and days_in_phase:
                    result.append((phase, days_in_phase))
                phase = token
                days_in_phase = []
        else:
            day.append(token)

    return result

class PathDataset(Dataset):
    def __init__(self, encoded_paths, max_length=1024):
        self.data = [p[:max_length] for p in encoded_paths]
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_ids = self.data[idx]
        attention_mask = [1] * len(input_ids)
        padding = [0] * (self.max_length - len(input_ids))

        return {
            "input_ids": torch.tensor(input_ids + padding),
            "attention_mask": torch.tensor(attention_mask + padding),
            "labels": torch.tensor(input_ids + padding)
        }


# ---------- Step 3: Build GPT Model ----------
class GPT2WithStructureLoss(GPT2LMHeadModel):
    def __init__(self, config):
        super().__init__(config)

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        outputs = super().forward(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        return (loss, outputs.logits)

def build_model(vocab_size):
    config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=1024,
        n_ctx=1024,
        n_embd=256,
        n_layer=6,
        n_head=4
    )
    return GPT2WithStructureLoss(config)


def generate_path_with_phases(model, start_token_id, max_length=512):
    #model.eval()
    input_ids = torch.tensor([[start_token_id]]).to(model.device)

    for _ in range(max_length):
        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs[1][0, -1, :]
            next_token_id = torch.argmax(logits).item()
            input_ids = torch.cat((input_ids, torch.tensor([[next_token_id]]).to(model.device)), dim=1)
            if next_token_id == 2:  # <EOS>
                break

    return input_ids[0].tolist()


# ---------- Step 4: Evaluation ----------
def get_item_frequencies(paths):
    item_counter = Counter()
    for path in paths:
        item_counter.update([p for p in path if not p.startswith('<')])
    total = sum(item_counter.values())
    item_freq = {k: v / total for k, v in item_counter.items()}
    return item_freq, item_counter

def evaluate_generated_path(generated_path, item_freq, item_counter, high_thresh=0.8, low_thresh=0.2):
    sorted_items = sorted(item_freq.items(), key=lambda x: x[1], reverse=True)
    total_items = len(sorted_items)
    top_n = int(total_items * (1 - high_thresh))
    bottom_n = int(total_items * low_thresh)

    high_freq_set = set([k for k, _ in sorted_items[:top_n]])
    low_freq_set = set([k for k, _ in sorted_items[-bottom_n:]])

    all_items = [item for item in generated_path if not item.startswith('<')]
    if not all_items:
        return {}

    high_hits = sum(1 for item in all_items if item in high_freq_set)
    low_hits = sum(1 for item in all_items if item in low_freq_set)
    avg_freq = sum(item_freq.get(item, 0) for item in all_items) / len(all_items)

    # 新增：识别未出现的高频项目
    generated_set = set(all_items)
    missed_high_freq = sorted([item for item in high_freq_set if item not in generated_set],
                              key=lambda x: item_counter[x], reverse=True)

    return {
        "生成路径项目数": len(all_items),
        "高频覆盖率": round(high_hits / len(all_items), 3),
        "低频比例": round(low_hits / len(all_items), 3),
        "平均频率": round(avg_freq, 4),
        "未包含的高频项目（按频率降序）": missed_high_freq
    }


# ---------- Step 5: Main Pipeline ----------
from sklearn.model_selection import train_test_split


csv_file='./data/raw/diagnosis.csv'
preprocess_for_gpt(csv_file)

with open("paths.json", "r", encoding="utf-8") as f:
    paths = json.load(f)
with open("vocab.json", "r", encoding="utf-8") as f:
    vocab = json.load(f)
inv_vocab = {str(v): k for k, v in vocab.items()}

encoded = [encode_path(p, vocab) for p in paths]
dataset = PathDataset(encoded)

train_data,eval_data = train_test_split(dataset,test_size=0.2,random_state=42)

# 创建 DataLoader
train_loader = DataLoader(train_data, batch_size=5, shuffle=True)
eval_loader = DataLoader(eval_data, batch_size=5)

# 构建模型配置
config = GPT2Config(
    vocab_size=len(vocab),
    n_positions=1024,
    n_ctx=1024,
    n_embd=256,
    n_layer=6,
    n_head=4
)

model = GPT2LMHeadModel(config)

# 优化器
optimizer = AdamW(model.parameters(), lr=5e-5)
# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 早停参数
best_loss = float("inf")
patience = 3
no_improve_epochs = 0
num_epochs = 50

# 训练循环
for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0
    for batch in train_loader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_train_loss += loss.item()
    # 验证循环
    model.eval()
    eval_losses = []
    with torch.no_grad():
        for batch in eval_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, labels=labels)
            eval_losses.append(outputs.loss.item())

    avg_eval_loss = np.mean(eval_losses)
    print(f"Epoch {epoch+1}: Train Loss: {total_train_loss:.4f}, Eval Loss: {avg_eval_loss:.4f}")

    # Early stopping 判断
    if avg_eval_loss < best_loss:
        best_loss = avg_eval_loss
        no_improve_epochs = 0
        torch.save(model.state_dict(), "best_model.pt")  # 保存最佳模型
    else:
        no_improve_epochs += 1
        if no_improve_epochs >= patience:
            print("Early stopping triggered.")
            break
# 
# 早停参数


#

print("\n💡 生成一条典型路径：")
model.eval()
generated_ids = generate_path_with_phases(
    model, start_token_id=vocab['<BOS>'], max_length=512
)
phases = decode_by_phases_and_days(generated_ids, inv_vocab)
for i, (phase_tag, days) in enumerate(phases, 1):
    print(f"{phase_tag}：")
    for d, items in enumerate(days, 1):
        print(f"  第{d}天：{', '.join(items)}")

with open("generated_path.txt", "w", encoding="utf-8") as f:
    for i, (phase_tag, days) in enumerate(phases, 1):
        f.write(f"{phase_tag}：\n")
        for d, items in enumerate(days, 1):
            f.write(f"  第{d}天：{', '.join(items)}\n")

print("\n📊 评估生成路径：")
item_freq, item_counter = get_item_frequencies(paths)
flat_items = [item for _, days in phases for day in days for item in day if not item.startswith('<')]
eval_result = evaluate_generated_path(flat_items, item_freq, item_counter)
with open("evaluation_result.txt", "w", encoding="utf-8") as f:
    for k, v in eval_result.items():
        if isinstance(v, list):
            f.write(f"{k}：\n")
            for item in v:
                f.write(f"  {item}\n")
        else:
            print(f"{k}：{v}")
            f.write(f"{k}：{v}\n")


