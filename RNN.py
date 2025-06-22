# rnn_path_generator.py

import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from collections import defaultdict, Counter
import numpy as np


# ---------- 数据预处理函数 ----------
def preprocess_for_rnn(csv_path, path_output='paths_rnn.json', vocab_output='vocab_rnn.json'):
    df = pd.read_csv(csv_path, parse_dates=['stfsj'])
    df = df.dropna(subset=['zyid', 'stfsj', 'label'])
    df['date'] = df['stfsj'].dt.date

    grouped = df.sort_values(by=['zyid', 'stfsj']).groupby(['zyid', 'date'])
    paths_by_day = defaultdict(list)
    label_counter = Counter()

    for (zyid, date), group in grouped:
        items = group['label'].tolist()
        paths_by_day[zyid].append(items)
        label_counter.update(items)

    all_paths = list(paths_by_day.values())

    with open(path_output, 'w', encoding='utf-8') as f:
        json.dump(all_paths, f, ensure_ascii=False, indent=2)

    vocab = {"<PAD>": 0, "<BOS>": 1, "<EOS>": 2, "<DAY>": 3}
    for i, label in enumerate(sorted(label_counter), start=4):
        vocab[label] = i

    with open(vocab_output, 'w', encoding='utf-8') as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)


# ---------- 数据集定义 ----------
class PathDatasetRNN(Dataset):
    def __init__(self, paths, vocab, max_len=512):
        self.data = [self.encode(p, vocab)[:max_len] for p in paths]
        self.max_len = max_len

    def encode(self, path_by_day, vocab):
        tokens = [vocab['<BOS>']]
        for day_items in path_by_day:
            tokens.extend([vocab.get(it, vocab['<PAD>']) for it in day_items])
            tokens.append(vocab['<DAY>'])
        tokens.append(vocab['<EOS>'])
        return tokens

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_ids = self.data[idx]
        target_ids = input_ids[1:] + [0]
        pad_len = self.max_len - len(input_ids)
        input_ids += [0] * pad_len
        target_ids += [0] * pad_len
        return torch.tensor(input_ids), torch.tensor(target_ids)


# ---------- 简单RNN模型 ----------
class RNNPathModel(nn.Module):
    def __init__(self, vocab_size, emb_dim=128, hidden_dim=256):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.rnn = nn.GRU(emb_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.emb(x)
        out, _ = self.rnn(x)
        logits = self.fc(out)
        return logits


# ---------- 训练函数 ----------
def train_rnn_model(csv_file):
    preprocess_for_rnn(csv_file)

    with open("paths_rnn.json", "r", encoding="utf-8") as f:
        paths = json.load(f)
    with open("vocab_rnn.json", "r", encoding="utf-8") as f:
        vocab = json.load(f)
    inv_vocab = {v: k for k, v in vocab.items()}

    dataset = PathDatasetRNN(paths, vocab)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    model = RNNPathModel(vocab_size=len(vocab))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    for epoch in range(50):
        model.train()
        total_loss = 0
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}: Loss = {total_loss/len(dataloader):.4f}")

    return model, vocab, inv_vocab


# ---------- 生成路径 ----------
def generate_rnn_path(model, vocab, inv_vocab, max_len=200):
    model.eval()
    input_id = torch.tensor([[vocab['<BOS>']]]).to(next(model.parameters()).device)
    result = []
    with torch.no_grad():
        for _ in range(max_len):
            logits = model(input_id)
            next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(0)
            token = next_token.item()
            if token == vocab['<EOS>']:
                break
            result.append(token)
            input_id = torch.cat([input_id, next_token], dim=1)

    items = [inv_vocab.get(t, '<UNK>') for t in result]
    days, current = [], []
    for it in items:
        if it == '<DAY>':
            if current:
                days.append(current)
                current = []
        else:
            current.append(it)
    if current:
        days.append(current)
    for i, d in enumerate(days):
        print(f"第{i+1}天: {'、'.join(d)}")


# ---------- 运行入口 ----------
if __name__ == "__main__":
    model, vocab, inv_vocab = train_rnn_model("/home/vipuser/桌面/js/diagnosis.csv")
    print("\n💡 使用 RNN 生成一条路径:")
    generate_rnn_path(model, vocab, inv_vocab)
