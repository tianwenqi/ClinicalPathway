#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 11 12:23:50 2025

@author: root
"""


import pandas as pd
from prefixspan import PrefixSpan
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.preprocessing import TransactionEncoder

# 1. 读取数据
df = pd.read_csv('/home/vipuser/桌面/mn/diagnosis50.csv', parse_dates=['stfsj'])
df = df.dropna(subset=['zyid', 'label'])

# 每个zyid的label集合（去重）
itemsets = df.groupby('zyid')['label'].apply(lambda x: set(x)).tolist()

# 转为适合频繁项集算法的格式
te = TransactionEncoder()
te_ary = te.fit(itemsets).transform(itemsets)
df_itemsets = pd.DataFrame(te_ary, columns=te.columns_)

# 支持度阈值可调整
frequent_itemsets = fpgrowth(df_itemsets, min_support=0.5, use_colnames=True)
# 查看最大项集
frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))
max_len = frequent_itemsets['length'].max()
print(frequent_itemsets[frequent_itemsets['length'] == max_len])