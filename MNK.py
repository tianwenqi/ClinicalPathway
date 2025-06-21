import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import Counter

# Step 1: Load and preprocess data
df = pd.read_csv("/home/vipuser/桌面/SPF/diagnosis.csv", parse_dates=["stfsj"])

df['stfsj'] = pd.to_datetime(df['stfsj'])
df = df.sort_values(by=['zyid', 'stfsj'])

# Step 2: Convert each zyid into a sequence of labels
grouped = df.groupby('zyid')['label'].apply(list)
sequences = grouped.tolist()
ids = grouped.index.tolist()

# Step 3: Modified Needleman-Wunsch distance function
def nw_distance(seq1, seq2, match=0, mismatch=1, gap=1):
    m, n = len(seq1), len(seq2)
    dp = np.zeros((m+1, n+1))
    for i in range(m+1):
        dp[i][0] = i * gap
    for j in range(n+1):
        dp[0][j] = j * gap
    for i in range(1, m+1):
        for j in range(1, n+1):
            if seq1[i-1] == seq2[j-1]:
                cost = match
            else:
                cost = mismatch
            dp[i][j] = min(
                dp[i-1][j-1] + cost,
                dp[i-1][j] + gap,
                dp[i][j-1] + gap
            )
    return dp[m][n]

# Step 4: Compute distance matrix
n = len(sequences)
dist_matrix = np.zeros((n, n))
print("Calculating distance matrix...")
for i in tqdm(range(n)):
    for j in range(i+1, n):
        d = nw_distance(sequences[i], sequences[j])
        dist_matrix[i][j] = d
        dist_matrix[j][i] = d

# Step 5: Perform clustering
num_clusters = 3  # 可根据需求调整
model = AgglomerativeClustering(n_clusters=num_clusters, affinity='precomputed', linkage='average')
labels = model.fit_predict(dist_matrix)

# Step 6: Summarize each cluster by most common sequence (mode)
def get_representative_sequence(cluster_seqs):
    """从路径中找出最常见的“近似路径”"""
    # 方法：按位置取最多label，如果长度不同则补空
    max_len = max(len(seq) for seq in cluster_seqs)
    padded_seqs = [seq + [''] * (max_len - len(seq)) for seq in cluster_seqs]
    representative = []
    for pos in range(max_len):
        items = [seq[pos] for seq in padded_seqs if seq[pos] != '']
        if items:
            representative.append(Counter(items).most_common(1)[0][0])
    return representative


from sklearn_extra.cluster import KMedoids

# 使用 Needleman-Wunsch 距离矩阵进行聚类
n_clusters = 1  # 你想聚成几个类
kmedoids = KMedoids(n_clusters=n_clusters, metric='precomputed', random_state=42)
labels = kmedoids.fit_predict(dist_matrix)

# 得到每个簇的中心点（medoid），即真实存在的代表路径
medoid_indices = kmedoids.medoid_indices_
medoid_sequences = [sequences[i] for i in medoid_indices]

# 输出每个簇的代表路径
print("\n=== Clustered Clinical Pathways (Medoids) ===")
for i, medoid_seq in enumerate(medoid_sequences):
    count = sum(labels == i)
    print(f"Cluster {i+1} (n={count}):\n", " → ".join(medoid_seq), "\n")
    

out = pd.DataFrame(dist_matrix)
out.to_csv("/home/vipuser/桌面/SPF/dist_matrix.csv",index=False,header=False)





    





