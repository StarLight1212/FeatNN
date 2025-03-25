"""
@author: ansehen
@Date: 2022.3.18
@Data clean
"""
import numpy as np

# 初始化警告列表和集合，以及过滤后的数据列表
warning_lst = []
warning_set = set()
fa_lst, fb_lst, anb_lst, bnb_lst, nbs_mat_lst, \
seq_lst, aff_lst, chain_lst, pid_lst = [], [], [], [], [], [], [], [], []

# 加载数据
datapack = np.load('../../data/comp_graph/comp_prot_ec50.npy', allow_pickle=True)
fa, fb, anb, bnb, nbs_mat, seq, affinity, chain, pid = datapack

# 构建结构文件路径
path = r'../../data/DT/'
idx = [i for i in range(len(fa))]
structure_path = [path + single + '.npy' for single in pid]
print("Original length: ", len(fa))

# 数据过滤循环
for i, struct_info in enumerate(structure_path):
    try:
        ddm = np.load(struct_info, allow_pickle=True).item()["distance_matrix"][chain[i]]
        fa_lst.append(fa[i])
        fb_lst.append(fb[i])
        anb_lst.append(anb[i])
        bnb_lst.append(bnb[i])
        nbs_mat_lst.append(nbs_mat[i])
        seq_lst.append(seq[i])
        aff_lst.append(float(affinity[i]))
        chain_lst.append(chain[i])
        pid_lst.append(pid[i])
    except KeyError as KE:
        warning_lst.append(struct_info)
        warning_set.add(struct_info)
        continue

# 设置随机种子以确保可重现性
np.random.seed(42)

# 获取过滤后数据的样本数
N = len(fa_lst)

# 生成并打乱索引
indices = np.arange(N)
np.random.shuffle(indices)

# 计算训练集和测试集的划分点
split = int(0.85 * N)
train_indices = indices[:split]
test_indices = indices[split:]

# 使用索引创建训练集和测试集
fa_train = [fa_lst[i] for i in train_indices]
fa_test = [fa_lst[i] for i in test_indices]
fb_train = [fb_lst[i] for i in train_indices]
fb_test = [fb_lst[i] for i in test_indices]
anb_train = [anb_lst[i] for i in train_indices]
anb_test = [anb_lst[i] for i in test_indices]
bnb_train = [bnb_lst[i] for i in train_indices]
bnb_test = [bnb_lst[i] for i in test_indices]
nbs_mat_train = [nbs_mat_lst[i] for i in train_indices]
nbs_mat_test = [nbs_mat_lst[i] for i in test_indices]
seq_train = [seq_lst[i] for i in train_indices]
seq_test = [seq_lst[i] for i in test_indices]
aff_train = [aff_lst[i] for i in train_indices]
aff_test = [aff_lst[i] for i in test_indices]
chain_train = [chain_lst[i] for i in train_indices]
chain_test = [chain_lst[i] for i in test_indices]
pid_train = [pid_lst[i] for i in train_indices]
pid_test = [pid_lst[i] for i in test_indices]

# 将训练集和测试集转换为 NumPy 数组并打包
train_data_pack = [
    np.array(fa_train),
    np.array(fb_train),
    np.array(anb_train),
    np.array(bnb_train),
    np.array(nbs_mat_train),
    np.array(seq_train),
    np.array(aff_train),
    np.array(chain_train),
    np.array(pid_train)
]

test_data_pack = [
    np.array(fa_test),
    np.array(fb_test),
    np.array(anb_test),
    np.array(bnb_test),
    np.array(nbs_mat_test),
    np.array(seq_test),
    np.array(aff_test),
    np.array(chain_test),
    np.array(pid_test)
]

# 分别保存训练集和测试集
np.save("../../data/datapack/comp_prot_ec50_train.npy", train_data_pack)
np.save("../../data/datapack/comp_prot_ec50_test.npy", test_data_pack)

# 输出信息
print("Total valid samples:", N)
print("Training set size:", len(train_indices))
print("Testing set size:", len(test_indices))
print("Original length: ", len(fa))
print(warning_set)
print(len(warning_set))
print(len(warning_lst))