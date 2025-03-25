import os
import torch
import numpy as np
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from FEATNN import FEATNN
from utils import *

# 全局配置
CONFIG = {
    'data_dir': '../data/DT/',                  # 数据目录
    'model_save_path': '../results/model/opt_model_42.pth',  # 模型保存路径
    'batch_size': 16,                            # 批量大小
}

def load_structure_data(structure_path, chain):
    """加载结构数据并返回距离矩阵和扭转矩阵。

    Args:
        structure_path (str): 结构文件路径。
        chain (str): 链标识符。

    Returns:
        tuple: (distance_matrix, torsion_matrix) 或 (None, None) 如果加载失败。
    """
    try:
        data = np.load(structure_path, allow_pickle=True).item()
        return data['distance_matrix'][chain], data['dihedral_array'][chain]
    except Exception as e:
        print(f"加载 {structure_path} 时出错: {e}")
        return None, None

def prepare_batch_data(data, indices, device):
    """准备批量数据。

    Args:
        data: 测试数据。
        indices: 批量索引（可以是 slice 或数组）。
        device: 设备类型。

    Returns:
        tuple: 处理后的批量数据，或 None 如果数据加载失败。
    """
    input_vertex, input_edge, input_atom_adj, input_bond_adj, input_num_nbs, input_seq, affinity_label, chain, structure_path = \
        [data[data_idx][indices] for data_idx in range(9)]

    structure_path = [os.path.join(CONFIG['data_dir'], single_path + '.npy') for single_path in structure_path]

    distance_mats, torsion_mats = [], []
    for path, c in zip(structure_path, chain):
        dist, tors = load_structure_data(path, c)
        if dist is None or tors is None:
            continue
        distance_mats.append(dist)
        torsion_mats.append(tors)

    if not distance_mats:
        return None

    inputs = [input_vertex, input_edge, input_atom_adj, input_bond_adj, input_num_nbs, input_seq, distance_mats, torsion_mats]
    vertex_mask, vertex, edge, atom_adj, bond_adj, nbs_mask, seq_mask, sequence, dist_mat, torsion_mat, dist_mat_mask = batch_data_process(inputs)

    affinity_label = torch.FloatTensor(np.array(affinity_label, dtype=np.float32)).to(device)
    return vertex_mask, vertex, edge, atom_adj, bond_adj, nbs_mask, seq_mask, sequence, dist_mat, torsion_mat, dist_mat_mask, affinity_label

def test_func(net, test_data, batch_size, device):
    """测试模型性能并返回预测结果。

    Args:
        net: 模型实例。
        test_data: 测试数据集。
        batch_size (int): 批量大小。
        device (str): 设备类型。

    Returns:
        tuple: (性能指标列表, 标签列表, 输出列表)。
    """
    net.eval()
    output_list, label_list = [], []

    with torch.no_grad():
        for i in tqdm(range(int(math.ceil(len(test_data[0]) / batch_size))), desc="Testing Batch"):
            batch_indices = slice(i * batch_size, (i + 1) * batch_size)
            batch_data = prepare_batch_data(test_data, batch_indices, device)
            if batch_data is None:
                continue

            vertex_mask, vertex, edge, atom_adj, bond_adj, nbs_mask, seq_mask, sequence, dist_mat, torsion_mat, dist_mat_mask, affinity_label = batch_data

            affinity_pred = net(vertex_mask, vertex, edge, atom_adj, bond_adj, nbs_mask, seq_mask, sequence, dist_mat, torsion_mat, dist_mat_mask)
            output_list.extend(affinity_pred.cpu().detach().numpy().reshape(-1).tolist())
            label_list.extend(affinity_label.cpu().detach().numpy().reshape(-1).tolist())

    output_list = np.array(output_list)
    label_list = np.array(label_list)
    rmse, pearson, spearman, r_square_score, MedAE, MAE, MAPE = reg_scores(label_list, output_list)
    return [rmse, pearson, spearman, r_square_score, MedAE, MAE, MAPE], label_list, output_list


if __name__ == "__main__":
    # 清空GPU缓存并设置设备
    torch.cuda.empty_cache()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")

    # 加载训练好的模型
    import json
    from utils import *
    measure = "EC50"
    with open('param.json', 'r') as f:
        params = json.load(f)
    init_A, init_B = loading_comp_repr(measure)
    net = FEATNN(init_A, init_B, params)
    state_dict = torch.load(CONFIG['model_save_path'])
    net.load_state_dict(state_dict)

    # net = torch.load(CONFIG['model_save_path'])
    net.to(device)
    print("模型已加载！")

    # 加载测试数据
    test_data = np.load('../data/datapack/comp_prot_ec50_test.npy', allow_pickle=True)
    print(f"测试样本数: {len(test_data[0])}")

    # 对测试数据进行预测
    batch_size = CONFIG['batch_size']
    test_performance, test_label, test_output = test_func(net, test_data, batch_size, device)
    print("预测完成！")

    # 使用Seaborn绘制散点图和直方图
    df = pd.DataFrame({'True Affinity': test_label, 'Predicted Affinity': test_output})
    g = sns.jointplot(x='True Affinity', y='Predicted Affinity', data=df, kind='reg',
                      joint_kws={'line_kws': {'color': 'red'}, 'scatter_kws': {'alpha':0.5}})

    # 在散点图上标注Pearson相关系数
    pearson_corr = test_performance[1]  # test_performance[1] 是Pearson相关系数
    g.ax_joint.text(0.1, 0.9, f'Pearson: {pearson_corr:.2f}', transform=g.ax_joint.transAxes)
    plt.savefig('../fig/ec50_results.svg')
    # 显示图形
    plt.show()
    print("可视化完成！")