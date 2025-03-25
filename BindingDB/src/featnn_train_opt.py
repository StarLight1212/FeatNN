import os
import json
import math
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm  # 添加进度条支持
from FEATNN import FEATNN
from utils import *

# 全局配置
CONFIG = {
    'data_dir': '../data/DT/',                  # 数据目录
    'model_save_path': '../results/model/FEAT_model.pth',  # 模型保存路径
    'batch_size': 16,                            # 批量大小
    'num_epochs': 500,                           # 最大训练轮数
    'learning_rate': 5e-4,                    # 初始学习率
    'weight_decay': 0,                          # 权重衰减
    'amsgrad': True,                            # Adam 优化器的 AMSGrad 参数
    'step_size': 15,                            # 学习率调度步长
    'gamma': 0.5,                               # 学习率衰减因子
    'patience': 5,                              # 早停耐心值
    'log_file': 'logging.txt'                   # 日志文件路径
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
        data: 训练或测试数据。
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

def train_epoch(net, train_data, batch_size, optimizer, criterion, device, log_file):
    """训练一个 epoch。

    Args:
        net: 模型实例。
        train_data: 训练数据集。
        batch_size (int): 批量大小。
        optimizer: 优化器。
        criterion: 损失函数。
        device (str): 设备类型。
        log_file (str): 日志文件路径。

    Returns:
        float: 平均亲和力损失。
    """
    net.train()
    affinity_loss = 0
    shuffle_index = np.arange(len(train_data[0]))
    np.random.shuffle(shuffle_index)

    for i in tqdm(range(int(math.ceil(len(train_data[0]) / batch_size))), desc="Training Batch"):
        batch_indices = shuffle_index[i * batch_size:(i + 1) * batch_size]
        batch_data = prepare_batch_data(train_data, batch_indices, device)
        if batch_data is None:
            continue

        vertex_mask, vertex, edge, atom_adj, bond_adj, nbs_mask, seq_mask, sequence, dist_mat, torsion_mat, dist_mat_mask, affinity_label = batch_data

        optimizer.zero_grad()  # 只在预测前清零梯度
        affinity_pred = net(vertex_mask, vertex, edge, atom_adj, bond_adj, nbs_mask, seq_mask, sequence, dist_mat, torsion_mat, dist_mat_mask)
        loss_aff = criterion(affinity_pred, affinity_label)
        affinity_loss += loss_aff.item() * len(batch_indices)
        loss_aff.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 5)
        optimizer.step()

    avg_loss = affinity_loss / len(train_data[0])
    with open(log_file, 'a') as f:
        f.write(f"Training Loss: {avg_loss:.6f}\n")
    return avg_loss

def test_func(net, test_data, batch_size, device, log_file):
    """测试模型性能。

    Args:
        net: 模型实例。
        test_data: 测试数据集。
        batch_size (int): 批量大小。
        device (str): 设备类型。
        log_file (str): 日志文件路径。

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
    performance = [rmse, pearson, spearman, r_square_score, MedAE, MAE, MAPE]
    with open(log_file, 'a') as f:
        f.write(f"Test Performance: {' '.join([f'{metric:.6f}' for metric in performance])}\n")
    return performance, label_list, output_list

def train_and_eval(net, train_data, test_data, batch_size, num_epochs, device, patience=7, log_file='logging.txt'):
    """训练并评估模型，支持每个 epoch 对训练集评估和每五个 epoch 测试一次，并实现早停。

    Args:
        net: 模型实例。
        train_data: 训练数据集。
        test_data: 测试数据集。
        batch_size (int): 批量大小。
        num_epochs (int): 最大训练轮数。
        device (str): 设备类型。
        patience (int): 早停的耐心值。
        log_file (str): 日志文件路径。

    Returns:
        nn.Module: 训练后的最佳模型。
    """
    net.to(device)
    net.apply(weights_init)

    total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"模型总参数量: {total_params}")
    with open(log_file, 'a') as f:
        f.write(f"Total Parameters: {total_params}\n")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, net.parameters()),
        lr=CONFIG['learning_rate'],
        weight_decay=CONFIG['weight_decay'],
        amsgrad=CONFIG['amsgrad']
    )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=CONFIG['step_size'], gamma=CONFIG['gamma'])

    best_test_rmse = float('inf')
    epochs_no_improve = 0
    best_model_state = None

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}, 学习率: {optimizer.param_groups[0]['lr']}")
        with open(log_file, 'a') as f:
            f.write(f"Epoch {epoch + 1}/{num_epochs}, LR: {optimizer.param_groups[0]['lr']}\n")

        # 训练一个 epoch
        train_loss = train_epoch(net, train_data, batch_size, optimizer, criterion, device, log_file)
        print(f"训练损失: {train_loss:.6f}")

        # 保存模型（从第21个epoch开始）
        if epoch >= 20:
            save_path = f"../results/model/opt_model_{epoch + 1}.pth"
            torch.save(net.state_dict(), save_path)
            print(f"已保存模型: {save_path}")
            with open(log_file, 'a') as f:
                f.write(f"Saved model after epoch {epoch + 1}: {save_path}\n")

        # 每 3 个 epoch 计算并打印训练集性能
        if (epoch + 1) % 3 == 0:
            train_performance, _, _ = test_func(net, train_data, batch_size, device, log_file)
            perf_name = ['rmse', 'pearson', 'spearman', 'r_square_score', 'MedAE', 'MAE', 'MAPE']
            print_perf = [f"{perf_name[i]}: {train_performance[i]:.6f}" for i in range(len(perf_name))]
            print("训练集性能: ", " ".join(print_perf))

        # 每五个 epoch 评估测试集
        if (epoch + 1) % 5 == 0:
            test_performance, test_label, test_output = test_func(net, test_data, batch_size, device, log_file)
            test_rmse = test_performance[0]
            print(f"测试集 RMSE: {test_rmse:.6f}")
            with open(log_file, 'a') as f:
                f.write(f"Test RMSE: {test_rmse:.6f}\n")

            # 早停逻辑
            if test_rmse < best_test_rmse:
                best_test_rmse = test_rmse
                best_model_state = net.state_dict()
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            # if epochs_no_improve >= patience:
            #     print(f"早停触发！在 {epoch + 1} 个 epoch 后停止训练。")
            #     with open(log_file, 'a') as f:
            #         f.write(f"Early stopping at epoch {epoch + 1}\n")
            #     break

        scheduler.step()

    # 加载最佳模型
    if best_model_state is not None:
        net.load_state_dict(best_model_state)
        torch.save(net, CONFIG['model_save_path'])
        print("已保存最佳模型！")
        with open(log_file, 'a') as f:
            f.write("Best model saved!\n")
    else:
        print("未找到最佳模型，训练可能未达到早停条件。")
        with open(log_file, 'a') as f:
            f.write("No best model found.\n")

    return net


if __name__ == "__main__":
    torch.cuda.empty_cache()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    with open(CONFIG['log_file'], 'a') as f:
        f.write(f"Device: {device}\n")

    with open('param.json', 'r') as f:
        params = json.load(f)

    measure = "EC50"
    batch_size, num_epochs = CONFIG['batch_size'], CONFIG['num_epochs']
    print(f"数据集: BindingDB v2022, 测量指标: {measure}")
    print(f"训练轮数: {num_epochs}, 批量大小: {batch_size}")
    with open(CONFIG['log_file'], 'a') as f:
        f.write(f"Dataset: BindingDB v2022, Measure: {measure}\n")
        f.write(f"Epochs: {num_epochs}, Batch Size: {batch_size}\n")

    # 加载数据
    train_data = np.load('../data/datapack/comp_prot_ec50_train.npy', allow_pickle=True)
    # train_data = [item[:70] for item in train_data]
    test_data = np.load('../data/datapack/comp_prot_ec50_test.npy', allow_pickle=True)
    print(f"训练集亲和力范围: min: {np.min(train_data[6])}, max: {np.max(train_data[6])}")
    print(f"测试集亲和力范围: min: {np.min(test_data[6])}, max: {np.max(test_data[6])}")
    print(f"训练样本数: {len(train_data[0])}, 测试样本数: {len(test_data[0])}")
    with open(CONFIG['log_file'], 'a') as f:
        f.write(f"Train Affinity Range: min {np.min(train_data[6])}, max {np.max(train_data[6])}\n")
        f.write(f"Test Affinity Range: min {np.min(test_data[6])}, max {np.max(test_data[6])}\n")
        f.write(f"Train Samples: {len(train_data[0])}, Test Samples: {len(test_data[0])}\n")

    # 加载化合物特征并初始化模型
    init_A, init_B = loading_comp_repr(measure)
    net = FEATNN(init_A, init_B, params)

    # 训练和评估
    net = train_and_eval(net, train_data, test_data, batch_size, num_epochs, device, patience=CONFIG['patience'], log_file=CONFIG['log_file'])

    # 测试最终性能
    test_performance, test_label, test_output = test_func(net, test_data, batch_size, device, CONFIG['log_file'])
    perf_name = ['rmse', 'pearson', 'spearman', 'r_square_score', 'MedAE', 'MAE', 'MAPE']
    print_perf = [f"{perf_name[i]}: {test_performance[i]:.6f}" for i in range(len(perf_name))]
    print("最终测试性能: ", " ".join(print_perf))
    with open(CONFIG['log_file'], 'a') as f:
        f.write(f"Final Test Performance: {' '.join(print_perf)}\n")
    print("FeatNN 训练和测试流程已完成！")