import math
import json
import torch.optim as optim
from utils import *
import numpy as np
import torch
from torch import nn
from FEATNN import FEATNN
import gc


# train and evaluate
def train_and_eval(net, train_data, batch_size, num_epoch):
    path_dir = r'../data/DT/'
    net.cuda()
    net.apply(weights_init)
    # Compute the total parameters
    pytorch_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print('total num params', pytorch_total_params)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=0.0005, weight_decay=0, amsgrad=True)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)

    train_result = []
    shuffle_index = np.arange(len(train_data[0]))

    for epoch in range(num_epoch):
        np.random.shuffle(shuffle_index)
        for param_group in optimizer.param_groups:
            scheduler.step()
            print('learning rate:', param_group['lr'])
        affinity_loss = 0
        # Train process start!
        net.train()
        for i in range(int(len(train_data[0]) / batch_size)):
            if i % 100 == 0:
                print('epoch', epoch, 'batch', i)

            input_vertex, input_edge, input_atom_adj, input_bond_adj, input_num_nbs, input_seq, affinity_label, chain, structure_path = \
                [train_data[data_idx][shuffle_index[i * batch_size:(i + 1) * batch_size]] for data_idx in range(9)]
            structure_path = [path_dir + single_path + '.npy' for single_path in structure_path]

            distance_mat = [np.load(pair_rep_info, allow_pickle=True).item()['distance_matrix'][chain[i]] for
                            i, pair_rep_info in enumerate(structure_path)]

            torsion_mat = [np.load(pair_rep_info, allow_pickle=True).item()['dihedral_array'][chain[i]] for
                           i, pair_rep_info in enumerate(structure_path)]

            inputs = [input_vertex, input_edge, input_atom_adj, input_bond_adj, input_num_nbs, input_seq, distance_mat,
                      torsion_mat]

            vertex_mask, vertex, edge, atom_adj, bond_adj, nbs_mask, seq_mask, sequence, \
            dist_mat, torsion_mat, dist_mat_mask = batch_data_process(inputs)

            affinity_label = np.array(affinity_label, dtype=np.float32)
            affinity_label = torch.FloatTensor(affinity_label).cuda()
            optimizer.zero_grad()
            affinity_pred = net(vertex_mask, vertex, edge, atom_adj, bond_adj, nbs_mask, seq_mask, sequence, dist_mat,
                                torsion_mat, dist_mat_mask)
            loss_aff = criterion(affinity_pred, affinity_label)
            affinity_loss += float(loss_aff.data * batch_size)
            loss_aff.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 5)
            optimizer.step()
            optimizer.zero_grad()
        print('Epoch:', epoch, ' Affinity Loss: ', str(round(affinity_loss / float(len(train_data[0])), 6)))
        perf_name = ['rmse', 'pearson', 'spearman', 'r_square_score', 'MedAE', 'MAE', 'MAPE']
        train_performance, train_label, train_output = test_func(net, train_data, batch_size)
        train_result.append(train_performance)
        print_perf = [perf_name[i] + ' ' + str(round(train_performance[i], 6)) for i in range(len(perf_name))]
        print('Train Performances: ', len(train_output), ' '.join(print_perf))
        # Save the model after each epoch
        torch.save(net, f'../results/model/epoch_{epoch}_model_old.pth')

    print('Finished Training Process!')
    return net


def test_performances_of_FeatNN(net, test_data, batch_size):
    perf_name = ['rmse', 'pearson', 'spearman', 'r_square_score', 'MedAE', 'MAE', 'MAPE']
    test_result = []
    test_performance, test_label, test_output = test_func(net, test_data, batch_size)
    print_perf = [perf_name[i] + ' ' + str(round(test_performance[i], 6)) for i in range(len(perf_name))]
    print('Test Performances: ', len(test_output), ' '.join(print_perf))
    test_result.append(test_performance)
    return test_performance, test_label, test_output


def test_func(net, test_data, batch_size):
    path_dir = r'../data/DT/'
    output_list, label_list = [], []
    net.eval()
    shuffle_index = np.arange(len(test_data[0]))
    with torch.no_grad():
        for i in range(int(math.ceil(len(test_data[0]) / float(batch_size)))):
            input_vertex, input_edge, input_atom_adj, input_bond_adj, input_num_nbs, input_seq, affinity_label, chain, structure_path = \
                [train_data[data_idx][shuffle_index[i * batch_size:(i + 1) * batch_size]] for data_idx in range(9)]
            structure_path = [path_dir + single_path + '.npy' for single_path in structure_path]

            distance_mat = [np.load(pair_rep_info, allow_pickle=True).item()['distance_matrix'][chain[i]] for
                            i, pair_rep_info in enumerate(structure_path)]

            torsion_mat = [np.load(pair_rep_info, allow_pickle=True).item()['dihedral_array'][chain[i]] for
                           i, pair_rep_info in enumerate(structure_path)]

            inputs = [input_vertex, input_edge, input_atom_adj, input_bond_adj, input_num_nbs, input_seq, distance_mat,
                      torsion_mat]

            vertex_mask, vertex, edge, atom_adj, bond_adj, nbs_mask, seq_mask, sequence, \
            dist_mat, torsion_mat, dist_mat_mask = batch_data_process(inputs)

            affinity_pred = net(vertex_mask, vertex, edge, atom_adj, bond_adj, nbs_mask, seq_mask, sequence, dist_mat, torsion_mat, dist_mat_mask)
            output_list += affinity_pred.cpu().detach().numpy().reshape(-1).tolist()
            label_list += affinity_label.reshape(-1).tolist()
        output_list = np.array(output_list)
        label_list = np.array(label_list)
    rmse, pearson, spearman, r_square_score, MedAE, MAE, MAPE = reg_scores(label_list, output_list)

    test_performance = [rmse, pearson, spearman, r_square_score, MedAE, MAE, MAPE]
    return test_performance, label_list, output_list


if __name__ == "__main__":
    # Clean the cache Info of CUDA
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    # evaluate params
    measure = "EC50"
    clu_thre, n_epoch, n_rep, batch_size = 0.3, 50, 2, 16

    # loading params of the model
    with open('param.json', 'r') as f:
        params = json.load(f)
    f.close()

    # print evaluation scheme
    print('Dataset: BindingDB v2022 with measurement', measure)
    print('Clustering threshold:', clu_thre)
    print('Number of epochs:', n_epoch)
    print('Number of repeats:', n_rep)

    # load data
    train_data = np.load('../data/datapack/comp_prot_ec50_train.npy', allow_pickle=True)
    # train_data = [item[:70] for item in train_data]
    test_data = np.load('../data/datapack/comp_prot_ec50_test.npy', allow_pickle=True)

    print('train num:', len(train_data[0]), 'test num:', len(test_data[0]))

    # load compound features
    init_A, init_B = loading_comp_repr(measure)
    # Construct the object of Model FeatNN
    net = FEATNN(init_A, init_B, params)

    feat = train_and_eval(net, train_data, batch_size, n_epoch)

    test_performance, test_label, test_output = test_performances_of_FeatNN(feat, test_data, batch_size)
    print("Final Test Performance: ", test_performance)
    print('FeatNN train&test processes have been finished!')