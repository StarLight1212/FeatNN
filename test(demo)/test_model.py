import argparse
import math
import numpy as np
import torch
from src.FEAT_utils import *
from src.FEATNN import FEATNN
import pickle


def test_performances_of_FeatNN(net, test_data, batch_size):
    perf_name = ['RMSE', 'Pearson', 'Spearman', 'R_square_score', 'MedAE', 'MAE', 'MAPE']
    test_result = []
    test_performance, test_label, test_output = test_func(net, test_data, batch_size)
    print_perf = [perf_name[i] + ' ' + str(round(test_performance[i], 6)) for i in range(len(perf_name))]
    print('Test ', len(test_output), ' '.join(print_perf))
    test_result.append(test_performance)
    return test_performance, test_label, test_output


def test_func(net, test_data, batch_size):
    path_dir = r'../Datasets/PDBbind_Struct/'
    output_list, label_list = [], []
    for i in range(int(math.ceil(len(test_data[0]) / float(batch_size)))):
        input_vertex, input_edge, input_atom_adj, input_bond_adj, input_num_nbs, input_seq, affinity_label, structure_path, chain = \
            [test_data[data_idx][i * batch_size:(i + 1) * batch_size] for data_idx in range(9)]

        structure_path = [path_dir + single_path for single_path in structure_path]
        distance_mat = [np.load(pair_rep_info, allow_pickle=True).item()['distance_matrix'][chain[i]] for   \
                        i, pair_rep_info in enumerate(structure_path)]

        torsion_mat = [np.load(pair_rep_info, allow_pickle=True).item()['dihedral_array'][chain[i]] for i, pair_rep_info
                       in enumerate(structure_path)]

        inputs = [input_vertex, input_edge, input_atom_adj, input_bond_adj, input_num_nbs, input_seq, distance_mat,
                  torsion_mat]

        vertex_mask, vertex, edge, atom_adj, bond_adj, nbs_mask, seq_mask, sequence, \
        dist_mat, torsion_mat, dist_mat_mask = batch_data_process(inputs)

        affinity_pred = net(vertex_mask, vertex, edge, atom_adj, bond_adj, nbs_mask, seq_mask,
                                               sequence, dist_mat, torsion_mat, dist_mat_mask)
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="FEAT_model_IC50_m1.pth", help="The directory of model file 'xxx.pth'")
    parser.add_argument("--batch_size", type=int, default=10,
                        help="Value of batch size for the test dataset")
    parser.add_argument("--measures", type=str, default="IC50",
                        help="The measurement of the binding affinity, IC50 or KIKD.")

    opt = parser.parse_args()
    measure, batch_size = opt.measures, opt.batch_size
    # load test data to verify FeatNN model
    test_data = pickle.load(
        file=open('./testdata/' + measure + '_ComClu_0_3_test', 'rb'))
    print('test num:', len(test_data[0]))
    # Loading Model
    feat = torch.load('./model/'+opt.model_dir)
    test_performance, test_label, test_output = test_performances_of_FeatNN(feat, test_data, batch_size)
    print("Test perfpormance: ", test_performance)