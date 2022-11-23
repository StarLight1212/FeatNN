import json
import argparse
import torch.optim as optim
from FEAT_utils import *
from FEATNN import FEATNN
import pickle


# train and evaluate
def train_and_eval(net, train_data, valid_data, batch_size, num_epoch):
    path_dir = r'../Datasets/PDBbind_Struct/'
    net.cuda()
    net.apply(weights_init)
    # Compute the total parameters
    pytorch_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print('total num params', pytorch_total_params)

    criterion = nn.MSELoss()    # while nn.HuberLoss() did not achieve high performance
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=0.00029, weight_decay=0, amsgrad=True)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=32, gamma=0.5)

    valid_result, train_result = [], []
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

            input_vertex, input_edge, input_atom_adj, input_bond_adj, input_num_nbs, input_seq, affinity_label, structure_path, chain = \
                [train_data[data_idx][shuffle_index[i * batch_size:(i + 1) * batch_size]] for data_idx in range(9)]
            structure_path = [path_dir+single_path for single_path in structure_path]

            distance_mat = [np.load(pair_rep_info, allow_pickle=True).item()['distance_matrix'][chain[i]] for i, pair_rep_info in enumerate(structure_path)]

            torsion_mat = [np.load(pair_rep_info, allow_pickle=True).item()['dihedral_array'][chain[i]] for i, pair_rep_info in enumerate(structure_path)]

            inputs = [input_vertex, input_edge, input_atom_adj, input_bond_adj, input_num_nbs, input_seq, distance_mat, torsion_mat]

            vertex_mask, vertex, edge, atom_adj, bond_adj, nbs_mask, seq_mask, sequence, \
                                                    dist_mat, torsion_mat, dist_mat_mask = batch_data_process(inputs)

            affinity_label = torch.FloatTensor(affinity_label).cuda()
            optimizer.zero_grad()
            affinity_pred = net(vertex_mask, vertex, edge, atom_adj, bond_adj, nbs_mask, seq_mask, sequence, dist_mat, torsion_mat, dist_mat_mask)
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
        net.eval()
        with torch.no_grad():
            valid_performance, valid_label, valid_output = test_func(net, valid_data, batch_size)
            valid_result.append(valid_performance)
            print_perf = [perf_name[i] + ' ' + str(round(valid_performance[i], 6)) for i in range(len(perf_name))]
            print('Valid Performances: ', len(valid_output), ' '.join(print_perf))
    torch.save(net, './model_save/FEAT_model_IC50_m'+str(fold)+'.pth')
    print('Finished Training Process!')
    return valid_performance, valid_label, valid_output, net


def test_performances_of_FeatNN(net, test_data, batch_size):
    perf_name = ['rmse', 'pearson', 'spearman', 'r_square_score', 'MedAE', 'MAE', 'MAPE']
    test_result = []
    test_performance, test_label, test_output = test_func(net, test_data, batch_size)
    print_perf = [perf_name[i] + ' ' + str(round(test_performance[i], 6)) for i in range(len(perf_name))]
    print('Test Performances: ', len(test_output), ' '.join(print_perf))
    test_result.append(test_performance)
    return test_performance, test_label, test_output


def test_func(net, test_data, batch_size):
    path_dir = r'../Datasets/PDBbind_Struct/'
    output_list, label_list = [], []
    for i in range(int(math.ceil(len(test_data[0]) / float(batch_size)))):
        input_vertex, input_edge, input_atom_adj, input_bond_adj, input_num_nbs, input_seq, affinity_label, structure_path, chain = \
            [test_data[data_idx][i * batch_size:(i + 1) * batch_size] for data_idx in range(9)]

        structure_path = [path_dir + single_path for single_path in structure_path]
        distance_mat = [np.load(pair_rep_info, allow_pickle=True).item()['distance_matrix'][chain[i]] for i, pair_rep_info in enumerate(structure_path)]

        torsion_mat = [np.load(pair_rep_info, allow_pickle=True).item()['dihedral_array'][chain[i]] for i, pair_rep_info in enumerate(structure_path)]

        inputs = [input_vertex, input_edge, input_atom_adj, input_bond_adj, input_num_nbs, input_seq, distance_mat, torsion_mat]

        vertex_mask, vertex, edge, atom_adj, bond_adj, nbs_mask, seq_mask, sequence, dist_mat, torsion_mat, dist_mat_mask = batch_data_process(inputs)

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--measures", type=str, default="IC50", help="The measurement of the binding affinity, IC50 or KIKD.")
    parser.add_argument("--setting", type=str, default="ComClu", help="Cluster strategy: ComClu for Compound-Clustered or ProtClu for Protein-Clustered")
    parser.add_argument("--threshold", type=float, default=0.3, help="Threshold values of cluster strategy: 0.3, 0.4, 0.5 and 0.6.")
    parser.add_argument("--param", type=str, default="param.json", help="Param: the path of param setting file (.json format) which recording the parameter setting of FeatNN.")
    parser.add_argument("--batch_size", type=int, default=10, help="Batch size for train, valid and test dataset.")
    parser.add_argument("--n_epoch", type=int, default=32, help="Epoch for training process.")

    opt = parser.parse_args()
    measure, setting, clu_thre = opt.measures, opt.setting, opt.threshold
    n_epoch, batch_size = opt.n_epoch, opt.batch_size
    # loading params of the FeatNN
    with open(opt.param, 'r') as f:
        params = json.load(f)
    f.close()

    assert setting in ['ComClu', 'ProtClu']
    assert clu_thre in [0.3, 0.4, 0.5, 0.6]
    assert measure in ['All', 'IC50', 'KIKD']
    repeat_train = 3

    # print evaluation scheme
    print('Dataset: PDBbind v2020 with measurement', measure)
    print('Clustering threshold:', clu_thre)
    print('Number of epochs:', n_epoch)

    # load data
    thre_str = str(clu_thre)[0]+'_'+str(clu_thre)[2]
    train_data = pickle.load(file=open('../Datasets/DataProcessed/'+measure+'/'+measure+'_'+setting+'_'+thre_str+'_train', 'rb'))
    valid_data = pickle.load(
        file=open('../Datasets/DataProcessed/' + measure + '/' + measure + '_' + setting + '_' + thre_str + '_valid', 'rb'))
    test_data = pickle.load(
        file=open('../Datasets/DataProcessed/' + measure + '/' + measure + '_' + setting + '_' + thre_str + '_test', 'rb'))
    print('train num:', len(train_data[0]), 'valid num:', len(valid_data[0]), 'test num:', len(test_data[0]))
    init_A, init_B = loading_comp_repr(measure)
    net = FEATNN(init_A, init_B, params)
    for fold in range(6):
        valid_performance, valid_label, valid_output, feat = train_and_eval(net, train_data, valid_data, batch_size, n_epoch)
        test_performance, test_label, test_output = test_performances_of_FeatNN(feat, test_data, batch_size)
        print("Final Test Performance: ", test_performance)
        print('FeatNN train&test processes have been finished!')