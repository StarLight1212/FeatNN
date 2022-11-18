import pickle
from sklearn.model_selection import KFold
import argparse
from metrics import *


def data_from_index(data_pack, idx_list):
    fa, fb, anb, bnb, nbs_mat, seq_input = [data_pack[i][idx_list] for i in range(6)]
    aff_label = data_pack[6][idx_list].astype(float).reshape(-1,1)
    cid = data_pack[7][idx_list]
    pdbid = data_pack[9][idx_list]
    assert len(cid) == len(pdbid)
    structure_path = [pdbid[i]+'_'+cid[i]+'.npy' for i in range(len(cid))]
    chain = data_pack[10][idx_list]
    return [fa, fb, anb, bnb, nbs_mat, seq_input, aff_label, np.array(structure_path), np.array(chain)]


def construct_train_test_clusters(measure, clu_thre, n_fold):
    # load cluster dict
    cluster_path = r'./preprocessing/'+measure+'/'
    with open(cluster_path+measure+'_compound_cluster_dict_'+str(clu_thre), 'rb') as f:
        C_cluster_dict = pickle.load(f)
    f.close()
    with open(cluster_path+measure+'_protein_cluster_dict_'+str(clu_thre), 'rb') as f:
        P_cluster_dict = pickle.load(f)
    f.close()
    C_cluster_set = set(list(C_cluster_dict.values()))
    P_cluster_set = set(list(P_cluster_dict.values()))
    C_cluster_list = np.array(list(C_cluster_set))
    P_cluster_list = np.array(list(P_cluster_set))
    np.random.shuffle(C_cluster_list)
    np.random.shuffle(P_cluster_list)
    # n-fold split
    c_kf = KFold(n_fold, shuffle=True)
    p_kf = KFold(n_fold, shuffle=True)
    c_train_clusters, c_test_clusters = [], []
    print("c_kf", c_kf)
    print("p_kf", p_kf)
    for train_idx, test_idx in c_kf.split(C_cluster_list):    # split(C_cluster_list):
        c_train_clusters.append(C_cluster_list[train_idx])
        c_test_clusters.append(C_cluster_list[test_idx])
    p_train_clusters, p_test_clusters = [], []
    for train_idx, test_idx in p_kf.split(P_cluster_list):    # split(P_cluster_list):
        p_train_clusters.append(P_cluster_list[train_idx])
        p_test_clusters.append(P_cluster_list[test_idx])

    return c_train_clusters, c_test_clusters, p_train_clusters, p_test_clusters, C_cluster_dict, P_cluster_dict


def load_dataset(measure, setting, clu_thre, n_fold):
    # load data
    data_pack = np.load('./preprocessing/'+measure+'/pdbbind_combined_input_'+measure+'.npy', allow_pickle=True)
    cid_list, pid_list = data_pack[7], data_pack[8]
    n_sample = len(cid_list)
    # train-test split
    train_idx_list, valid_idx_list, test_idx_list = [], [], []
    c_train_clusters, c_test_clusters, p_train_clusters, p_test_clusters, C_cluster_dict, P_cluster_dict = construct_train_test_clusters(measure, clu_thre, n_fold)

    print('setting:', setting)
    if setting == 'ProtClu':
        for fold in range(n_fold):
            p_train_valid, p_test = p_train_clusters[fold], p_test_clusters[fold]
            p_valid = np.random.choice(p_train_valid, int(len(p_train_valid)*0.125), replace=False)
            p_train = set(p_train_valid)-set(p_valid)
            train_idx, valid_idx, test_idx = [], [], []
            for ele in range(n_sample):
                if P_cluster_dict[pid_list[ele]] in p_train:
                    train_idx.append(ele)
                elif P_cluster_dict[pid_list[ele]] in p_valid:
                    valid_idx.append(ele)
                elif P_cluster_dict[pid_list[ele]] in p_test:
                    test_idx.append(ele)
                else:
                    print('error')
            train_idx_list.append(train_idx)
            valid_idx_list.append(valid_idx)
            test_idx_list.append(test_idx)
            print('fold', fold, 'train ',len(train_idx),'test ',len(test_idx),'valid ',len(valid_idx))

    elif setting == 'ComClu':
        for fold in range(n_fold):
            c_train_valid, c_test = c_train_clusters[fold], c_test_clusters[fold]
            c_valid = np.random.choice(c_train_valid, int(len(c_train_valid)*0.125), replace=False)
            c_train = set(c_train_valid)-set(c_valid)
            train_idx, valid_idx, test_idx = [], [], []
            for ele in range(n_sample):
                if C_cluster_dict[cid_list[ele]] in c_train:
                    train_idx.append(ele)
                elif C_cluster_dict[cid_list[ele]] in c_valid:
                    valid_idx.append(ele)
                elif C_cluster_dict[cid_list[ele]] in c_test:
                    test_idx.append(ele)
                else:
                    print('error')
            train_idx_list.append(train_idx)
            valid_idx_list.append(valid_idx)
            test_idx_list.append(test_idx)
            print('fold', fold, 'train ',len(train_idx),'test ',len(test_idx),'valid ',len(valid_idx))

    else:
        raise KeyError('Wrong input setting values!')
    return data_pack, train_idx_list, valid_idx_list, test_idx_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--measures", type=str, default="IC50",
                        help="The measurement of the binding affinity, IC50 or KIKD.")
    parser.add_argument("--setting", type=str, default="ComClu", help="Cluster strategy: ComClu or ProtClu")
    parser.add_argument("--threshold", type=float, default=0.3,
                        help="Threshold value of cluster strategy: 0.3, 0.4, 0.5 and 0.6.")
    parser.add_argument("--n_fold", type=int, default=1, help="K Fold value for the k fold validation, in FeatNN we apply 5-fold cross validation!")

    opt = parser.parse_args()
    # Obtain the setting parameters from the input command.
    measure, setting, clu_thre, n_fold = opt.measures, opt.setting, opt.threshold, opt.n_fold
    assert measure in ['IC50', 'KIKD'], 'Measurement with wrong input!'
    assert setting in ['ComClu', 'ProtClu'], 'Setting with wrong input!'
    assert clu_thre in [0.3, 0.4, 0.5, 0.6], 'Threshold with wrong input!'
    # Obtain the datapack and the index list for train, valid and test dataset.
    datapack, trn_idx_lst, val_idx_lst, tst_idx_lst = load_dataset(measure, setting, clu_thre, n_fold)
    for fold in range(n_fold):
        train_data = data_from_index(datapack, trn_idx_lst)
        valid_data = data_from_index(datapack, val_idx_lst)
        test_data = data_from_index(datapack, tst_idx_lst)
        with open('../Datasets/Train_test_data/'+measure+'/'+setting+'/'+'train_set_fold_' + str(fold)+'_thresh_'+str(clu_thre)+'_'+str(setting), 'wb') as f:
            pickle.dump(train_data, f)
        f.close()
        with open('../Datasets/Train_test_data/'+measure+'/'+setting+'/'+'test_set_fold_' + str(fold)+'_thresh_'+str(clu_thre)+'_'+str(setting), 'wb') as f:
            pickle.dump(test_data, f)
        f.close()
        with open('../Datasets/Train_test_data/'+measure+'/'+setting+'/'+'valid_set_fold_' + str(fold)+'_thresh_'+str(clu_thre)+'_'+str(setting), 'wb') as f:
            pickle.dump(valid_data, f)
        f.close()