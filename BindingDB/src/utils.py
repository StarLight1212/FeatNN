import pickle
import torch
from torch import nn
import numpy as np
from torch.autograd import Variable
import math
from metrics import *
import random


# Chem Elements list
elem_list = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'W', 'Ru', 'Nb', 'Re', 'Te', 'Rh', 'Tc', 'Ba', 'Bi', 'Hf', 'Mo', 'U', 'Sm', 'Os', 'Ir', 'Ce','Gd','Ga','Cs', 'unknown']

atom_fdim = len(elem_list) + 6 + 6 + 6 + 1
bond_fdim = 6
max_nb = 6
LongTensor = torch.LongTensor
FloatTensor = torch.FloatTensor


def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True


# Model parameter intializer
def weights_init(m):
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
        nn.init.normal_(m.weight.data, mean=0, std=min(1.0 / math.sqrt(m.weight.data.shape[-1]), 0.1))
        nn.init.constant_(m.bias, 0)


def reg_scores(label, pred):
    label = label.reshape(-1)
    pred = pred.reshape(-1)
    return rmse(label, pred), pearson(label, pred), spearman(label, pred), r_square_score(label, pred), MedAE(label, pred), MAE(label, pred), MAPE(label, pred)


def pack2D(arr_list):
    N = max([x.shape[0] for x in arr_list])
    M = max([x.shape[1] for x in arr_list])
    a = np.zeros((len(arr_list), N, M))
    for i, arr in enumerate(arr_list):
        n = arr.shape[0]
        m = arr.shape[1]
        a[i, 0:n, 0:m] = arr
    return a


def pack1D(arr_list):
    N = max([x.shape[0] for x in arr_list])
    a = np.zeros((len(arr_list), N))
    for i, arr in enumerate(arr_list):
        n = arr.shape[0]
        a[i, 0:n] = arr
    return a


def get_mask2D(arr_list):
    N = max([x.shape[0] for x in arr_list])
    M = max([x.shape[1] for x in arr_list])
    a = np.zeros((len(arr_list), N, M))
    for i, arr in enumerate(arr_list):
        a[i,0:arr.shape[0],0:arr.shape[1]] = 1
    return a


def get_mask(arr_list):
    N = max([x.shape[0] for x in arr_list])
    a = np.zeros((len(arr_list), N))
    for i, arr in enumerate(arr_list):
        a[i,:arr.shape[0]] = 1
    return a


# embedding selection function
def add_index(input_array, embed_size):
    batch_size, n_vertex, n_nbs = np.shape(input_array)
    # UPDATE
    add_idx = np.array([i for i in range(0, (embed_size)*batch_size, embed_size)]*(n_nbs*n_vertex))
    add_idx = np.transpose(add_idx.reshape(-1, batch_size))
    add_idx = add_idx.reshape(-1)
    new_array = input_array.reshape(-1) + add_idx
    return new_array


# Function for generating batch data
def batch_data_process(data):
    # Load all data from datapack
    vertex, edge, atom_adj, bond_adj, nbs, sequence, dist_mat, torsion_mat = data
    # Compounds Data Package
    vertex_mask = get_mask(vertex)
    vertex, edge = pack1D(vertex), pack1D(edge)
    atom_adj, bond_adj, nbs_mask = pack2D(atom_adj), pack2D(bond_adj), pack2D(nbs)
    # add index
    atom_adj = add_index(atom_adj, np.shape(atom_adj)[1])
    bond_adj = add_index(bond_adj, np.shape(edge)[1])

    # Proteins Data Package and make masks
    seq_mask = get_mask(sequence)
    sequence = pack1D(sequence+1)
    # Structure information of Protein
    dist_mat_mask = get_mask2D(dist_mat)
    dist_mat, torsion_mat = pack2D(dist_mat), pack2D(torsion_mat)

    # convert to torch cuda data type
    vertex_mask = Variable(FloatTensor(vertex_mask)).cuda()
    vertex = Variable(LongTensor(vertex)).cuda()
    edge = Variable(LongTensor(edge)).cuda()
    atom_adj = Variable(LongTensor(atom_adj)).cuda()
    bond_adj = Variable(LongTensor(bond_adj)).cuda()
    nbs_mask = Variable(FloatTensor(nbs_mask)).cuda()
    dist_mat = Variable(LongTensor(dist_mat)).cuda()
    torsion_mat = Variable(FloatTensor(torsion_mat)).cuda()
    dist_mat_mask = Variable(FloatTensor(dist_mat_mask)).cuda()
    seq_mask = Variable(FloatTensor(seq_mask)).cuda()
    sequence = Variable(LongTensor(sequence)).cuda()
    return vertex_mask, vertex, edge, atom_adj, bond_adj, nbs_mask, seq_mask, sequence, dist_mat, torsion_mat, dist_mat_mask


def loading_comp_repr(measure):
    #load intial atom and bond features (i.e., embeddings)
    f = open('./preprocessing/pc_dict/pdbbind_atom_dict_EC50', 'rb')
    atom_dict = pickle.load(f)
    f.close()

    f = open('./preprocessing/pc_dict/pdbbind_bond_dict_EC50', 'rb')
    bond_dict = pickle.load(f)
    f.close()

    print('atom dict size:', len(atom_dict), ', bond dict size:', len(bond_dict))

    init_atom_features = np.zeros((len(atom_dict), atom_fdim))
    init_bond_features = np.zeros((len(bond_dict), bond_fdim))

    for key,value in atom_dict.items():
        init_atom_features[value] = np.array(list(map(int, key)))

    for key,value in bond_dict.items():
        init_bond_features[value] = np.array(list(map(int, key)))

    init_atom_features = Variable(torch.FloatTensor(init_atom_features)).cuda()
    init_bond_features = Variable(torch.FloatTensor(init_bond_features)).cuda()

    return init_atom_features, init_bond_features

