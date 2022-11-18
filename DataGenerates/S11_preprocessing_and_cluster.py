from collections import defaultdict
import os
import pickle
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from scipy.cluster.hierarchy import fcluster, linkage, single
from scipy.spatial.distance import pdist


def get_valid_pdbid(path):
    valid_pdbid = []
    for root, dirs, files in os.walk(path):
        print(len(files))
    for pdbvligand in files:
        valid_pdbid.append(pdbvligand[:4])
    np.save('./fileprocessing/output/validpdbid.npy', valid_pdbid)


def onek_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def atom_features(atom):
    return np.array(onek_encoding_unk(atom.GetSymbol(), elem_list)
                    + onek_encoding_unk(atom.GetDegree(), [0, 1, 2, 3, 4, 5])
                    + onek_encoding_unk(atom.GetExplicitValence(), [1, 2, 3, 4, 5, 6])
                    + onek_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5])
                    + [atom.GetIsAromatic()], dtype=np.float32)


def bond_features(bond):
    bt = bond.GetBondType()
    return np.array(
        [bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE, bt == Chem.rdchem.BondType.TRIPLE, \
         bt == Chem.rdchem.BondType.AROMATIC, bond.GetIsConjugated(), bond.IsInRing()], dtype=np.float32)


def Mol2Graph(mol):
    # convert molecule to GNN input
    idxfunc = lambda x: x.GetIdx()

    n_atoms = mol.GetNumAtoms()

    assert mol.GetNumAtoms() >= 0
    assert mol.GetNumBonds() >= 0

    n_bonds = max(mol.GetNumBonds(), 1)
    fatoms = np.zeros((n_atoms,), dtype=np.int32)  # atom feature ID
    fbonds = np.zeros((n_bonds,), dtype=np.int32)  # bond feature ID
    atom_nb = np.zeros((n_atoms, max_nb), dtype=np.int32)
    bond_nb = np.zeros((n_atoms, max_nb), dtype=np.int32)
    num_nbs = np.zeros((n_atoms,), dtype=np.int32)
    num_nbs_mat = np.zeros((n_atoms, max_nb), dtype=np.int32)

    for atom in mol.GetAtoms():
        idx = idxfunc(atom)
        fatoms[idx] = atom_dict[''.join(str(x) for x in atom_features(atom).astype(int).tolist())]

    for bond in mol.GetBonds():
        a1 = idxfunc(bond.GetBeginAtom())
        a2 = idxfunc(bond.GetEndAtom())
        idx = bond.GetIdx()
        fbonds[idx] = bond_dict[''.join(str(x) for x in bond_features(bond).astype(int).tolist())]
        try:
            atom_nb[a1, num_nbs[a1]] = a2
            atom_nb[a2, num_nbs[a2]] = a1
        except:
            return [], [], [], [], []
        bond_nb[a1, num_nbs[a1]] = idx
        bond_nb[a2, num_nbs[a2]] = idx
        num_nbs[a1] += 1
        num_nbs[a2] += 1

    for i in range(len(num_nbs)):
        num_nbs_mat[i, :num_nbs[i]] = 1

    return fatoms, fbonds, atom_nb, bond_nb, num_nbs_mat


def Protein2Sequence(sequence, ngram=1):
    # convert sequence to CNN input
    sequence = sequence.upper()
    word_list = [sequence[i:i + ngram] for i in range(len(sequence) - ngram + 1)]
    output = []
    for word in word_list:
        if word not in aa_list:
            output.append(word_dict['X'])
        else:
            output.append(word_dict[word])
    if ngram == 3:
        output = [-1] + output + [-1]  # pad
    return np.array(output, np.int32)


def get_mol_dict():
    if os.path.exists('./fileprocessing/output/mol_dict'):
        with open('./fileprocessing/output/mol_dict', 'rb') as f:
            mol_dict = pickle.load(f, encoding='iso-8859-1')
    else:
        mol_dict = {}
        mols = Chem.SDMolSupplier('./fileprocessing/Components-pub.sdf')
        for m in mols:
            if m is None:
                continue
            name = m.GetProp("_Name")
            mol_dict[name] = m
        with open('./fileprocessing/output/mol_dict', 'wb') as f:
            pickle.dump(mol_dict, f)
    # print('mol_dict',len(mol_dict))
    return mol_dict


def get_fps(mol_list):
    fps = []
    for mol in mol_list:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024, useChirality=True)
        fps.append(fp)
    return fps


def calculate_sims(fps1, fps2, simtype='tanimoto'):
    sim_mat = np.zeros((len(fps1), len(fps2)))  # ,dtype=np.float32)
    for i in range(len(fps1)):
        fp_i = fps1[i]
        if simtype == 'tanimoto':
            sims = DataStructs.BulkTanimotoSimilarity(fp_i, fps2)
        elif simtype == 'dice':
            sims = DataStructs.BulkDiceSimilarity(fp_i, fps2)
        sim_mat[i, :] = sims
    return sim_mat


def compound_clustering(ligand_list, mol_list):
    print('start compound clustering...')
    fps = get_fps(mol_list)
    sim_mat = calculate_sims(fps, fps)
    print('compound sim mat', sim_mat.shape)
    C_dist = pdist(fps, 'jaccard')
    C_link = single(C_dist)
    for thre in [0.3, 0.4, 0.5, 0.6]:
        C_clusters = fcluster(C_link, thre, 'distance')
        len_list = []
        for i in range(1, max(C_clusters) + 1):
            len_list.append(C_clusters.tolist().count(i))
        print('thre', thre, 'total num of compounds', len(ligand_list), 'num of clusters', max(C_clusters),
              'max length', max(len_list))
        C_cluster_dict = {ligand_list[i]: C_clusters[i] for i in range(len(ligand_list))}
        with open('./preprocessing/'+MEASURE+'/' + MEASURE + '_compound_cluster_dict_' + str(thre), 'wb') as f:
            pickle.dump(C_cluster_dict, f, protocol=0)
        f.close()


def protein_clustering(protein_list, idx_list):
    print('start protein clustering...')
    protein_sim_mat = np.load('./fileprocessing/output/pdbbind_protein_sim_mat.npy').astype(np.float32)
    sim_mat = protein_sim_mat[idx_list, :]
    sim_mat = sim_mat[:, idx_list]
    print('original protein sim_mat', protein_sim_mat.shape, 'subset sim_mat', sim_mat.shape)
    P_dist = []
    for i in range(sim_mat.shape[0]):
        P_dist += (1 - sim_mat[i, (i + 1):]).tolist()
    P_dist = np.array(P_dist)
    P_link = single(P_dist)
    for thre in [0.3, 0.4, 0.5, 0.6]:
        P_clusters = fcluster(P_link, thre, 'distance')
        len_list = []
        for i in range(1, max(P_clusters) + 1):
            len_list.append(P_clusters.tolist().count(i))
        print('thre', thre, 'total num of proteins', len(protein_list), 'num of clusters', max(P_clusters),
              'max length', max(len_list))
        P_cluster_dict = {protein_list[i]: P_clusters[i] for i in range(len(protein_list))}
        with open('./preprocessing/'+MEASURE+'/' + MEASURE + '_protein_cluster_dict_' + str(thre), 'wb') as f:
            pickle.dump(P_cluster_dict, f, protocol=0)
        f.close()

def pickle_dump(dictionary, file_name):
    pickle.dump(dict(dictionary), open(file_name, 'wb'), protocol=0)


if __name__ == "__main__":
    MEASURE = 'IC50'  # 'IC50' or 'KIKD'
    print('Create dataset for measurement:', MEASURE)
    print('Step 1/5, loading dict...')
    # load label dicts
    mol_dict = get_mol_dict()
    elem_list = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K',
                 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd',
                 'In',
                 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'W', 'Ru', 'Nb', 'Re', 'Te', 'Rh', 'Tc', 'Ba', 'Bi', 'Hf', 'Mo',
                 'U',
                 'Sm', 'Os', 'Ir', 'Ce', 'Gd', 'Ga', 'Cs', 'unknown']
    aa_list = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    atom_fdim = len(elem_list) + 6 + 6 + 6 + 1
    bond_fdim = 6
    max_nb = 6
    with open('./fileprocessing/output/FEATNN_final_interaction_dict', 'rb') as f:
        interaction_dict = pickle.load(f)
    f.close()
    # get the valid pdbid from the directory of DDM_TM
    get_valid_pdbid('./DDM_TM/')
    valid_pdbid = np.load('./fileprocessing/output/valid_pdbid.npy', allow_pickle=True)  # num 14509

    # initialize feature dicts
    wlnn_train_list = []
    atom_dict = defaultdict(lambda: len(atom_dict))
    bond_dict = defaultdict(lambda: len(bond_dict))
    word_dict = defaultdict(lambda: len(word_dict))
    for aa in aa_list:
        word_dict[aa]
    word_dict['X']

    # get labels
    i = 0
    pair_info_dict = {}
    f = open('./fileprocessing/output/out2_pdbbind_data_datafile.tsv')
    print('Step 2/5, generating labels...')
    for line in f.readlines():
        i += 1
        if i % 1000 == 0:
            print('processed sample num', i)
        pdbid, pid, cid, inchi, seq, measure, label = line.strip().split('\t')
        if pdbid not in valid_pdbid:
            continue
        # filter interaction type and invalid molecules
        if MEASURE == 'All':
            pass
        elif MEASURE == 'KIKD':
            if measure not in ['Ki', 'Kd']:
                continue
        elif measure != MEASURE:
            continue
        if cid not in mol_dict:
            print('ligand not in mol_dict')
            continue
        mol = mol_dict[cid]
        # get labels
        value = float(label)
        # get interact chain
        chain = interaction_dict[pdbid]['chain']
        sequence = interaction_dict[pdbid]['uniprot_seq']
        # handle the condition when multiple PDB entries have the same Uniprot ID and Inchi
        if inchi + ' ' + pid not in pair_info_dict:
            pair_info_dict[inchi + ' ' + pid] = [pdbid, cid, pid, value, mol, sequence, chain]
        else:
            if pair_info_dict[inchi + ' ' + pid][3] < value:
                pair_info_dict[inchi + ' ' + pid] = [pdbid, cid, pid, value, mol, sequence, chain]
    f.close()

    print('Step 3/5, generating inputs...')
    valid_value_list, valid_cid_list, valid_pid_list, valid_chain = [], [], [], []
    mol_inputs, seq_inputs = [], []

    # get inputs
    for item in pair_info_dict:
        pdbid, cid, pid, value, mol, seq, s_chain = pair_info_dict[item]
        fa, fb, anb, bnb, nbs_mat = Mol2Graph(mol)
        if fa == []:
            print('num of neighbor > 6, ', cid)
            continue
        mol_inputs.append([fa, fb, anb, bnb, nbs_mat])
        seq_inputs.append(Protein2Sequence(seq, ngram=1))
        valid_value_list.append(value)
        valid_cid_list.append(cid)
        valid_pid_list.append(pid)
        valid_chain.append(s_chain)
        wlnn_train_list.append(pdbid)

    print('Step 4/5, saving data...')
    # get data pack
    fa_list, fb_list, anb_list, bnb_list, nbs_mat_list = zip(*mol_inputs)
    data_pack = [np.array(fa_list), np.array(fb_list), np.array(anb_list), np.array(bnb_list), np.array(nbs_mat_list),
                 np.array(seq_inputs), np.array(valid_value_list), np.array(valid_cid_list), np.array(valid_pid_list),
                 np.array(wlnn_train_list), np.array(valid_chain)]
    index_lst = [np.array(valid_cid_list), np.array(valid_pid_list), np.array(wlnn_train_list)]
    np.save('./preprocessing/'+MEASURE+'/id_index_'+MEASURE+'.npy', index_lst)
    # save data
    # with open('./preprocessing/'+MEASURE+'/pdbbind_combined_input_' + MEASURE, 'wb') as f:
    #     pickle.dump(data_pack, f, protocol=0)
    # f.close()
    np.save('./preprocessing/'+MEASURE+'/pdbbind_combined_input_' + MEASURE, data_pack)

    np.save('./preprocessing/'+MEASURE+'/wlnn_train_list_' + MEASURE, wlnn_train_list)

    pickle_dump(atom_dict, './preprocessing/'+MEASURE+'/pdbbind_atom_dict_' + MEASURE)
    pickle_dump(bond_dict, './preprocessing/'+MEASURE+'/pdbbind_bond_dict_' + MEASURE)
    pickle_dump(word_dict, './preprocessing/'+MEASURE+'/pdbbind_word_dict_' + MEASURE)

    print('Step 5/5, clustering...')
    compound_list = list(set(valid_cid_list))
    protein_list = list(set(valid_pid_list))
    mol_list = [mol_dict[ligand] for ligand in compound_list]

    compound_clustering(compound_list, mol_list)
    # protein clustering
    ori_protein_list = np.load('./fileprocessing/output/pdbbind_protein_list.npy', allow_pickle=True).tolist()
    # Add
    refined_protein_list = []
    for pid in protein_list:
        if pid in ori_protein_list:
            refined_protein_list.append(pid)

    idx_list = [ori_protein_list.index(pid) for pid in refined_protein_list]
    protein_clustering(refined_protein_list, idx_list)

    print('=' * 50)
    print('Finish generating dataset for measurement', MEASURE)
    print('Number of valid samples', len(valid_value_list))
    print('Number of unique compounds', len(compound_list))
    print('Number of unique proteins', len(protein_list))
