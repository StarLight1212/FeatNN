import numpy as np
import pickle
import os
from rdkit import Chem
import math


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
    return mol_dict


def get_pdb_align_list():
    pdbbind_set = set()
    f = open('./fileprocessing/output/out6.3_pdb_align.txt')
    for line in f.readlines():
        if 'query_name' in line:
            uniprotis = line.strip().split('_')[-1]
            pdbbind_set.add(uniprotis)
    f.close()
    return list(pdbbind_set)


def test():
    pdbbind_set = set()
    f = open('./fileprocessing/output/out6.3_pdb_align.txt')
    i = 0
    for line in f.readlines():
        if 'P61823' in line:
            i += 1
            print(i)
    f.close()
    return list(pdbbind_set)


def get_uniprot_id(pdb_align_list:list):
    mol_dict = get_mol_dict()
    uniprot_id_set = set()
    f = open('./fileprocessing/output/out2_pdbbind_data_datafile.tsv')
    for line in f.readlines():
        pdbid, uniprot_id, cid, inchi, seq, measure, label = line.strip().split('\t')
        if measure not in ['Ki','Kd','IC50']:
            continue
        if cid not in mol_dict:
            continue
        if uniprot_id in pdb_align_list:
            uniprot_id_set.add(uniprot_id)

    f.close()
    return list(uniprot_id_set)


def construct_sequence_fasta_file(protein_list, path):
    protein_list = protein_list
    fasta_dict = {}
    with open(path) as f:
        for line in f.readlines():
            uniprot_id = line.strip().split('\t')[1]
            seq = line.strip().split('\t')[4]
            if uniprot_id not in fasta_dict:
                if uniprot_id in protein_list:
                    fasta_dict[uniprot_id] = seq
            else:
                assert seq == fasta_dict[uniprot_id]
                continue
    f.close()
    fw = open("./fileprocessing/output/protein_sequences_info.fasta", "w")
    for name in protein_list:
        fw.write('>' + name + '\n')
        fw.write(fasta_dict[name] + '\n')

    fw.close()


def identity_alignment_score():
    identity_score_dict = {}
    with open("./fileprocessing/output/protein_sequence_alignment.txt", 'r') as f:
        for line in f.readlines():
            if "target_name" in line:
                targetid = line.strip().split(" ")[1]
                # print(targetid)
            elif "query_name" in line:
                queryid = line.strip().split(" ")[1]
            if "optimal" in line:
                if queryid == targetid:
                    identity_score = line.strip().split("\t")[0].split(" ")[1]
                    identity_score_dict[queryid] = int(identity_score)
    return identity_score_dict


def paired_alignment_score():
    paired_score_dict = {}
    with open("./fileprocessing/output/protein_sequence_alignment.txt", 'r') as f:
        for line in f.readlines():
            if "target_name" in line:
                targetid = line.strip().split(" ")[1]
                # print(targetid)
            elif "query_name" in line:
                queryid = line.strip().split(" ")[1]
            if "optimal" in line:
                if queryid != targetid:
                    identity_score = line.strip().split("\t")[0].split(" ")[1]
                    paired_score_dict[queryid+"_"+targetid] = int(identity_score)
        return paired_score_dict


def msa_construction(protein_list, identity_score, paired_score):
    protein_list = protein_list
    identity_score = identity_score
    paired_score = paired_score
    n_length = len(protein_list)
    # initialize the sim matrix as zero matrix
    protein_sim_mat = np.zeros((n_length,n_length))
    for i in protein_list:
        for j in range(n_length):
            if i == protein_list[j]:
                protein_sim_mat[protein_list.index(i)][j] = identity_score[i]/identity_score[i]
            else:
                protein_sim_mat[protein_list.index(i)][j] = paired_score[i+"_"+protein_list[j]]/math.sqrt(identity_score[i]*identity_score[protein_list[j]])
    return protein_sim_mat


if __name__ == '__main__':
    # test()
    # MSA_score_normalization()
    pdb_align_list = get_pdb_align_list()
    protein_list = get_uniprot_id(pdb_align_list=pdb_align_list)
    print(len(pdb_align_list))
    np.save('./fileprocessing/output/pdbbind_protein_list.npy', pdb_align_list)
    # or
    ori_protein_list = np.load('./fileprocessing/output/pdbbind_protein_list.npy', allow_pickle=True).tolist()
    print(ori_protein_list)
    # protein list is generated from the out2_pdbbind_all_datafile1.tsv in the Scripts preprocessing and clustering
    idx_list = [ori_protein_list.index(pid) for pid in protein_list]
    print(len(ori_protein_list))
    print(idx_list)
    print(len(idx_list))
    # Construct FASTA File with uniprotid and protein sequences
    print(ori_protein_list)
    path = r"./fileprocessing/output/out2_pdbbind_data_datafile.tsv"
    construct_sequence_fasta_file(protein_list=ori_protein_list, path=path)

    # calculate the score of sw(i, j) where i=j
    identity_score = identity_alignment_score()
    # calculate the score of sw(i, j) where i!=j
    paired_score = paired_alignment_score()
    # Construct the MSA matrix through smith-waterman methods
    protein_sim_mat = msa_construction(protein_list=ori_protein_list, identity_score=identity_score, paired_score=paired_score)
    print(protein_sim_mat)
    np.save('./fileprocessing/output/pdbbind_protein_sim_mat.npy', protein_sim_mat)
    print(protein_sim_mat.shape)