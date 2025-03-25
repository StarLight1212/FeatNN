import Bio.PDB
from Bio.PDB.Polypeptide import three_to_one
import numpy as np
import os


def get_seq(pdbid, seq_query):
    p = Bio.PDB.PDBParser()
    try:
        structure = p.get_structure(pdbid, '../../data/pdb_files/' + pdbid + '.pdb')
    except Bio.PDB.PDBExceptions.PDBConstructionException as e:
        print("Wrong PDB File Info", pdbid)
        return None, None
    for model in structure:
        for chain in model:
            chain_id = chain.get_id()
            if chain_id == ' ':
                continue
            seq = ''
            for res in chain:
                if res.get_id()[0] != ' ' or res.get_id()[2] != ' ':  # remove HETATM
                    continue
                try:
                    seq += three_to_one(res.get_resname())
                except:
                    print('unexpected aa name', res.get_resname())
            if seq in seq_query:
                print("Found Target! Now return result! And chain id is ", chain_id)
                return seq, chain_id
    return None, None


def valid_pdbid(path):
    valid_pdbid = []
    for root, dirs, files in os.walk(path):
        print(len(files))
    with open('./output_files/pdb_id.txt', 'w') as f:
        for pdbvligand in files:
            valid_pdbid.append(pdbvligand[:4])
            f.write(pdbvligand[:4] + '\n')
    return valid_pdbid


if __name__ == '__main__':
    path_dir = '../../data/pdb_files/'
    valid_pdb = valid_pdbid(path_dir)
    sequence_list = []
    chain_lst = []
    print("Valid Protein Length", valid_pdb)
    with open('../../data/bindingdb_data_ec.csv', 'r') as f:
        for line in f.readlines()[:]:
            # pdbid_set.add(line.strip().split(',')[1])
            pdbid = line.strip().split(',')[1]
            if pdbid not in valid_pdb:
                continue
            seq_tmp = line.strip().split(',')[2]
            seq, chain = get_seq(pdbid, seq_tmp)
            if seq is None:
                continue
            chain_lst.append(chain)
            sequence_list.append(seq)
            # smiles = line.strip().split(',')[0]
            # affinity = line.strip().split(',')[5]
    f.close()
    print(len(sequence_list))
    np.save('./Data/chain_id.npy', chain_lst)
    np.save('./Data/seq.npy', sequence_list)