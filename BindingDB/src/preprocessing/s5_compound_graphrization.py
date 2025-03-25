import os
import Bio.PDB
from Bio.PDB.Polypeptide import three_to_one
from collections import defaultdict
from rdkit import Chem
import pickle
from rdkit.Chem import AllChem
from rdkit import DataStructs
import numpy as np


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


def Mol2Graph(mol, max_nb=6):

    idxfunc = lambda x: x.GetIdx()
    try:
        n_atoms = mol.GetNumAtoms()
    except AttributeError as AE:
        return [], [], [], [], []

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
        try:
            fatoms[idx] = atom_dict[''.join(str(x) for x in atom_features(atom).astype(int).tolist())]
        except KeyError as KK:
            print("Key Error!")
            return [], [], [], [], []

    for bond in mol.GetBonds():
        a1 = idxfunc(bond.GetBeginAtom())
        a2 = idxfunc(bond.GetEndAtom())
        idx = bond.GetIdx()
        try:
            fbonds[idx] = bond_dict[''.join(str(x) for x in bond_features(bond).astype(int).tolist())]
        except KeyError as KK:
            print("Key Error!")
            return [], [], [], [], []
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


def pickle_dump(dictionary, file_name):
    pickle.dump(dict(dictionary), open(file_name, 'wb'), protocol=0)


if __name__ == '__main__':
    MEASURE = 'EC50'
    path_dir = '../../data/DT/'
    aa_list = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    elem_list = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K',
                 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In',
                 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'W', 'Ru', 'Nb', 'Re', 'Te', 'Rh', 'Tc', 'Ba', 'Bi', 'Hf', 'Mo', 'U',
                 'Sm', 'Os', 'Ir', 'Ce', 'Gd', 'Ga', 'Cs', 'unknown']
    atom_dict = defaultdict(lambda: len(atom_dict))
    bond_dict = defaultdict(lambda: len(bond_dict))
    word_dict = defaultdict(lambda: len(word_dict))
    for aa in aa_list:
        word_dict[aa]
    word_dict['X']

    valid_pdb = np.load('../../data/valid_pdbid.npy', allow_pickle=True)
    print("Valid Protein Length", valid_pdb)
    mol_inputs, seq_inputs = [], []
    valid_value_list = []
    valid_chain, f_pid_list = [], []
    with open('../../data/bindingdb_data_ec.csv', 'r') as f:
        for idx, line in enumerate(f.readlines()):
            # pdbid_set.add(line.strip().split(',')[1])
            print("index: ", idx)
            pdbid = line.strip().split(',')[1]
            if pdbid not in valid_pdb:
                continue
            seq_tmp = line.strip().split(',')[2]
            seq, chain = get_seq(pdbid, seq_tmp)
            if seq is None or seq == '':
                continue
            smiles = line.strip().split(',')[0]
            affinity = line.strip().split(',')[5]
            # print(affinity)
            mol = Chem.MolFromSmiles(smiles)
            # print(mol)
            fa, fb, anb, bnb, nbs_mat = Mol2Graph(mol)
            if fa == []:
                print('num of neighbor > 6, ', mol)
                continue

            mol_inputs.append([fa, fb, anb, bnb, nbs_mat])
            seq_inputs.append(Protein2Sequence(seq))
            # print(seq)
            # print()
            assert len(seq) == len(seq_inputs[-1])
            # print("seq_inputs: ", seq_inputs[-1])
            valid_value_list.append(affinity)
            valid_chain.append(chain)
            f_pid_list.append(pdbid)
    f.close()
    fa_list, fb_list, anb_list, bnb_list, nbs_mat_list = zip(*mol_inputs)
    data_pack = [np.array(fa_list), np.array(fb_list), np.array(anb_list), np.array(bnb_list),
                 np.array(nbs_mat_list),
                 np.array(seq_inputs), np.array(valid_value_list), np.array(valid_chain),
                 np.array(f_pid_list)]
    pickle_dump(atom_dict, './pc_dict/pdbbind_atom_dict_' + MEASURE)
    pickle_dump(bond_dict, './pc_dict/pdbbind_bond_dict_' + MEASURE)
    pickle_dump(word_dict, './pc_dict/pdbbind_word_dict_' + MEASURE)

    np.save("../../data/comp_graph/comp_prot_ec50.npy", data_pack)
    print('Length of all data: ', len(valid_value_list))

