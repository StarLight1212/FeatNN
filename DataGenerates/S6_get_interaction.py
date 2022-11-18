from Bio.PDB import PDBParser, Selection
from Bio.PDB.Polypeptide import three_to_one
import os
import pickle


def get_pdbid_list():
    pdbid_list = []
    with open('./fileprocessing/output/out1.2_pdbid_list.txt') as f:
        for line in f.readlines():
            pdbid_list.append(line.strip())
    print('pdbid_list', len(pdbid_list))
    return pdbid_list


def get_pdbid_to_ligand():
    pdbid_to_ligand = {}
    with open('./fileprocessing/pdbbind_index/INDEX_general_PL.2020') as f:
        for line in f.readlines():
            if line[0] != '#':
                ligand = line.strip().split('(')[1].split(')')[0]
                if '-mer' in ligand:
                    continue
                elif '/' in ligand:
                    ligand = ligand.split('/')[0]
                if len(ligand) != 3:
                    # print(line[:4], ligand)
                    continue
                pdbid_to_ligand[line[:4]] = ligand
    print('pdbid_to_ligand', len(pdbid_to_ligand))
    return pdbid_to_ligand


def get_bonds(pdbid, ligand, atom_idx_list):
    bond_list = []
    f = open('./plip_result_all_set/' + pdbid + '_output.txt')
    isheader = False
    for line in f.readlines():
        if line[0] == '*':
            bond_type = line.strip().replace('*', '')
            isheader = True
        if line[0] == '|':
            if isheader:
                header = line.replace(' ', '').split('|')
                isheader = False
                continue
            lines = line.replace(' ', '').split('|')
            if ligand not in lines[5]:
                continue
            if bond_type in ['Salt Bridges']:
                aa_id, aa_name, aa_chain, ligand_id, ligand_name, ligand_chain = int(lines[1]), lines[2], lines[3], int(
                    lines[5]), lines[6], lines[7]
            else:
                aa_id, aa_name, aa_chain, ligand_id, ligand_name, ligand_chain = int(lines[1]), lines[2], lines[3], int(
                    lines[4]), lines[5], lines[6]
            if bond_type in ['Hydrogen Bonds', 'Water Bridges']:
                # print(lines[12],lines[14])
                atom_idx1, atom_idx2 = int(lines[12]), int(lines[14])
                if atom_idx1 in atom_idx_list and atom_idx2 in atom_idx_list:  # discard ligand-ligand interaction
                    continue
                if atom_idx1 in atom_idx_list:
                    atom_idx_ligand, atom_idx_protein = atom_idx1, atom_idx2
                elif atom_idx2 in atom_idx_list:
                    atom_idx_ligand, atom_idx_protein = atom_idx2, atom_idx1
                else:
                    print(pdbid, ligand, bond_type, 'error: atom index in plip result not in atom_idx_list')
                    print(atom_idx1, atom_idx2)
                    continue
                    # return None
                bond_list.append((bond_type + '_' + str(len(bond_list)), aa_chain, aa_name, aa_id, [atom_idx_protein],
                                  ligand_chain, ligand_name, ligand_id, [atom_idx_ligand]))
            elif bond_type == 'Hydrophobic Interactions':
                atom_idx_ligand, atom_idx_protein = int(lines[8]), int(lines[9])
                if atom_idx_ligand not in atom_idx_list:
                    continue
                elif atom_idx_ligand not in atom_idx_list:
                    print('error: atom index in plip result not in atom_idx_list')
                    print('Hydrophobic Interactions', atom_idx_ligand, atom_idx_protein)
                    continue
                    # return None
                bond_list.append((bond_type + '_' + str(len(bond_list)), aa_chain, aa_name, aa_id, [atom_idx_protein],
                                  ligand_chain, ligand_name, ligand_id, [atom_idx_ligand]))
            elif bond_type in ['pi-Stacking', 'pi-Cation Interactions']:
                # atom_idx_ligand_list = list(map(int, lines[11].split(',')))
                atom_idx_ligand_list = list(map(int, lines[12].split(',')))
                if len(set(atom_idx_ligand_list).intersection(set(atom_idx_list))) != len(atom_idx_ligand_list):
                    print(bond_type, 'error: atom index in plip result not in atom_idx_list')
                    print(atom_idx_ligand_list)
                    continue
                    # return None
                bond_list.append((bond_type + '_' + str(len(bond_list)), aa_chain, aa_name, aa_id, [], ligand_chain,
                                  ligand_name, ligand_id, atom_idx_ligand_list))
            elif bond_type == 'Salt Bridges':
                # atom_idx_ligand_list = list(set(map(int, lines[10].split(','))))
                atom_idx_ligand_list = list(set(map(int, lines[11].split(','))))
                if len(set(atom_idx_ligand_list).intersection(set(atom_idx_list))) != len(atom_idx_ligand_list):
                    print('error: atom index in plip result not in atom_idx_list')
                    print('Salt Bridges', atom_idx_ligand_list,
                          set(atom_idx_ligand_list).intersection(set(atom_idx_list)))
                    # return None
                    continue
                bond_list.append((bond_type + '_' + str(len(bond_list)), aa_chain, aa_name, aa_id, [], ligand_chain,
                                  ligand_name, ligand_id, atom_idx_ligand_list))
            elif bond_type == 'Halogen Bonds':
                atom_idx1, atom_idx2 = int(lines[11]), int(lines[13])
                if atom_idx1 in atom_idx_list and atom_idx2 in atom_idx_list:  # discard ligand-ligand interaction
                    continue
                if atom_idx1 in atom_idx_list:
                    atom_idx_ligand, atom_idx_protein = atom_idx1, atom_idx2
                elif atom_idx2 in atom_idx_list:
                    atom_idx_ligand, atom_idx_protein = atom_idx2, atom_idx1
                else:
                    print('error: atom index in plip result not in atom_idx_list')
                    print('Halogen Bonds', atom_idx1, atom_idx2)
                    return None
                bond_list.append((bond_type + '_' + str(len(bond_list)), aa_chain, aa_name, aa_id, [atom_idx_protein],
                                  ligand_chain, ligand_name, ligand_id, [atom_idx_ligand]))
            elif bond_type == 'Metal Complexes':
                pass
            else:
                print('bond_type', bond_type)
                print(lines)
                return None
    f.close()
    if len(bond_list) != 0:
        return bond_list


def get_atoms_from_pdb(ligand, pdbid):
    # from pdb protein structure, get ligand index list for bond extraction
    p = PDBParser()
    atom_idx_list = []
    atom_name_list = []

    structure = p.get_structure(pdbid, './pdb_files/' + pdbid + '.pdb')
    seq_dict = {}
    for model in structure:
        for chain in model:
            chain_id = chain.get_id()
            id_list = []
            for res in chain:
                if ligand == res.get_resname():
                    if res.get_id()[0] == ' ':
                        continue
                    for atom in res:
                        atom_idx_list.append(atom.get_serial_number())
                        atom_name_list.append(atom.get_id())
    if len(atom_idx_list) != 0:
        return atom_idx_list, atom_name_list
    else:
        return None, None


def get_seq(pdbid):
    p = PDBParser()
    structure = p.get_structure(pdbid, './pdb_files/' + pdbid + '.pdb')
    seq_dict = {}
    idx_to_aa_dict = {}
    for model in structure:
        for chain in model:
            chain_id = chain.get_id()
            if chain_id == ' ':
                continue
            seq = ''
            id_list = []
            for res in chain:
                if res.get_id()[0] != ' ' or res.get_id()[2] != ' ':  # remove HETATM
                    continue
                try:
                    seq += three_to_one(res.get_resname())
                    idx_to_aa_dict[chain_id + str(res.get_id()[1]) + res.get_id()[2].strip()] = three_to_one(
                        res.get_resname())
                except:
                    print('unexpected aa name', res.get_resname())
            seq_dict[chain_id] = seq
    return seq_dict, idx_to_aa_dict


def get_interact_residue(idx_to_aa_dict, bond_list):
    interact_residue = []
    for bond in bond_list:
        if bond[1] + str(bond[3]) not in idx_to_aa_dict:
            continue
        aa = idx_to_aa_dict[bond[1] + str(bond[3])]
        assert three_to_one(bond[2]) == aa
        interact_residue.append((bond[1] + str(bond[3]), aa, bond[0]))
    if len(interact_residue) != 0:
        return interact_residue
    else:
        return None


if __name__ == '__main__':
    no_valid_ligand = 0
    no_such_ligand_in_pdb_error = 0
    no_interaction_detected_error = 0
    no_ideal_pdb_error = 0
    empty_atom_interact_list = 0
    protein_seq_error = 0

    i = 0
    interaction_dict = {}
    pdbid_list = get_pdbid_list()
    pdbid_to_ligand = get_pdbid_to_ligand()
    for pdbid in pdbid_list:
        i += 1
        print(i, pdbid)
        if pdbid not in pdbid_to_ligand:
            no_valid_ligand += 1
            continue
        ligand = pdbid_to_ligand[pdbid]
        print(ligand)

        # get bond
        if not os.path.exists('./pdb_files/' + pdbid + '.pdb'):
            print(pdbid + ' is not exist!')
            continue
        atom_idx_list, atom_name_list = get_atoms_from_pdb(ligand, pdbid)  # for bond atom identification
        if atom_idx_list is None:
            no_such_ligand_in_pdb_error += 1
            print('no such ligand in pdb', 'pdbid', pdbid, 'ligand', ligand)
            continue
        bond_list = get_bonds(pdbid, ligand, atom_idx_list)
        if bond_list is None:
            print('empty bond list: pdbid', pdbid, 'ligand', ligand, 'atom_idx_list', len(atom_idx_list))
            no_interaction_detected_error += 1
            continue

        if len(atom_idx_list) == 0:
            empty_atom_interact_list += 1
            continue

        # get sequence interaction information
        seq_dict, idx_to_aa_dict = get_seq(pdbid)
        # distance_matrix, dihedral_array, torsion_mask = output_distance_matrix_and_dihedral_angle(pdbid)
        interact_residue_list = get_interact_residue(idx_to_aa_dict, bond_list)
        if interact_residue_list is None:
            protein_seq_error += 1
            continue

        interaction_dict[pdbid + '_' + ligand] = {}
        interaction_dict[pdbid + '_' + ligand]['sequence'] = seq_dict
        interaction_dict[pdbid + '_' + ligand]['residue_interact'] = interact_residue_list


    print('interaction_dict', len(interaction_dict))
    print('no_valid_ligand error', no_valid_ligand)
    print('no_such_ligand_in_pdb_error', no_such_ligand_in_pdb_error)
    print('no_interaction_detected_error', no_interaction_detected_error)
    print('no_ideal_pdb_error', no_ideal_pdb_error)
    print('empty_atom_interact_list', empty_atom_interact_list)
    print('protein_seq_error', protein_seq_error)

    with open('./fileprocessing/output/out4_interaction_dict', 'wb') as f:
        pickle.dump(interaction_dict, f, protocol=0)