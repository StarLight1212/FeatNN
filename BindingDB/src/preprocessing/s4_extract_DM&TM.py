import os
import Bio.PDB
from Bio.PDB.Polypeptide import three_to_one
import numpy as np
import pickle


def get_pdbid_list(file_path):
    pdbid_list = []
    with open(file_path) as f:
        for line in f.readlines():
            pdbid_list.append(line.strip())
    print('pdbid_list', len(pdbid_list))
    return pdbid_list


def get_pdbid_to_ligand(file_path):
    pdbid_to_ligand = {}
    with open(file_path) as f:
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


def get_res_coord(structure):
    try:
        model = structure[0]
    except KeyError as KE:
        print("KeyError: PDBid->{}".format(pdbid))
        return None, None
    # model = structure
    distance_matrix_dict = {}
    seq_dict = {}
    distance_list = []

    for chain in model:
        # chain.get_id() -> 'E'
        seq_length = 0
        seq = ''
        for res in chain:
            # res -> <Residue ILE het=  resseq=1 icode= >
            # res.get_id() -> (' ', 1, ' ')
            # res.get_resname() -> ALA; PRO; VAL; ASP; HIS
            # remove HETATM
            if res.get_id()[0] != ' ' or res.get_id()[2] != ' ':
                continue
            try:
                seq += three_to_one(res.get_resname())
            except:
                print('unexpected aa name', res.get_resname())

            if res.get_resname() == 'GLY':
                seq_length += 1
                for atom in res:
                    if atom.get_name() == 'CA':
                        distance_list.append(atom.coord)
            else:
                # atom.get_name -> <bound method Atom.get_name of <Atom CE>>
                # atom.get_name() -> 'CE'; 'CA'; 'N';
                # atom.coord -> [34.204 40.284 -1.02 ]
                seq_length += 1
                for atom in res:
                    if atom.get_name() == 'CB':
                        # print(atom.get_name())
                        distance_list.append(atom.coord)
                    # else:
                    #     print(res.get_resname())

        # if chain.get_id() == chain_id:
        if len(distance_list) != len(seq):
            print('-----------------')
            return None, None
        # print(len(distance_list), 'UUU', len(seq))
        # assert len(distance_list) == len(seq)
        seq_dict[chain.get_id()] = seq
        distance_matrix_dict[chain.get_id()] = distance_list[:]
        distance_list.clear()
    return distance_matrix_dict, seq_dict


def calc_dist_matrix(distance_matrix_dict, min_distance=3.25, max_distance=50.75, step=1.25):
    # distance_matrix = np.zeros((seq_length, seq_length, 39), np.int8)
    distance_dict = {}
    for chain, coord in distance_matrix_dict.items():
        seq_length = len(distance_matrix_dict[chain])
        distance_matrix = np.zeros((seq_length, seq_length), np.int8)
        for row, res1 in enumerate(coord):
            for column, res2 in enumerate(coord):
                diff_vector = res1 - res2
                # euclidean_distance
                euclidean_distance = np.sqrt(np.sum(diff_vector * diff_vector))
                if euclidean_distance < min_distance:
                    # print(euclidean_distance)
                    continue
                elif euclidean_distance > max_distance:
                    distance_bins = 38
                else:
                    distance_bins = int(
                        (euclidean_distance - min_distance) // step)
                # distance_matrix[row, column, distance_bins] = euclidean_distance
                # distance_matrix[row, column, distance_bins] = 1.0
                distance_matrix[row, column] = distance_bins + 1
            # print(euclidean_distance)
        distance_dict[chain] = distance_matrix
        # print(distance_dict)
    return distance_dict


def get_phi_psi_torsion(structure):
    phi_torsion_dict, psi_torsion_dict = {}, {}
    for model in structure:
        for chain in model:
            seq = ''
            phi_torsion = []
            psi_torsion = []
            for res in chain:
                # remove HETATM
                if res.get_id()[0] != ' ' or res.get_id()[2] != ' ':
                    continue
                try:
                    seq += three_to_one(res.get_resname())
                except:
                    print('unexpected aa name', res.get_resname())

            # poly = Bio.PDB.Polypeptide.Polypeptide(chain)
            # polypeptides = Bio.PDB.PPBuilder().build_peptides(chain)[0]
            polypeptides = Bio.PDB.CaPPBuilder().build_peptides(chain)
            for poly in polypeptides:
                # print(poly.get_phi_psi_list())
                phi_psi_torsion = poly.get_phi_psi_list()
                # print("len phi_psi_torsion",len(phi_psi_torsion))
                for phi, psi in phi_psi_torsion:
                    if phi == None and psi == None:
                        phi = np.nan
                        psi = np.nan
                    elif psi == None:
                        psi = np.nan
                    elif phi == None:
                        phi = np.nan
                    phi_torsion.append(phi)
                    psi_torsion.append(psi)
                    # print('len phi_torsion: ', len(phi_torsion))
                    # print('len psi_torsion:', len(psi_torsion))
            assert len(phi_torsion) == len(psi_torsion)
            if len(phi_torsion) != len(seq):
                print('Discard Tor-Angle')
                return None, None
            phi_torsion_dict[chain.get_id()] = phi_torsion[:]
            psi_torsion_dict[chain.get_id()] = psi_torsion[:]
            phi_torsion.clear()
            psi_torsion.clear()
    return phi_torsion_dict, psi_torsion_dict


def cal_sine_or_cosine(torsion_dict, type):
    sine_dict, cosine_dict = {}, {}
    for chain, torsion_list in torsion_dict.items():
        if type == 'phi':
            sine = np.sin(torsion_list)
            cosine = np.cos(torsion_list)
        elif type == 'psi':
            # print(torsion_list)
            sine = np.sin(torsion_list)
            cosine = np.cos(torsion_list)
        else:
            raise TypeError("Type parameter got wrong input content!")
        sine_dict[chain] = sine
        cosine_dict[chain] = cosine
    return sine_dict, cosine_dict


def get_torsion_mask(torsion_dict):
    torsion_mask = {}
    for chain, torsion_list in torsion_dict.items():
        mask = np.ones((len(torsion_list), 4))
        for row, torsion in enumerate(torsion_list):
            for column, value in enumerate(torsion):
                if value == 0.:
                    mask[row, column] = 0.0
        torsion_mask[chain] = mask
    return torsion_mask


def output_distance_matrix_and_dihedral_angle(pdb_id):
    try:
        structure = Bio.PDB\
            .PDBParser()\
            .get_structure(pdb_id,
                           '../../data/pdb_files/' + pdb_id + '.pdb')
    except Exception as e:
        print("NO PDB File!")
        return None, None, None
    distance_matrix_dict, seq_dict = get_res_coord(structure)
    if distance_matrix_dict is None:
        return None, None, None
    # Distance tensor (Nres, Nres, 39)
    distance_matrix = calc_dist_matrix(distance_matrix_dict)

    phi_torsion_dict, psi_torsion_dict = get_phi_psi_torsion(structure)
    if phi_torsion_dict is None:
        return None, None, None
    phi_sine_dict, phi_cosine_dict = cal_sine_or_cosine(
        phi_torsion_dict, 'phi')
    psi_sine_dict, psi_cosine_dict = cal_sine_or_cosine(
        psi_torsion_dict, 'psi')
    # Torsion angle (Nres, 4)
    dihedral_dict = {}
    for chain in phi_sine_dict:
        dihedral_array = np.stack(
            (phi_sine_dict[chain], psi_sine_dict[chain], phi_cosine_dict[chain], psi_cosine_dict[chain])).transpose()
        dihedral_dict[chain] = np.nan_to_num(dihedral_array)
    # torsion_mask = get_torsion_mask(dihedral_dict)

    return seq_dict, distance_matrix, dihedral_dict


if __name__ == '__main__':
    no_valid_ligand = 0
    no_such_ligand_in_pdb_error = 0
    no_interaction_detected_error = 0
    no_ideal_pdb_error = 0
    empty_atom_interact_list = 0
    protein_seq_error = 0
    err = 0
    i = 0
    valid_pdb_list = []
    pdbid_list = get_pdbid_list('./output_files/pdb_id.txt')

    output_dist_mtx_folder = r"../../data/DT/"
    # 自动创建文件夹
    if not os.path.exists(output_dist_mtx_folder):
        os.makedirs(output_dist_mtx_folder)

    for pdbid in pdbid_list[:]:
        i += 1
        print(i, pdbid)

        # get bond
        if not os.path.exists('../../data/pdb_files/' + pdbid + '.pdb'):
            print(pdbid + ' is not exist!')
            continue

        seq_dict, distance_matrix, dihedral_array = output_distance_matrix_and_dihedral_angle(
            pdbid)
        if distance_matrix is None:
            err += 1
            print("GAP_n", err)
            continue
        output_dict = {
            "sequence": seq_dict,
            "distance_matrix": distance_matrix,
            "dihedral_array": dihedral_array,
        }
        valid_pdb_list.append(pdbid)
        print(output_dict)
        np.save(
            open(f"{output_dist_mtx_folder}/{pdbid}.npy", 'wb'),
            output_dict
        )
    print("Length of valid pdbid is: ", len(valid_pdb_list))
    np.save('../../data/valid_pdbid.npy', valid_pdb_list)