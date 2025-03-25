import os
from Bio.PDB import PDBParser, CaPPBuilder
from Bio.PDB.Polypeptide import three_to_one
import numpy as np
import pickle
from tqdm import tqdm
import multiprocessing as mp
from functools import partial


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
        print(f"KeyError: PDBid->{structure.id}")
        return None, None

    distance_matrix_dict = {}
    seq_dict = {}
    distance_list = []

    for chain in model:
        seq_length = 0
        seq = ''
        for res in chain:
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
                seq_length += 1
                for atom in res:
                    if atom.get_name() == 'CB':
                        distance_list.append(atom.coord)

        if len(distance_list) != len(seq):
            print('-----------------')
            return None, None

        seq_dict[chain.get_id()] = seq
        distance_matrix_dict[chain.get_id()] = distance_list[:]
        distance_list.clear()
    return distance_matrix_dict, seq_dict


def calc_dist_matrix(distance_matrix_dict, min_distance=3.25, max_distance=50.75, step=1.25):
    distance_dict = {}
    for chain, coord in distance_matrix_dict.items():
        seq_length = len(distance_matrix_dict[chain])
        distance_matrix = np.zeros((seq_length, seq_length), np.int8)
        for row, res1 in enumerate(coord):
            for column, res2 in enumerate(coord):
                diff_vector = res1 - res2
                euclidean_distance = np.sqrt(np.sum(diff_vector * diff_vector))
                if euclidean_distance < min_distance:
                    continue
                elif euclidean_distance > max_distance:
                    distance_bins = 38
                else:
                    distance_bins = int((euclidean_distance - min_distance) // step)
                distance_matrix[row, column] = distance_bins + 1
        distance_dict[chain] = distance_matrix
    return distance_dict


def get_phi_psi_torsion(structure):
    phi_torsion_dict, psi_torsion_dict = {}, {}
    for model in structure:
        for chain in model:
            seq = ''
            phi_torsion = []
            psi_torsion = []
            for res in chain:
                if res.get_id()[0] != ' ' or res.get_id()[2] != ' ':
                    continue
                try:
                    seq += three_to_one(res.get_resname())
                except:
                    print('unexpected aa name', res.get_resname())

            polypeptides = CaPPBuilder().build_peptides(chain)
            for poly in polypeptides:
                phi_psi_torsion = poly.get_phi_psi_list()
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


def output_distance_matrix_and_dihedral_angle(pdb_id, pdb_dir='../../data/pdb_files/'):
    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure(pdb_id, os.path.join(pdb_dir, f"{pdb_id}.pdb"))
    except Exception:
        return None, None, None

    distance_matrix_dict, seq_dict = get_res_coord(structure)
    if distance_matrix_dict is None:
        return None, None, None

    distance_matrix = calc_dist_matrix(distance_matrix_dict)

    phi_torsion_dict, psi_torsion_dict = get_phi_psi_torsion(structure)
    if phi_torsion_dict is None:
        return None, None, None

    phi_sine_dict = {chain: np.sin(torsion) for chain, torsion in phi_torsion_dict.items()}
    phi_cosine_dict = {chain: np.cos(torsion) for chain, torsion in phi_torsion_dict.items()}
    psi_sine_dict = {chain: np.sin(torsion) for chain, torsion in psi_torsion_dict.items()}
    psi_cosine_dict = {chain: np.cos(torsion) for chain, torsion in psi_torsion_dict.items()}

    dihedral_dict = {
        chain: np.nan_to_num(np.stack((
            phi_sine_dict[chain],
            psi_sine_dict[chain],
            phi_cosine_dict[chain],
            psi_cosine_dict[chain]
        )).T)
        for chain in phi_sine_dict
    }

    return seq_dict, distance_matrix, dihedral_dict


def process_pdb(pdb_id, output_dir, pdb_dir):
    if not os.path.exists(os.path.join(pdb_dir, f"{pdb_id}.pdb")):
        print(f"{pdb_id} is not exist!")
        return None

    seq_dict, distance_matrix, dihedral_array = output_distance_matrix_and_dihedral_angle(pdb_id, pdb_dir)
    if distance_matrix is None:
        return None

    output_dict = {
        "sequence": seq_dict,
        "distance_matrix": distance_matrix,
        "dihedral_array": dihedral_array,
    }

    np.save(os.path.join(output_dir, f"{pdb_id}.npy"), output_dict)
    return pdb_id


if __name__ == '__main__':
    err = 0
    valid_pdb_list = []

    pdb_dir = "../../data/pdb_files/"
    output_dist_mtx_folder = "../../data/DT/"
    os.makedirs(output_dist_mtx_folder, exist_ok=True)

    pdbid_list = get_pdbid_list('./output_files/pdb_id.txt')

    num_processes = mp.cpu_count() - 1
    pool = mp.Pool(processes=num_processes)

    process_func = partial(process_pdb, output_dir=output_dist_mtx_folder, pdb_dir=pdb_dir)

    with tqdm(total=len(pdbid_list), desc="Processing PDB files") as pbar:
        results = []
        for result in pool.imap_unordered(process_func, pdbid_list):
            if result is not None:
                valid_pdb_list.append(result)
            else:
                err += 1
            pbar.update(1)

    pool.close()
    pool.join()

    print(f"Length of valid pdbid is: {len(valid_pdb_list)}")
    print(f"Number of errors: {err}")
    np.save('../../data/valid_pdbid.npy', valid_pdb_list)