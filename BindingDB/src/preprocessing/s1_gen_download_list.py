import os

def get_uniprot_id_list(path: str):
    bindingDB_id_set = set()
    with open(path, 'r') as f:
        for line in f.readlines():
            bindingDB_id_set.add(line.strip().split(',')[1])
    f.close()
    print('pdbid_list', len(bindingDB_id_set))
    return bindingDB_id_set


def write_txt(pdb_list):
    with open('./output_files/pdbid_set_set_l.txt', 'w') as f:
        for pdbid in pdb_list:
            f.write(pdbid + '\n')
    f.close()

    with open('./output_files/pdbid_wget_set_l.txt', 'w') as f:
        for pdbid in pdb_list:
            f.write('https://files.rcsb.org/download/' + pdbid + '.pdb\n')
    f.close()


if __name__ == '__main__':
    path = '../../data/bindingdb_data_ec.csv'
    pdbid_code = get_uniprot_id_list(path)
    print(pdbid_code)
    print('Length of PDBid: ', len(pdbid_code))
    write_txt(pdbid_code)



