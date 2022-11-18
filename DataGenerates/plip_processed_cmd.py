import os
import sys


def get_file_name(file_dir):
    for root, dirs, files in os.walk(file_dir):
        print(files)
    files = [i[:-4] for i in files]
    return files


if __name__ == '__main__':
    sys.path.append(r"./plip-master/plip/")
    print(sys.path)

    file_name = get_file_name(file_dir="./pdb_files/")
    for pdb_id in file_name:
        os.system(r'python ./plip-master/plip/plipcmd.py -f ./pdb_files/'+pdb_id+' -t --name '+pdb_id+'_output')