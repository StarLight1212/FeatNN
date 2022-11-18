import os
import urllib.request as re
import requests
import urllib.error


def down_load_ligand_pdbfile():
    with open("./fileprocessing/output/out1.2_pdbbind_crawl_ligand.txt", "r") as f:
        lines = f.readlines()
        lines_count = lines.__len__()
        for count in range(lines_count):
            line_split = lines[count]
            print(line_split)
            line_split_num = line_split.__len__()
            try:
                file_path = os.path.join(r"./pdb_files/ligand/", line_split.split("https://files.rcsb.org/ligands/download/")[1].split(".pdb\n")[0] + ".pdb")
                print(file_path.split("./pdb_files/ligand/")[1])
                re.urlretrieve(line_split, file_path)
            except (urllib.error.HTTPError, urllib.error.URLError) as e:
                print("HTTPError, URLError")
                pass
            except Exception as e:
                print("e")


def down_load_complex_pdbfile():
    with open("./fileprocessing/output/out1.2_pdbbind_crawl_complex.txt", "r") as f:
        lines = f.readlines()
        lines_count = lines.__len__()
        path = r"./pdb_files/complex/"
    
        for count in range(lines_count):
            line_split = lines[count]
            print(line_split)
            # print(line_split.split("https://files.rcsb.org/download/")[1].split(".pdb\n")[0])
            line_split_num = line_split.__len__()
            if line_split_num == 1:
                folder = line_split[0]
                dir_down = os.path.join(path, folder)
                if not os.path.exists(dir_down):
                    os.makedirs(dir_down.replace('\n', ""))
            try:
                file_path = os.path.join(r"./pdb_files/complex/", line_split.split("https://files.rcsb.org/download/")[1].split(".pdb\n")[0] + ".pdb")
                print(file_path.split("./pdb_files/complex/")[1])
                re.urlretrieve(line_split, file_path)
    
            except (urllib.error.HTTPError, urllib.error.URLError) as e:
                print("HTTPError, URLError")
                pass
            except Exception as e:
                print("e")


if __name__ == '__main__':
    # Crawl complex structure data
    down_load_complex_pdbfile()
    
    # Crawl ligand data
    down_load_ligand_pdbfile()