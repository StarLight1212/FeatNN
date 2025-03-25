import os
import urllib.request as re
import requests
import urllib.error


def down_load_data():
    with open("./output_files/pdbid_wget_set_l.txt", "r") as f:
        lines = f.readlines()[70:800]
        lines_count = lines.__len__()
        for count in range(lines_count):
            line_split = lines[count]
            print(line_split)
            file_path = os.path.join(r"../../data/pdb_files/", line_split.split("https://files.rcsb.org/download/")[1].split(".pdb\n")[0] + ".pdb")
            print(file_path.split("../../data/pdb_files/")[1])
            try:
                re.urlretrieve(line_split, file_path)
            except (urllib.error.HTTPError, urllib.error.URLError) as e:
                print("HTTPError, URLError")
                pass
            except Exception as e:
                print("Other Error need to be find!")
    f.close()


if __name__ == '__main__':
    down_load_data()