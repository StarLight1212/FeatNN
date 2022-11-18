def get_uniprot_id_from_index():
    uniprotid_set = set()
    with open('./fileprocessing/pdbbind_index/INDEX_general_PL_name.2020', 'r') as f:
        for line in f.readlines():
            if line[0] != '#':
                lines = line.strip().split('  ')
                if lines[2] != '------':
                    uniprotid_set.add(lines[2])
    f.close()
    print('Uniprotid_set step1', len(uniprotid_set))

    with open('./fileprocessing/output/out1.4_pdb_uniprot_mapping.tab', 'r') as f:
        for line in f.readlines()[1:]:
            lines = line.split('\t')
            uniprotid_set.add(lines[1])
    f.close()
    print('Uniprotid_set step2', len(uniprotid_set))
    return uniprotid_set


if __name__ == '__main__':
    uniprotid_set = get_uniprot_id_from_index()
    uniprotid_list = list(uniprotid_set)
    print(uniprotid_list)
    fw = open('./fileprocessing/output/out1.5_uniprotid_list.txt', 'w')
    for uniprotid in uniprotid_list:
        fw.write(uniprotid + '\n')
    fw.close()