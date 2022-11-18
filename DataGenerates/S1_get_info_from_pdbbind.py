# Extract protein-ligand complex from PDBBindv2020

def get_pdbid_list():
	pdbid_list = []
	with open('./fileprocessing/pdbbind_files/INDEX_general_PL.2020') as f:
		for line in f.readlines():
			if line[0] != '#':
				pdbid_list.append(line.strip().split()[0])
	print('pdbid_list',len(pdbid_list))
	return pdbid_list


# ligand pdb file with index of pdb id
def get_pdbid_to_ligand():
	pdbid_to_ligand = {}
	with open('./fileprocessing/pdbbind_files/INDEX_general_PL.2020') as f:
		for line in f.readlines():
			if line[0] != '#':
				ligand = line.strip().split('(')[1].split(')')[0]
				if '-mer' in ligand:
					continue
				elif '/' in ligand:
					ligand = ligand.split('/')[0]
				if len(ligand) != 3:
					#print(line[:4], ligand)
					continue
				pdbid_to_ligand[line[:4]] = ligand
	print('pdbid_to_ligand',len(pdbid_to_ligand))
	return pdbid_to_ligand


if __name__ == '__main__':
	pdbid_list = get_pdbid_list()
	# Obtain PDB Code (Protein-Ligand Complex)
	fw = open('./fileprocessing/output/out1.2_pdbid_list.txt','w')
	for pdbid in pdbid_list:
		fw.write(pdbid+'\n')
	fw.close()
	
	# Obtain Downloading web Link to Crawl Complex Structure Data in S2
	fw = open('./fileprocessing/output/out1.2_pdbbind_crawl_complex.txt','w')
	for pdbid in pdbid_list:
		fw.write('https://files.rcsb.org/download/'+pdbid+'.pdb\n')
	fw.close()
	
	# Obtain Ligand web Link to Crawl Ligand Data in S2
	pdbid_to_ligand = get_pdbid_to_ligand()
	fw = open('./fileprocessing/output/out1.2_pdbbind_crawl_ligand.txt', 'w')
	for pdbid in pdbid_list:
		if pdbid in pdbid_to_ligand:
			ligand = pdbid_to_ligand[pdbid]
			fw.write('https://files.rcsb.org/ligands/download/'+ligand+'_ideal.pdb\n')
	fw.close()
