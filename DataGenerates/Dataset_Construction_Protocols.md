# Standard Datasets Construction Protocols


**This FeatNN project uses the dataset construction protocols below based on the PDBbind-v2020 database.
<u>*Apart from the data in computer vision (CV) and natural language process (NLP), datasets generated from biology experiments have unneglectable biology information behind them.*</u> Therefore to avoid the notorious data leakage caused by such reasons, we should preprocess the data instead of directly splitting the data into the train, valid, and test datasets according to a certain ratio.
Here, we also want to provide a standard method for how constructing and processing the dataset in the domain of compound-protein affinity prediction.**



**INPUT:** Files and related information are downloaded and processed from the databases of PDB and PDBbind

**OUTPUT:** train, valid and test datasets includes the information of: 1) protein 3D structure information,
                    2) protein sequence information, 3) compound smiles information, and 4) protein-compound binding affinity


**The main construction process will be finished in the folder "DataGenerates".**
---------------------

> **Step1: Download data from PDBbind and extract initial information from the file of "INDEX_general_PL_name.2020"**
> 1) PDBbind data: Login(if do not have an account, please register one at first) and download PDBbind (v2020, general set, while the refined data enjoys only a small amount of data for training) data from <http://www.pdbbind.org.cn/download.php>.
Unzip the download package, put the protein-ligand complex data into "./fileprocessing/pdbbind_files/", and put the index files into "./fileprocessing/pdbbind_index/".
> 2) PDB data: Then, download the SDF file containing the ligand graphs from <http://ligand-expo.rcsb.org/ld-download.html>. Put the file "./fileprocessing/Components-pub.sdf" under the current folder.
Run **"S1_get_info_from_pdbbind.py"** to extract PDB IDs and ligand IDs of all protein-ligand complex from "./pdbbind_index/INDEX_general_PL.2020", and generate the URLs into the files "./fileprocessing/output/out1.2_pdbbind_crawl_complex.txt"
and "./fileprocessing/output/out1.2_pdbbind_crawl_ligand.txt" for wget. This also generates "./fileprocessing/output/output1.2_pdbid_list.txt" for PDB ID to UniProt ID mapping.
Note: Here, only single ligands with standard PDB ligand IDs will have remained. That is, ligand names like e.g. "5-mer", "SIA-GAL" and "0LU/0LW" will be discarded.
> 3) Create the path "./pdb_files/" under current directory, and then run **"S2_download_PDB_file.py"** to download all pdb files from PDB: <br>
    - function - 'down_load_complex_pdbfile' is applied to crawl complex data from Protein Data Bank <br>
    - function - 'down_load_ligand_pdbfile' is applied to crawl ligand data from Protein Data Bank
> 4) Upload the "out1.2_pdbid_list.txt" to <https://www.uniprot.org/uploadlists/> (select options from "PDB" to "UniProtKB") and download "Tab-separated" mapping result.
Name the file as "out1.4_pdb_uniprot_mapping.tab", and put the file under folder of "./fileprocessing/output/".
> 5) Run **"S3_get_uniprotID_list.py"** to get all possible UniProt IDs that might be used.
The output file will be "out1.5_uniprotid_list.txt", and put the file under folder of "./fileprocessing/output/".
> 6) Upload the "out1.5_uniprotid_list.txt" to <https://www.uniprot.org/uploadlists/> (select options from "UniProtKB AC/ID" to "UniProtKB"),
 and download the file with fata format which contains the UniProt sequences information.
 Rename the file as "out1.6_pdbbind_seqs.fasta", , and put the file under folder of "./fileprocessing/output/".
 And also download the "Tab-separated" result and name it as "out1.6_uniprot_uniprot_mapping.tab" from the bottom of the web site, because some old-version Uniprot IDs need to be updated.

-----------------------

> **Step2: Extract the information containing the protein sequence information, PDB ID, Uniprot ID and smiles sequence and affinity values from the index file of PDBbind-v2020.** <br>
> Run **"S4_generate_data_file_output.py"** and the code will output "out2_pdbbind_data_datafile.tsv". The columns in the datafile are:<br>
	- PDB ID of the complex<br>
	- Uniprot ID of the protein<br>
	- PDB ID of the ligand<br>
	- Inchi of the ligand<br>
	- Sequence of the protein<br>
	- Measurement (Ki, Kd, IC50)<br>
	- Affinity values (-log10 Ki/Kd/IC50)<br>

-----------------------

> **Step3: Based on the pdb file download from the PDBbind-v2020, we extract the sequence information and constructed the discrete distance matrix(DDM) and torsion matrix(TorM) from the PDB file of the protein**<br>
> Run: **"S5_val_DDM_Seq_TorM.py"**, the information of each protein will be recorded as: <br>
> output_dict = {<br>
             "sequence": sequence,<br>
            "distance_matrix": DDM,<br>
            "dihedral_array": TorM,<br>
        }<br>
> The sequence and structure information (DDM and TorM) of each protein will be processed and recorded in the folder of "./DDM_TM/".

-----------------------

> **Step4: Based on PLIP, we calculate the non-covalent interactions between proteins and ligands**<br>
> Here, we really appreciate for the code from Melissa F. Adasme et al(<https://github.com/ssalentin/plip/>) from the Dresden University in Deutschland.<br>
> Extract the non-covalent interactions by using 1) source code of PLIP (<https://github.com/ssalentin/plip/>) or 2) the website server (<https://plip-tool.biotec.tu-dresden.de/>)<br>
> Here, we recommend the method of 1) PLIP source code in github:<br>
> 1) Firstly, download the plip project (plip-master) under current folder of "DataGenerates".<br>
> 2) Secondly, run the script of **"plip_processed_cmd.py"** to calculate the non-covalent interactions between proteins and ligands,
    the command of "python ./plip-master/plip/plipcmd.py -f ./pdb_files/'+pdb_id+' -t --name '+pdb_id+'_output" will be execute for all PDB file of complexes.<br>
> 3) Finally, all processed file will be automatically saved in the folder of "./plip_result_all_set/".<br>
>**Note:** command for using PLIP: python plipcmd.py -f xxx.pdb -t --name xxx_output

-----------------------

> **Step5: extract the interaction information from PLIP output**<br>
> Run **"S6_get_interaction.py"** to read raw ".txt" format plip output, map ligand interaction sites<br>
> **Output:** output the file "out4_interaction_dict" into the folder of "./fileprocessing/output/", which is a dictionary packed by pickle.

-----------------------

> **Step6: extract the pocket positions information from the file downloaded from PDBbind (folder: "./fileprocessing/pdbbind_files/")**<br>
> Run **"S7_get_pocket_seq_info.py"** to get pocket information from PDBbind files.<br>
> **Output:** output the file "out5_pocket_dict" into the folder "./fileprocessing/output/".

-----------------------

> **Step7: Before constructing a protein similarity matrix, this step is mainly about applying the multi-sequence alignment (MSA) between different protein sequence, which is vital and pivotal for
 the protein cluster in S11. Sequence alignment here between the sequence from the complex structures and the UniProt sequences.**<br>
> Here, we really appreciate for the code from Shuya Li et al(<https://github.com/lishuya17/MONN>).<br>
> 1) Run **"S8_get_fasta_seq_for_msa.py"** to prepare the .fasta files for sequence alignment
> **Output:** output the files of "out6.1_query_pdb.fasta", "out6.1_target_uniprot_pdb.fasta", "out6.2_query_pdbbind.fasta" and "out6.2_target_uniprot_pdbbind.fasta"
> into both folders of "./fileprocessing/output/", "./Smith-Waterman-MSA/"
> 2) For sequence alignment, we utilized the code in the folder "Smith-Waterman-MSA" in the Linux OS (recommend CentOS or Ubuntu, python interpreter of Python2.7)<br>
>> First, add the path of libssw.so into LD_LIBRARY_PATH:<br>
>>>         export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD/Smith-Waterman-MSA<br>
>>
>> Second, copy files into the Smith-Waterman-MSA directory:<br>
>>>         cp out6.1_query_pdb.fasta ./Smith-Waterman-MSA
>>>         cp out6.1_target_uniprot_pdb.fasta ./Smith-Waterman-MSA
>>>         cp out6.2_query_pdbbind.fasta ./Smith-Waterman-MSA
>>>         cp out6.2_target_uniprot_pdbbind.fasta ./Smith-Waterman-MSA
>>
>> Third, run the alignment scripts and recorded th results of PDB and PDBbind MSA:<br>
>>>         cd Smith-Waterman-MSA
>>>         python pyssw_pairwise.py -c -p out6.1_query_pdb.fasta out6.1_target_uniprot_pdb.fasta  > out6.3_pdb_align.txt
>>>         python pyssw_pairwise.py -c -p out6.2_query_pdbbind.fasta out6.2_target_uniprot_pdbbind.fasta  > out6.4_pdbbind_align.txt
>>>         cd ..
>>
>> Finally, we obtain the file of 1) "protein_sequence_alignment.txt", 2) "out6.3_pdb_align.txt", and 3) "out6.4_pdbbind_align.txt". Then, we put all of these files into the folder "./fileprocessing/output/".

-----------------------

> **Step8: Combine the alignment result and the interaction result to get the final interaction dictionary**<br>
> Run **"S9_get_interaction_information_with_chain.py"**<br>
> **Output:** output all the files "FEATNN_final_interaction_dict", "chain_dict" and "valid_pdbid" into the folder "./fileprocessing/output/".

-----------------------

> **Step9: Based on the MSA results generated from the Step7**<br>
> Here, we calculate the protein similarity matrix from the algorithm of<br>
>>         protein_sim_mat[i][j] = identity_score[i]/identity_score[i] (if i == protein_list[j])
>>         protein_sim_mat[i][j] = paired_score[i+"_"+protein_list[j]]/math.sqrt(identity_score[i]*identity_score[protein_list[j]]) (if i != protein_list[j])".
>Run the script of **"S10_geenrate_msa_sim_mat.py"**, and finally, we will obtain a protein similarity matrix recording the similarity score between each protein in the PDBbind-v2020 database.
>**Output:** output both the files "pdbbind_protein_sim_mat.npy" and "pdbbind_protein_list.npy" into the folder "./fileprocessing/output/".

-----------------------

> **Step10: Preprocessing and cluster the final data with "compound-clustered" and "protein-clustered" strategies (details in the paper of FeatNN) according to the similarity among compounds and proteins.**<br>
> Here we respectively process the dataset with the measurement of "KIKD" and "IC50" by setting the variable of "MEASURE" as "KIKD" and "IC50".<br>
> Run the script **"S11_preprocessing_and_cluster.py"**, we cluster the dataset from the threshold of 0.3, 0.4, 0.5, and 0.6, with "compound-clustered" and "protein-clustered" strategies.<br>
> **Output:** Finally, all processed data will be saved in the folder "./preprocessing/IC50/" or "./preprocessing/KIKD/" according to the binding affinity measurement of "IC50" and "KIKD".<br>

-----------------------

> **S11: Generate the train, valid and test dataset with the command of:**<br>
>>     python S12_trn_valid_tst_construction.py --measures=IC50 --setting=ComClu --threshold=0.3 --n_fold=5" for the IC50 measurement, compound-clustered strategy with the threshold of 0.3, and with 5-fold cross validation
>>     python S12_trn_valid_tst_construction.py --measures=KIKD --setting=ComClu --threshold=0.4 --n_fold=5" for the KIKD measurement, compound-clustered strategy with the threshold of 0.4, and with 5-fold cross validation
>>     python S12_trn_valid_tst_construction.py --measures=KIKD --setting=ProtClu --threshold=0.4 --n_fold=5" for the KIKD measurement, protein-clustered strategy with the threshold of 0.4, and with 5-fold cross validation
>>     ...
>
> **Output:** the final data will be saved in the folder "./Train_test_data/", and you can move the file into the folder '../Datasets/' for training the FeatNN.

-----------------------

***Finally, here, we again appreciate for the code from Melissa F. Adasme et al (<https://github.com/ssalentin/plip/>) from the Dresden University in Deutschland and the code from Shuya Li et al (<https://github.com/lishuya17/MONN>) sincerely.***
