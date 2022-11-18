## Download datasets

-------------

#### Download the datasets of proteins and compounds information preprocessed from PDBbind

1) Download the datasets (include structure information in "PDBbind_struct" and compound graph, 
protein sequence and affinity information in "Traindata") generated from general set of PDBbind-v2020 with the measurement of "IC50" and "KIKD" from the URL link of 
<https://drive.google.com/drive/folders/1GnnwBOFlewZBfSodpfz3hb04D9oTKsT5?usp=sharing>

2) And then, please unzip the file under the folder of "Datasets" in this project before training. 
(***Please make sure the paths of "FeatNN/Datasets/PDBbind_Struct/", and "FeatNN/Datasets/Traindata/" are exist in this project.***)


#### Download the datasets of processed train. valid and test datasets

1) Download the file "DataProcessed" which includes the train, valid, and test dataset with the measurement of KIKD 
and IC50 constructed based on the strategy of "Compound-Clustered" and "Protein-Clustered") from the URL link of 
<https://drive.google.com/drive/folders/1LbhSXNShSYrK60sTEvnyoL7I2cSRx7iU?usp=share_link>

2) And then, please unzip the files of "IC50" and "KIKD" under the folder of "FeatNN/Datasets/DataProcessed/" in this project before training. 
(***Please make sure the paths of "FeatNN/Datasets/DataProcessed/IC50/", and "FeatNN/Datasets/DataProcessed/KIKD/" exist in this project.***)


The script of "train_FeatNN.py" will successfully execute with after the deployment of these steps.
