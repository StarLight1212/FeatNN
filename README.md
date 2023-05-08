# FeatNN
Highly improved precision of binding affinity prediction with evolutionary strategy and deep GCN block(addressing the over-smoothing problem).

![alt text](https://github.com/StarLight1212/FeatNN/blob/main/meta/Fig1.jpg "FeatNN Architecture")

FeatNN achieves the SOTA performances in general datasets.
![alt text](https://github.com/StarLight1212/FeatNN/blob/main/meta/Fig2.jpg "SOTA Results")


FeatNN could effectively used in drug discovery and drug screening.

![alt text](https://github.com/StarLight1212/FeatNN/blob/main/meta/Fig6.jpg "Drug Screening Results")

Note: The scripts were encrypted with the encoded algorthm. If you want know the detail of the project, please contact us with email: ansehen@163.com. If you have special needs or are interested in the data set construction process, please contact us, and we will upload and open source the code of the data set construction part in succession. The code of the model part is prohibited from commercial use. If you need the source code for the purpose of learning and education, please contact us. After verification, or you will be sent the model source code by email.
___  

## Required python packages
- python 3.6-3.8
- pytorch torch-deploy-1.8 (GPU 3090, CUDA 11.2)
- torch = 1.10.1+cu113
- torchvision = 0.11.2+cu113
- rdkit (2021.09.4)
- bio = 1.3.3
- biopython = 1.79
- sklearn = 1.0.2
- numpy = 1.21.5
- pandas = 0.24.2
- scipy = 1.7.1
- openbabel = 3.1.1
- argparse


**News:**    
```yaml
The improvement directions of FeatNN mainly include the following 5 points:  
```  
- [x] 1. On a dataset constructed from PDBbind, our model greatly outperforms the SOTA models in CPA prediction tasks. FeatNN considerably outperforms the SOTA baselines in CPA prediction, with R2, root mean square error (RMSE) and Pearson coefficient values that are highly elevated  
- [x] 2. An Evo-Updating block is employed in the protein encoding module to interactively update the sequence and structure information of proteins so that the high-quality features of proteins are extracted and presented, enabling FeatNN to outperform various SOTA models.  
- [x] 3. In FeatNN, the distance matrices of protein residues were discretized into one dimension, and a word embedding strategy was applied to encode protein structure information and significantly reduce the computational cost of the proposed method; however, it still effectively represented the structure information of proteins. 
- [x] 4. With respect to the extraction of compound features, a specific residual connection is applied to represent the molecular graph, in which the features of the initial nodes are added onto each layer of the GCN, such that the graph features representation limitation caused by the notorious oversmoothing problem in traditional deep GCNs is solved.
- [x] 5. With the pretraining and fine-tuning strategy, both the generalization and R2 performance of the optimized model, FeatNNoptm, further increases compared to that of FeatNN.
- [x] 6. FeatNN has excellent generalization in the affinity prediction task, which is vital and pivotal in the drug screening domain. Targeting severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2) 3-chymotrypsin (3C)-like protease and Akt-1, the generalization of FeatNN vastly outperforms the SOTA baseline in the affinity value prediction task.
- [x] 7. The prediction results of FeatNN with different conformations of the same protein are robust when 3D structure information is directly introduced in the model while neglecting the molecular dynamics of the protein.

------

## Usage 
### Datasets Generation Protocol
Here, we want to provide a standard method for how constructing and processing the dataset in the domain of compound-protein affinity prediction to help the other researchers well processing the data before training.
  > [Standard Datasets Construction Protocols Link](https://github.com/StarLight1212/FeatNN/blob/main/DataGenerates/Dataset_Construction_Protocols.md)


### Datasets
Download Train/Val/Test datasets constructed from PDBBind-v2020 follow the protocols:  
  > [Datasets Download Protocol Link](https://github.com/StarLight1212/FeatNN/blob/main/Datasets/Download_Dataset_from_GoogleDrive.md)

### Introduction & Inference

Use the command : 'python train_FeatNN.py --measures=IC50 --setting=new_compound --threshold=0.3 --param=param.json' to set the train dataset as IC50 measurement, compound-clustered with 0.3 threshold, while the hyper-parameters in the file of 'param.json' could be modified by yourself. The value of treshold can choose from 0.3 to 0.6 with step of 0.1. The clustered strategy (--setting) can set as 'new_compound' or 'new_protein'. The measurment can be set as 'IC50' or 'KIKD'.


The other command example are given as follows:
```yaml
'python train_FeatNN.py --measures=IC50 --setting=ComClu --threshold=0.3 --param=param.json --batch_size=16'

'python train_FeatNN.py --measures=KIKD --setting=ComClu --threshold=0.3 --param=param.json --batch_size=16'

'python train_FeatNN.py --measures=IC50 --setting=ProtClu --threshold=0.3 --param=param.json --batch_size=16'

'python train_FeatNN.py --measures=KIKD --setting=ProtClu --threshold=0.6 --param=param.json --batch_size=16'

...
```  
### Test (Demo) the Results of FeatNN
Before Test the Result of FeatNN, please download the structure and sequence features according to the [Datasets Download Protocol Link](https://github.com/StarLight1212/FeatNN/blob/main/Datasets/Download_Dataset_from_GoogleDrive.md).

In the directory of [test(demo)](https://github.com/StarLight1212/FeatNN/tree/main/test(demo)), run the command of:
```yaml
'python test_model.py --model_dir=FeatNN_model_IC50_m0.pth --batch_size=16 --measures=IC50'

'python test_model.py --model_dir=FeatNN_model_IC50_m1.pth --batch_size=16 --measures=IC50'

'python test_model.py --model_dir=FeatNN_model_IC50_m2.pth --batch_size=16 --measures=IC50'

'python test_model.py --model_dir=FeatNN_model_KIKD_k0.pth --batch_size=16 --measures=KIKD'

'python test_model.py --model_dir=FeatNN_model_KIKD_k1.pth --batch_size=16 --measures=KIKD'

'python test_model.py --model_dir=FeatNN_model_KIKD_k2.pth --batch_size=16 --measures=KIKD'
```  
More test models could be found in the link: 
 > [Models' Link](https://drive.google.com/drive/folders/1ZLciROL7MPn0PY63Bd6HMCFE7fDPDg4c?usp=share_link)

Download all files into the directory of './test(demo)/model/'.

## License  
This repo is made freely available to academic and non-academic entities for non-commercial purposes such as academic research, teaching, and scientific publications. Permission is granted to use FeatNN given that you agree to my licensing terms. Regarding the request for commercial use, please contact us via email to help you obtain the authorization letter.  

## Author  
Binjie Guo, Hanyu Zheng, Haohan Jiang, Xuhua Wang


## Acknowledgement
Finally, here, we sincerely appreciate for the code from Melissa F. Adasme et al(https://github.com/ssalentin/plip/) from the Dresden University in Deutschland and the code from Shuya Li et al(https://github.com/lishuya17/MONN).
