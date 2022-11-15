# FeatNN
Highly improved precision of binding affinity prediction with evolutionary strategy and deep GCN block(addressing the over-smoothing problem).

Note: The scripts were encrypted with the encoded algorthm. If you want know the detail of the project, please contact us with email: ansehen@163.com. If you have special needs or are interested in the data set construction process, please contact us, and we will upload and open source the code of the data set construction part in succession. The code of the model part is prohibited from commercial use. If you need the source code for the purpose of learning and education, please contact us. After verification, or you will be sent the model source code by email.
___  

## Required python packages
- python 3.6-3.7
- pytorch torch-deploy-1.8 (GPU 3090, CUDA 11.2)
- rdkit
- sklearn
- numpy
- pandas
- argparse


**News:**    
```yaml
The improvement directions of FeatNN mainly include the following 4 points:  
```  
- [x] 1. On a dataset constructed from PDBbind, our model greatly outperforms the SOTA models in CPA prediction tasks. FeatNN considerably outperforms the SOTA baselines in CPA prediction, with R2, root mean square error (RMSE) and Pearson coefficient values that are highly elevated  
- [x] 2. An Evo-Updating block is employed in the protein encoding module to interactively update the sequence and structure information of proteins so that the high-quality features of proteins are extracted and presented, enabling FeatNN to outperform various SOTA models.  
- [x] 3. In FeatNN, the distance matrices of protein residues were discretized into one dimension, and a word embedding strategy was applied to encode protein structure information and significantly reduce the computational cost of the proposed method; however, it still effectively represented the structure information of proteins..  
- [x] 4. A specific residual connection was applied to represent a molecular graph, in which the features of the initial nodes were added onto each layer of a graph convolutional network (GCN), thereby solving the notorious oversmoothing problem in traditional deep GCNs.  

------

## Usage 
### Datasets
Download Train/Val/Test datasets constructed from PDBBind and BindingDB  
  > [Datasets' Link](https://www.terabox.com/sharing/link?surl=_7iLdiJ7TvmtAGxJ2rKSmQ)

### Introduction & Inference

Use the command : 'python train_FeatNN.py --measures=IC50 --setting=new_compound --threshold=0.3 --param=param.json' to set the train dataset as IC50 measurement, compound-clustered with 0.3 threshold, while the hyper-parameters in the file of 'param.json' could be modified by yourself. The value of treshold can choose from 0.3 to 0.6 with step of 0.1. The clustered strategy (--setting) can set as 'new_compound' or 'new_protein'. The measurment can be set as 'IC50' or 'KIKD'.


The other command example are given as follows:
```yaml
'python train_FeatNN.py --measures=KIKD --setting=new_compound --threshold=0.3 --param=param.json'

'python train_FeatNN.py --measures=KIKD --setting=new_protein --threshold=0.3 --param=param.json'

'python train_FeatNN.py --measures=KIKD --setting=new_protein --threshold=0.6 --param=param.json'
```  
### Test (Demo) the Results of FeatNN
In the directory of [test], run the command of:
```yaml
'python test_FeatNN.py --model=./model/FeatNN_n1.pth'
```  
## License  
This repo is made freely available to academic and non-academic entities for non-commercial purposes such as academic research, teaching, and scientific publications. Permission is granted to use FeatNN given that you agree to my licensing terms. Regarding the request for commercial use, please contact us via email to help you obtain the authorization letter.  

## Author  
Binjie Guo, Hanyu Zheng, Xuhua Wang
