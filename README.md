# FeatNN
Binding affinity prediction with evolutionary strategy

# Required python packages
python 3.5-3.7

rdkit

pytorch 

sklearn

numpy

pandas

# Usage Information & Introduction

Use the command : 'python train_FeatNN.py --measures=IC50 --setting=new_compound --threshold=0.3 --param=param.json' to set the train dataset as IC50 measurement, compound-clustered with 0.3 threshold, while the hyper-parameters in the file of 'param.json' could be modified by yourself. The value of treshold can choose from 0.3 to 0.6 with step of 0.1. The clustered strategy (--setting) can set as 'new_compound' or 'new_protein'. The measurment can be set as 'IC50' or 'KIKD'.

The other command example are given as follows:

'python train_FeatNN.py --measures=KIKD --setting=new_compound --threshold=0.3 --param=param.json'

'python train_FeatNN.py --measures=KIKD --setting=new_protein --threshold=0.3 --param=param.json'

'python train_FeatNN.py --measures=KIKD --setting=new_protein --threshold=0.6 --param=param.json'
