import os


python_line = ['python featnn_train.py', 'python train_merge.py', 'python featnn_train_opt.py']

for line in python_line:
    try:
        os.system(line)
    except Exception as e:
        continue
