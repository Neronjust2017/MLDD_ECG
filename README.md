## MLDG
This is a code sample for the paper "Learning to Generalize: Meta-Learning for Domain Generalization" https://arxiv.org/pdf/1710.03463.pdf
based on https://github.com/HAHA-DL/MLDG

## DATA
Ecg data from 'CSPC', 'PTB_XL', 'G12EC', 'Challenge2017'

## Requirements
Python 3.x

Pytorch 1.x.x

## Run the baseline
Please get the processed data first, you can contact with me.

sh run_baseline.sh       

## Run the MLDG

sh run_mldg.sh 

## Bibtex
```
 @inproceedings{Li2018MLDG,
   title={Learning to Generalize: Meta-Learning for Domain Generalization},
   author={Li, Da and Yang, Yongxin and Song, Yi-Zhe and Hospedales, Timothy},
  	booktitle={AAAI Conference on Artificial Intelligence},
  	year={2018}
 }
 ```
 
 ## Your own data
 Please tune the 'meta_step_size' and 'meta_val_beta' for your own data, this parameter is related to 'alpha' and 'beta' in paper which should be tuned for different cases.
