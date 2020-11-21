#!/usr/bin/env bash

times=4
for j in `seq 1 $times`
do
max=3
for i in `seq 0 $max`
do
# python main_mldg.py train \
python main_mldg.py \
--lr=0.001 \
--num_classes=2 \
--test_every=500 \
--logs='run_'$j'/logs_mldg_'$i'/' \
--batch_size=128 \
--val_split=0.9 \
--model_path='run_'$j'/models_mldg_'$i'/' \
--unseen_index=$i \
--inner_loops=45001 \
--step_size=15000 \
--state_dict='' \
--data_root='/home/weiyuhua/Challenge2020/Data/DG' \
--meta_step_size=5e-1 \
--meta_val_beta=1.0
done
done


# --lr=5e-4 --num_classes=7 --test_every=500 --logs=logs_mldg --batch_size=64 --model_path=models_mldg --unseen_index=0 --inner_loops=45001 --step_size=15000 --state_dict='' --data_root=data --meta_step_size=5e-1 --meta_val_beta=1.0