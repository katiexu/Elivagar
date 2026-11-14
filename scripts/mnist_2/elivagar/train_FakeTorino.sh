#!/bin/bash

# [IBMQ-原始命令，已停用] Kept for reference; CNR scores are now stored under cnr_32/ibm_FakeTorino.
# python train_elivagar_circuits.py --dataset mnist_2 --circs_dir ./experiments/mnist_2/20_params/ibm_osaka/elivagar --device_name ibm_osaka --encoding_type angle --noise_importance 0.25 --num_data_for_rep_cap 32 --num_params_for_rep_cap 32 --num_cdcs 32 --num_circs 2500 --num_epochs 200 --batch_size 256 --learning_rate 0.01

# [FakeBackend] device_name only selects the cnr_32/<device_name>/cnr.txt files and names the search_100_<device_name> results directory.
# [TrainMetrics] Save final full-training-set accuracy and loss for each run under its trial directory.
python train_elivagar_circuits.py --dataset mnist_2 --circs_dir ./experiments/mnist_2/20_params/ibm_FakeTorino/elivagar --device_name ibm_FakeTorino --encoding_type angle --noise_importance 0.25 --num_data_for_rep_cap 32 --num_params_for_rep_cap 32 --num_cdcs 32 --num_circs 2500 --num_epochs 200 --batch_size 256 --learning_rate 0.01 --save_train_metrics
