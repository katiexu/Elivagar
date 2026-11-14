#!/bin/bash

# [IBMQ-原始命令，已停用] Original training used the mnist_4 / ibm_osaka CNR results.
# python train_elivagar_circuits.py --dataset mnist_4 --circs_dir ./experiments/mnist_4/40_params/ibm_osaka/elivagar --device_name ibm_osaka --encoding_type angle --noise_importance 0.25 --num_data_for_rep_cap 64 --num_params_for_rep_cap 32 --num_cdcs 32 --num_circs 2500 --num_epochs 200 --batch_size 256 --learning_rate 0.01

# [FakeBackend] Select circuits using their RepCap and FakeTorino-snapshot CNR scores, then train each selected structure.
# [TrainMetrics] Save final full-training-set accuracy and loss for each run under its trial directory.
python train_elivagar_circuits.py --dataset mnist_4_0123 --circs_dir ./experiments/mnist_4_0123/40_params/ibm_FakeTorino/elivagar --device_name ibm_FakeTorino --encoding_type angle --noise_importance 0.25 --num_data_for_rep_cap 64 --num_params_for_rep_cap 32 --num_cdcs 32 --num_circs 2500 --num_epochs 200 --batch_size 256 --learning_rate 0.01 --save_train_metrics
