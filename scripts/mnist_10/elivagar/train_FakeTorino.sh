#!/bin/bash

# [IBMQ-原始命令，已停用] Retained for comparison with the original Osaka workflow.
# python train_elivagar_circuits.py --dataset mnist_10 --circs_dir ./experiments/mnist_10/72_params/ibm_osaka/elivagar --device_name ibm_osaka --encoding_type angle --noise_importance 0.25 --num_data_for_rep_cap 160 --num_params_for_rep_cap 32 --num_cdcs 32 --num_circs 2500 --num_epochs 200 --batch_size 256 --learning_rate 0.01 --dataset_file_extension npy --loss nll

# [MNIST-10 NLL] The default trains on all 50,000 samples; optionally add --num_train_samples 512 for a short pilot.
# [TrainMetrics] Save final full-training-set accuracy and NLL for each run.
python train_elivagar_circuits.py --dataset mnist_10_6_nll --circs_dir ./experiments/mnist_10_6_nll/72_params/ibm_FakeTorino/elivagar --device_name ibm_FakeTorino --encoding_type angle --noise_importance 0.25 --num_data_for_rep_cap 160 --num_params_for_rep_cap 32 --num_cdcs 32 --num_circs 2500 --num_epochs 50 --batch_size 256 --learning_rate 0.01 --dataset_file_extension npy --loss nll --save_train_metrics
