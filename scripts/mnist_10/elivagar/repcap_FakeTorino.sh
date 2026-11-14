#!/bin/bash

# [IBMQ-原始命令，已停用] Retained for comparison with the original Osaka workflow.
# python compute_rep_cap.py --dataset mnist_10 --num_circs 2500 --encoding_type angle --circs_dir ./experiments/mnist_10/72_params/ibm_osaka/elivagar --dataset_file_extension npy

# [FakeBackend] RepCap uses the existing integer-label dataset and does not simulate device noise.
python compute_rep_cap.py --dataset mnist_10_6_nll --num_circs 2500 --encoding_type angle --circs_dir ./experiments/mnist_10_6_nll/72_params/ibm_FakeTorino/elivagar --dataset_file_extension npy --num_samples_per_class 16 --num_param_samples 32
