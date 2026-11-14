#!/bin/bash

# [IBMQ-原始命令，已停用] Original noisy inference used the IBMQ-backed noise model and mnist_4 trials.
# python circuit_inference.py --dataset mnist_4 --num_circs 25 --num_runs_per_circ 5 --encoding_type angle --device_name ibm_osaka --circs_dir ./experiments/mnist_4/40_params/ibm_osaka/elivagar/search_100_ibm_osaka --num_test_samples 2000 --results_save_dir noise_sim --transpiler_opt_level 1 --circ_prefix trial --use_qubit_mapping --qubit_mapping_filename qubit_mapping.txt

# [FakeBackend] Run noisy inference with the local FakeTorino snapshot; by default only mapped-qubit errors are used.
python circuit_inference.py --dataset mnist_4_0123 --num_circs 25 --num_runs_per_circ 5 --encoding_type angle --device_name ibm_FakeTorino --circs_dir ./experiments/mnist_4_0123/40_params/ibm_FakeTorino/elivagar/search_100_ibm_FakeTorino --num_test_samples 2000 --results_save_dir noise_sim --transpiler_opt_level 1 --circ_prefix trial --use_qubit_mapping --qubit_mapping_filename qubit_mapping.txt
