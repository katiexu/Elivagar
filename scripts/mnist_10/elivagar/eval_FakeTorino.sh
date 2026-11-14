#!/bin/bash

# [IBMQ-原始命令，已停用] Retained for comparison with the original Osaka workflow.
# python circuit_inference.py --dataset mnist_10 --num_circs 25 --num_runs_per_circ 5 --encoding_type angle --device_name ibm_osaka --circs_dir ./experiments/mnist_10/72_params/ibm_osaka/elivagar/search_100_ibm_osaka --num_test_samples 400 --results_save_dir noise_sim --transpiler_opt_level 1 --circ_prefix trial --use_qubit_mapping --qubit_mapping_filename qubit_mapping.txt --dataset_file_extension npy

# [FakeBackend] Evaluate integer-label predictions using the local FakeTorino snapshot noise model.
python circuit_inference.py --dataset mnist_10_6_nll --num_circs 25 --num_runs_per_circ 5 --encoding_type angle --device_name ibm_FakeTorino --circs_dir ./experiments/mnist_10_6_nll/72_params/ibm_FakeTorino/elivagar/search_100_ibm_FakeTorino --num_test_samples 400 --results_save_dir noise_sim --transpiler_opt_level 1 --circ_prefix trial --use_qubit_mapping --qubit_mapping_filename qubit_mapping.txt --dataset_file_extension npy --loss nll
