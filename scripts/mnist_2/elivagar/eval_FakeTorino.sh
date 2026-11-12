#!/bin/bash

# [IBMQ-原始命令，已停用] Kept for reference; it downloaded the ibm_osaka noise model through an IBMQ account.
# python circuit_inference.py --dataset mnist_2 --num_circs 25 --num_runs_per_circ 5 --encoding_type angle --device_name ibm_osaka --circs_dir ./experiments/mnist_2/20_params/ibm_osaka/elivagar/search_100_ibm_osaka --num_test_samples 400 --results_save_dir noise_sim --transpiler_opt_level 1 --circ_prefix trial --use_qubit_mapping --qubit_mapping_filename qubit_mapping.txt

# [FakeBackend] Noisy inference with the local FakeTorino snapshot; the noise model is restricted to each trial's
# mapped qubits by default (add --full_noise_model to use the full 133-qubit model).
python circuit_inference.py --dataset mnist_2 --num_circs 25 --num_runs_per_circ 5 --encoding_type angle --device_name ibm_FakeTorino --circs_dir ./experiments/mnist_2/20_params/ibm_FakeTorino/elivagar/search_100_ibm_FakeTorino --num_test_samples 400 --results_save_dir noise_sim --transpiler_opt_level 1 --circ_prefix trial --use_qubit_mapping --qubit_mapping_filename qubit_mapping.txt
