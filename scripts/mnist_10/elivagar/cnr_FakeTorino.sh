#!/bin/bash

# [IBMQ-原始命令，已停用] Retained for comparison with the original Osaka workflow.
# python compute_clifford_nr.py --dataset mnist_10 --num_circs 2500 --num_cdcs 32 --device_name ibm_osaka --use_qubit_mapping --save_cnr_scores --encoding_type angle --circs_dir ./experiments/mnist_10/72_params/ibm_osaka/elivagar --dataset_file_extension npy

# [FakeBackend] Use the snapshot noise model and restrict it to each circuit's mapped qubits.
python compute_clifford_nr.py --dataset mnist_10_6_nll --num_circs 2500 --num_cdcs 32 --device_name ibm_FakeTorino --use_qubit_mapping --save_cnr_scores --encoding_type angle --circs_dir ./experiments/mnist_10_6_nll/72_params/ibm_FakeTorino/elivagar --dataset_file_extension npy
