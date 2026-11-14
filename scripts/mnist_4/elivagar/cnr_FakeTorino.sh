#!/bin/bash

# [IBMQ-原始命令，已停用] Computes CNR using the original MNIST-4 circuits and IBM Osaka noise model.
# python compute_clifford_nr.py --dataset mnist_4 --num_circs 2500 --num_cdcs 32 --device_name ibm_osaka --use_qubit_mapping --save_cnr_scores --encoding_type angle --circs_dir ./experiments/mnist_4/40_params/ibm_osaka/elivagar

# [FakeBackend] Compute CNR from the local FakeTorino snapshot; noise is restricted to each circuit's mapping by default.
python compute_clifford_nr.py --dataset mnist_4_0123 --num_circs 2500 --num_cdcs 32 --device_name ibm_FakeTorino --use_qubit_mapping --save_cnr_scores --encoding_type angle --circs_dir ./experiments/mnist_4_0123/40_params/ibm_FakeTorino/elivagar
