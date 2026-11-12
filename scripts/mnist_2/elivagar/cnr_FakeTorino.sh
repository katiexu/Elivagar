#!/bin/bash

# [IBMQ-原始命令，已停用] Kept for reference; it requires an IBMQ backend account.
# python compute_clifford_nr.py --dataset mnist_2 --num_circs 2500 --num_cdcs 32 --device_name ibm_osaka --use_qubit_mapping --save_cnr_scores --encoding_type angle --circs_dir ./experiments/mnist_2/20_params/ibm_osaka/elivagar

# [FakeBackend] Run CNR using the local FakeTorino snapshot and its generated circuits.
python compute_clifford_nr.py --dataset mnist_2 --num_circs 2500 --num_cdcs 32 --device_name ibm_FakeTorino --use_qubit_mapping --save_cnr_scores --encoding_type angle --circs_dir ./experiments/mnist_2/20_params/ibm_FakeTorino/elivagar