#!/bin/bash

# [IBMQ-原始命令，已停用] Kept for reference; the ibm_osaka circuit directory is no longer generated.
# python compute_rep_cap.py --dataset mnist_2 --num_circs 2500 --encoding_type angle --circs_dir ./experiments/mnist_2/20_params/ibm_osaka/elivagar

# [FakeBackend] RepCap is device-independent; only the circuit directory changes to the FakeTorino-generated circuits.
python compute_rep_cap.py --dataset mnist_2 --num_circs 2500 --encoding_type angle --circs_dir ./experiments/mnist_2/20_params/ibm_FakeTorino/elivagar
