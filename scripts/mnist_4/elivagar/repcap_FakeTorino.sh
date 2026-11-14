#!/bin/bash

# [IBMQ-原始命令，已停用] Original RepCap run on the mnist_4 / ibm_osaka circuit directory.
# python compute_rep_cap.py --dataset mnist_4 --num_circs 2500 --encoding_type angle --circs_dir ./experiments/mnist_4/40_params/ibm_osaka/elivagar

# [FakeBackend] RepCap itself is device-independent; use the circuits generated for mnist_4_0123 on the FakeTorino mapping.
python compute_rep_cap.py --dataset mnist_4_0123 --num_circs 2500 --encoding_type angle --circs_dir ./experiments/mnist_4_0123/40_params/ibm_FakeTorino/elivagar
