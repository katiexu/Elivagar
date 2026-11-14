#!/bin/bash

# [IBMQ-原始命令，已停用] Generates from the original MNIST-4 dataset and IBM Osaka backend.
# python generate_device_aware_circuits.py --target_dataset mnist_4 --num_circs 2500 --device_name ibm_osaka --save_dir ./experiments/mnist_4/40_params/ibm_osaka/elivagar --temp 0.5 --add_rotations

# [FakeBackend] Generate 0/1/2/3 circuits using hardware properties in the local FakeTorino snapshot.
python generate_device_aware_circuits.py --target_dataset mnist_4_0123 --num_circs 2500 --device_name ibm_FakeTorino --save_dir ./experiments/mnist_4_0123/40_params/ibm_FakeTorino/elivagar --temp 0.5 --add_rotations
