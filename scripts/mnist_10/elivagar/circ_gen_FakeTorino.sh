#!/bin/bash

# [IBMQ-原始命令，已停用] Retained for comparison with the original Osaka workflow.
# python generate_device_aware_circuits.py --target_dataset mnist_10_6_nll --num_circs 2500 --device_name ibm_osaka --save_dir ./experiments/mnist_10/72_params/ibm_osaka/elivagar --temp 0.5 --add_rotations

# [FakeBackend] Generate candidate circuits using hardware properties from the local FakeTorino snapshot.
python generate_device_aware_circuits.py --target_dataset mnist_10_6_nll --num_circs 2500 --device_name ibm_FakeTorino --save_dir ./experiments/mnist_10_6_nll/72_params/ibm_FakeTorino/elivagar --temp 0.5 --add_rotations
