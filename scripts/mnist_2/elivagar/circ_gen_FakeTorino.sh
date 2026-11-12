#!/bin/bash

python generate_device_aware_circuits.py --target_dataset mnist_2 --num_circs 2500 --device_name ibm_FakeTorino --save_dir ./experiments/mnist_2/20_params/ibm_FakeTorino/elivagar --temp 0.5 --add_rotations