# Élivágar: Efficient Quantum Circuit Search for Classification

Élivágar is a training-free, device- and noise-aware framework for Quantum Circuit Search (QCS) for classification tasks.

## Setup

To setup the environment required to run Élivágar, first clone this repository:

```
git clone https://github.com/SashwatAnagolum/Elivagar.git
```

Then, create a new virtual environment, and activate it:

```
python -m venv elivagar_venv
elivagar_venv/Scripts/activate
```

Install all of the required packages via pip:

```
pip install -r requirements.txt
```

Now, install the Pytorch-Tensor-Train-Network package:

```
cd qtn_vqc/Pytorch-Tensor-Train-Network
python -m pip install --no-deps .
cd ../..
```

`--no-deps` installs the local `tc` package without changing the versions pinned by `requirements.txt`.

Setup should now be complete.

## Example usage

As an example, we can use Élivágar to search for a circuit for the Moons dataset, targeting the IBM-Osaka device. To do so, we first generate candidate circuits for the target dataset-device combination:

```
./scripts/moons/elivagar/circ_gen_osaka.sh
```

Next, we compute the Clifford Noise Resilience (CNR) of each of the generated candidate circuits to estimate circuit noise robustness:

```
./scripts/moons/elivagar/cnr_osaka.sh
```

We then compute the representation capacity (RepCap) of candidate circuits, to estimate circuit performance:

```
./scripts/moons/elivagar/repcap_osaka.sh
```

Next, we can compute composite scores using the CNR and RepCap scores for each circuit, select the circuits to be trained, and train them on the Moons dataset:

```
./scripts/moons/elivagar/train_osaka.sh
```

Finally, we can evaluate the trained circuits on a withheld test set from the Moons dataset to check circuit performance:

```
./scripts/moons/elivagar/eval_osaka.sh
```

## Running with IBM fake backends

The legacy `elivagar` environment can continue to use the versions in `requirements.txt` for circuit generation, CNR, RepCap, training, and inference. Exporting a FakeBackend snapshot requires newer Qiskit packages, so do that once in a separate environment rather than installing them into `elivagar`:

```
conda create -n fake-torino python=3.10 -y
conda activate fake-torino
python -m pip install qiskit-ibm-runtime==0.52.0 qiskit-aer==0.17.2
```

From the repository root, export the local FakeTorino hardware and noise snapshot (no IBM account is required):

```
python extract_fake_torino.py --output fake_torino_properties.json
```

The resulting `fake_torino_properties.json` is a static calibration snapshot, not live device data. In the `elivagar` environment, `--device_name ibm_FakeTorino` resolves to this file automatically; use `--fake_backend_json <path>` if the snapshot is stored elsewhere.

CNR now restricts the FakeBackend noise model to the physical qubits in each circuit's mapping by default, retaining the mapped qubits' gate and readout errors. Based on the measured runtime, 2,500 circuits with 32 Clifford replicas are estimated to take about **1 hour**, rather than about **20 hours** with the full 133-qubit noise model. Add `--full_noise_model` to the `compute_clifford_nr.py` command to use the full model for comparison; the `cnr.txt` and score output formats and locations remain unchanged.

### MNIST-2 Élivágar workflow

Run the five scripts in order from the repository root, in the `elivagar` environment:

```bash
bash scripts/mnist_2/elivagar/circ_gen_FakeTorino.sh
bash scripts/mnist_2/elivagar/cnr_FakeTorino.sh
bash scripts/mnist_2/elivagar/repcap_FakeTorino.sh
bash scripts/mnist_2/elivagar/train_FakeTorino.sh
bash scripts/mnist_2/elivagar/eval_FakeTorino.sh
```

Circuit generation uses the snapshot's hardware properties. CNR and noisy inference build an `AerSimulator` from its noise model; neither runs a `FakeTorino` backend object or connects to IBM hardware. RepCap and model training are classical/device-independent stages, although training uses the CNR and RepCap scores. Install the additional RepCap dependency in `elivagar` with `python -m pip install "qutip<5"` if it is not already available.

### MNIST-4 digits 0–3 Élivágar workflow

The `mnist_4_0123` dataset contains all MNIST examples labeled 0, 1, 2, or 3 (28,911 total). The generator combines the original MNIST training and test files, then creates a stratified 80:20 split by default; consequently, it does not preserve the original official train/test boundary. To regenerate the four text files from the standard IDX files in `experiment_data/MNIST/raw`, run this from the repository root:

```bash
python experiment_data/make_mnist_4_0123.py
```

The output is written to `experiment_data/mnist_4_0123/` (23,129 training and 5,782 test examples with the default seed and ratio). The existing preprocessing produces 16 features per image and two-column bipolar labels. The split ratio and random seed can be changed with `--train_ratio` and `--seed`.

Run the five FakeTorino scripts in order from the repository root, in the `elivagar` environment:

```bash
bash scripts/mnist_4/elivagar/circ_gen_FakeTorino.sh
bash scripts/mnist_4/elivagar/cnr_FakeTorino.sh
bash scripts/mnist_4/elivagar/repcap_FakeTorino.sh
bash scripts/mnist_4/elivagar/train_FakeTorino.sh
bash scripts/mnist_4/elivagar/eval_FakeTorino.sh
```

These scripts save their circuits and results under `experiments/mnist_4_0123/40_params/ibm_FakeTorino/elivagar/`. RepCap samples 16 examples per class (64 total) for each circuit; training uses the full training split. The evaluation script follows the original MNIST-4 setting and samples 2,000 test examples rather than all 5,782. Remove `--num_test_samples 2000` from that script to evaluate the full test split. Training uses the larger dataset and can take substantially longer than MNIST-2; the Osaka-named scripts are retained unchanged for the original workflow.

### MNIST-10 NLL FakeTorino workflow

The `mnist_10_6_nll` dataset uses 36 features per image, 10 measured qubits/classes, and integer class labels from 0 to 9. Its NPY files contain 50,000 training and 10,000 test examples. Run the five FakeTorino scripts from the repository root in order:

```bash
bash scripts/mnist_10/elivagar/circ_gen_FakeTorino.sh
bash scripts/mnist_10/elivagar/cnr_FakeTorino.sh
bash scripts/mnist_10/elivagar/repcap_FakeTorino.sh
bash scripts/mnist_10/elivagar/train_FakeTorino.sh
bash scripts/mnist_10/elivagar/eval_FakeTorino.sh
```

The scripts use the existing `mnist_10_6_nll` NPY dataset and save outputs under `experiments/mnist_10_6_nll/72_params/ibm_FakeTorino/elivagar/`. Training uses NLL for 50 epochs and, by default, all 50,000 training examples; add `--num_train_samples 512` to the training command for a smaller pilot. Evaluation samples 400 test examples per trial. CNR and noisy inference use the FakeTorino snapshot through `AerSimulator`, not a live IBM backend. Runtime depends heavily on CPU performance and the generated circuits; measure the pilot before scheduling the full 2,500-circuit workflow.
