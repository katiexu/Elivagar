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
