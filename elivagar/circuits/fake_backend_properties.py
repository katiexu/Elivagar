"""Convert exported IBM FakeBackend snapshots to Elivagar device properties."""

import argparse
import copy
import json
import math
import re
import warnings
from functools import lru_cache
from pathlib import Path


_GATE_PARAM_NUMS = {
    'rz': 1, 'cx': 0, 'sx': 0, 'x': 0, 'y': 0,
    'z': 0, 'cz': 0, 'h': 0, 'rx': 1, 'ry': 1,
    'ecr': 0,
}
_NON_GATE_OPERATIONS = {'id', 'reset', 'delay', 'measure'}
_TIME_UNIT_TO_SECONDS = {
    's': 1.0,
    'sec': 1.0,
    'second': 1.0,
    'seconds': 1.0,
    'ms': 1e-3,
    'us': 1e-6,
    'µs': 1e-6,
    'μs': 1e-6,
    'ns': 1e-9,
    'ps': 1e-12,
}


def resolve_fake_backend_json_path(device_name, json_path=None):
    """Resolve an IBM FakeBackend name to its exported properties JSON."""
    if json_path is not None:
        return Path(json_path).expanduser()
    if not device_name:
        raise ValueError('A fake backend name or an explicit JSON path is required.')

    backend_name = str(device_name)
    if backend_name.startswith('ibm_'):
        backend_name = backend_name[len('ibm_'):]
    if not re.fullmatch(r'[A-Za-z0-9_]+', backend_name):
        raise ValueError(f'Unsupported fake backend name: {device_name!r}')

    snake_case_name = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', backend_name)
    snake_case_name = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1_\2', snake_case_name)
    root_dir = Path(__file__).resolve().parents[2]
    return root_dir / f'{snake_case_name.lower()}_properties.json'


@lru_cache(maxsize=8)
def _load_cached_json(path):
    with open(path, encoding='utf-8') as properties_file:
        properties = json.load(properties_file)
    if not isinstance(properties, dict):
        raise ValueError(f'Fake backend properties in {path} must be a JSON object.')
    return properties


def load_fake_backend_properties(path):
    """Load and cache a JSON snapshot produced by extract_fake_torino.py."""
    resolved_path = Path(path).expanduser().resolve()
    if not resolved_path.is_file():
        raise FileNotFoundError(
            f'Fake backend properties not found: {resolved_path}. '
            'Generate the snapshot with extract_fake_torino.py or pass '
            '--fake_backend_json with an existing snapshot.'
        )
    return _load_cached_json(str(resolved_path))


def _complexify_noise_parameters(value):
    if isinstance(value, list):
        return [_complexify_noise_parameters(item) for item in value]
    if isinstance(value, str):
        try:
            return complex(value)
        except ValueError:
            return value
    return value


def _restore_noise_model_complex_parameters(value):
    if isinstance(value, list):
        return [_restore_noise_model_complex_parameters(item) for item in value]
    if not isinstance(value, dict):
        return value

    instruction_name = value.get('name')
    restored = {}
    for key, item in value.items():
        if key == 'params' and instruction_name in {'kraus', 'unitary'}:
            restored[key] = _complexify_noise_parameters(item)
        else:
            restored[key] = _restore_noise_model_complex_parameters(item)
    return restored


def get_fake_backend_noise_model_data(properties):
    """Return an Aer-compatible noise dict, basis gates, and directed couplings."""
    raw_noise_model = properties.get('noise_model')
    if not isinstance(raw_noise_model, dict) or not isinstance(raw_noise_model.get('errors'), list):
        raise ValueError('Fake backend snapshot does not contain a valid Aer noise_model.')

    # [FakeBackend] Restore complex Kraus/unitary parameters serialized as strings in JSON.
    noise_model_data = _restore_noise_model_complex_parameters(copy.deepcopy(raw_noise_model))

    configuration = properties.get('configuration', {})
    basis_gates = configuration.get('basis_gates') if isinstance(configuration, dict) else None
    if not isinstance(basis_gates, list):
        basis_gates = properties.get('noise_model_basis_gates')
        if not isinstance(basis_gates, list):
            raise ValueError('Fake backend snapshot does not contain a basis-gate list.')
        basis_gates = [gate for gate in basis_gates if gate not in _NON_GATE_OPERATIONS]

    coupling_map = properties.get('coupling_map_directed')
    if not isinstance(coupling_map, list):
        raise ValueError('Fake backend snapshot does not contain a directed coupling map.')
    coupling_map = [[int(str(edge[0])), int(str(edge[1]))] for edge in coupling_map]

    return noise_model_data, list(basis_gates), coupling_map


def restrict_noise_model_data_to_qubits(noise_model_data, qubits):
    """[FakeBackend] Keep only noise errors scoped to the selected physical qubits."""
    if not isinstance(noise_model_data, dict) or not isinstance(noise_model_data.get('errors'), list):
        raise ValueError('Fake backend noise model data must contain an errors list.')

    selected_qubits = {int(qubit) for qubit in qubits}
    restricted_errors = []
    for error in noise_model_data['errors']:
        if not isinstance(error, dict):
            raise ValueError('Fake backend noise model errors must be dictionaries.')

        # [FakeBackend] Errors without gate_qubits are global and remain applicable.
        if 'gate_qubits' not in error:
            restricted_errors.append(dict(error))
            continue

        gate_qubits = error['gate_qubits']
        if not isinstance(gate_qubits, list):
            raise ValueError('Fake backend error gate_qubits must be a list.')
        retained_gate_qubits = [
            gate for gate in gate_qubits
            if all(int(qubit) in selected_qubits for qubit in gate)
        ]
        if retained_gate_qubits:
            # [FakeBackend] Copy each kept error and its filtered qargs without mutating the snapshot.
            retained_error = dict(error)
            retained_error['gate_qubits'] = retained_gate_qubits
            restricted_errors.append(retained_error)

    restricted_noise_model_data = dict(noise_model_data)
    restricted_noise_model_data['errors'] = restricted_errors
    return restricted_noise_model_data


def _finite_float(value, description):
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f'Invalid numeric value for {description}: {value!r}') from error
    if not math.isfinite(number):
        raise ValueError(f'Non-finite numeric value for {description}: {value!r}')
    return number


def _seconds(value, unit, description):
    if unit is None:
        raise ValueError(f'Missing time unit for {description}.')
    normalized_unit = str(unit).strip().lower()
    if normalized_unit not in _TIME_UNIT_TO_SECONDS:
        raise ValueError(f'Unsupported time unit {unit!r} for {description}.')
    return _finite_float(value, description) * _TIME_UNIT_TO_SECONDS[normalized_unit]


def _probability(value, description):
    probability = _finite_float(value, description)
    if probability < 0.0 or probability > 1.0:
        raise ValueError(f'{description} must be between 0 and 1; got {probability}.')
    return probability


def _mean(values):
    return sum(values) / len(values)


def _calibration_values(properties, num_qubits, name):
    calibrations = properties.get('qubit_calibrations', {})
    if not isinstance(calibrations, dict):
        raise ValueError('Fake backend qubit_calibrations must be a JSON object.')

    values = {}
    for qubit in range(num_qubits):
        calibration = calibrations.get(str(qubit), {})
        value = calibration.get(name)
        unit = calibration.get(f'{name}_unit')
        if value is not None and unit is not None:
            values[qubit] = _seconds(value, unit, f'{name} for qubit {qubit}')

    missing_qubits = [qubit for qubit in range(num_qubits) if qubit not in values]
    if not values:
        raise ValueError(f'Fake backend snapshot does not contain any {name} values.')
    if missing_qubits:
        fallback = _mean(list(values.values()))
        warnings.warn(
            f'Missing {name} calibration for {len(missing_qubits)} qubit(s); '
            'using the mean of available values.',
            RuntimeWarning,
            stacklevel=2,
        )
        for qubit in missing_qubits:
            values[qubit] = fallback
    return [values[qubit] for qubit in range(num_qubits)]


def _target_operations(properties):
    target = properties.get('target', {})
    operations = target.get('operations') if isinstance(target, dict) else None
    if not isinstance(operations, list):
        raise ValueError('Fake backend snapshot does not contain target.operations.')
    return {
        operation['name']: operation
        for operation in operations
        if isinstance(operation, dict) and isinstance(operation.get('name'), str)
    }


def extract_properties_from_fake_backend(properties):
    """Return the 11 device-property values expected by device_aware.py."""
    num_device_qubits = properties.get('num_qubits')
    if not isinstance(num_device_qubits, int) or num_device_qubits <= 0:
        raise ValueError('Fake backend snapshot must provide a positive num_qubits.')
    qubit_inds = list(range(num_device_qubits))

    t1_times = _calibration_values(properties, num_device_qubits, 'T1')
    t2_times = _calibration_values(properties, num_device_qubits, 'T2')

    calibrations = properties.get('qubit_calibrations', {})
    readout_success_values = {}
    for qubit in qubit_inds:
        calibration = calibrations.get(str(qubit), {})
        readout_error = calibration.get('readout_error')
        if readout_error is not None:
            readout_success_values[qubit] = 1.0 - _probability(
                readout_error, f'readout_error for qubit {qubit}'
            )
    missing_readout_qubits = [
        qubit for qubit in qubit_inds if qubit not in readout_success_values
    ]
    if not readout_success_values:
        raise ValueError('Fake backend snapshot does not contain any readout_error values.')
    if missing_readout_qubits:
        fallback = _mean(list(readout_success_values.values()))
        warnings.warn(
            f'Missing readout_error calibration for {len(missing_readout_qubits)} '
            'qubit(s); using the mean readout success probability.',
            RuntimeWarning,
            stacklevel=2,
        )
        for qubit in missing_readout_qubits:
            readout_success_values[qubit] = fallback
    meas_succ_probs = [readout_success_values[qubit] for qubit in qubit_inds]

    operations = _target_operations(properties)
    configuration = properties.get('configuration', {})
    configured_gates = configuration.get('basis_gates', [])
    if not isinstance(configured_gates, list):
        raise ValueError('Fake backend configuration.basis_gates must be a list.')

    basis_gates = [[], []]
    operation_widths = {}
    for gate in configured_gates:
        if gate in _NON_GATE_OPERATIONS:
            continue
        operation = operations.get(gate)
        if operation is None:
            continue
        width = operation.get('num_qubits')
        if width in (1, 2):
            basis_gates[width - 1].append(gate)
            operation_widths[gate] = width
    if not basis_gates[0] or not basis_gates[1]:
        raise ValueError(
            'Fake backend must expose at least one supported single- and two-qubit '
            'basis gate in configuration and target.operations.'
        )

    unknown_gates = [
        gate for gates in basis_gates for gate in gates
        if gate not in _GATE_PARAM_NUMS
    ]
    if unknown_gates:
        raise ValueError(
            'No parameter-count mapping is defined for fake backend gate(s): '
            f'{", ".join(unknown_gates)}.'
        )
    gates_param_nums = [
        [_GATE_PARAM_NUMS[gate] for gate in gates]
        for gates in basis_gates
    ]

    connectivity = {qubit: [] for qubit in qubit_inds}
    for edge in properties.get('coupling_map_directed', []):
        if not isinstance(edge, (list, tuple)) or len(edge) != 2:
            raise ValueError(f'Invalid directed coupling entry: {edge!r}')
        source, target = int(str(edge[0])), int(str(edge[1]))
        if source not in connectivity or target not in connectivity:
            raise ValueError(f'Coupling edge {edge!r} references an unknown qubit.')
        if target not in connectivity[source]:
            connectivity[source].append(target)

    qarg_records = {gate: {} for gates in basis_gates for gate in gates}
    duration_samples = {gate: [] for gates in basis_gates for gate in gates}
    for gate in qarg_records:
        operation = operations[gate]
        qargs_records = operation.get('qargs', [])
        if not isinstance(qargs_records, list):
            continue
        for record in qargs_records:
            qubits = record.get('qubits') if isinstance(record, dict) else None
            if not isinstance(qubits, (list, tuple)) or len(qubits) != operation_widths[gate]:
                continue
            qarg = tuple(int(str(qubit)) for qubit in qubits)
            qarg_records[gate][qarg] = record
            duration = record.get('duration')
            if duration is not None:
                duration = _finite_float(duration, f'{gate} duration on {qarg}')
                if duration < 0.0:
                    raise ValueError(f'Negative {gate} duration on {qarg}.')
                duration_samples[gate].append(duration)

    all_durations = [duration for samples in duration_samples.values() for duration in samples]
    if not all_durations:
        raise ValueError('Fake backend target does not contain any gate durations.')
    global_duration_fallback = max(all_durations)
    duration_fallbacks = {
        gate: max(samples) if samples else global_duration_fallback
        for gate, samples in duration_samples.items()
    }

    gate_times = [dict() for _ in qubit_inds]
    ent_succ_probs = [dict() for _ in qubit_inds]
    one_qubit_succ_probs = [dict() for _ in qubit_inds]
    missing_duration_count = [0]
    missing_error_count = [0]

    def gate_duration(gate, qarg):
        record = qarg_records[gate].get(qarg)
        duration = record.get('duration') if record else None
        if duration is None:
            missing_duration_count[0] += 1
            return duration_fallbacks[gate]
        return _finite_float(duration, f'{gate} duration on {qarg}')

    def gate_success(gate, qarg):
        record = qarg_records[gate].get(qarg)
        error = record.get('error') if record else None
        if error is None:
            missing_error_count[0] += 1
            return 0.0
        return 1.0 - _probability(error, f'{gate} error on {qarg}')

    for gate in basis_gates[0]:
        for qubit in qubit_inds:
            qarg = (qubit,)
            gate_times[qubit][gate] = gate_duration(gate, qarg)
            one_qubit_succ_probs[qubit][gate] = gate_success(gate, qarg)

    for gate in basis_gates[1]:
        for qubit in qubit_inds:
            gate_times[qubit][gate] = {}
            ent_succ_probs[qubit][gate] = {}
            for paired_qubit in connectivity[qubit]:
                qarg = (qubit, paired_qubit)
                gate_times[qubit][gate][paired_qubit] = gate_duration(gate, qarg)
                ent_succ_probs[qubit][gate][paired_qubit] = gate_success(gate, qarg)

    if missing_duration_count[0] > 0:
        warnings.warn(
            f'Missing duration for {missing_duration_count[0]} gate/qarg pair(s); '
            'using the largest available duration for the same gate (or the '
            'snapshot-wide maximum when that gate has no duration data).',
            RuntimeWarning,
            stacklevel=2,
        )
    if missing_error_count[0] > 0:
        warnings.warn(
            f'Missing error for {missing_error_count[0]} gate/qarg pair(s); treating '
            'their success probability as zero.',
            RuntimeWarning,
            stacklevel=2,
        )

    return (
        num_device_qubits,
        connectivity,
        t1_times,
        t2_times,
        meas_succ_probs,
        basis_gates,
        gates_param_nums,
        gate_times,
        ent_succ_probs,
        one_qubit_succ_probs,
        qubit_inds,
    )


def main():
    parser = argparse.ArgumentParser(
        description='Check and summarize a FakeBackend hardware-properties snapshot.'
    )
    parser.add_argument('--device_name', default='ibm_FakeTorino')
    parser.add_argument('--fake_backend_json', default=None)
    args = parser.parse_args()

    path = resolve_fake_backend_json_path(args.device_name, args.fake_backend_json)
    properties = load_fake_backend_properties(path)
    extracted = extract_properties_from_fake_backend(properties)
    print(f"Loaded {properties.get('backend_name', args.device_name)} from {path}")
    print(f'Qubits: {extracted[0]}')
    print(f'Directed couplings: {sum(len(edges) for edges in extracted[1].values())}')
    print(f'Basis gates: {extracted[5]}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
