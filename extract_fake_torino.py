#!/usr/bin/env python3
"""Export FakeTorino's hardware target and Aer noise model as JSON.

Install the current fake-provider and Aer packages in a separate environment
(not the repository's legacy Qiskit environment), then run:

    python extract_fake_torino.py --output fake_torino_properties.json

Fake backends contain a static calibration snapshot; this script does not
connect to IBM Quantum or require an IBM account.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def to_jsonable(value: Any) -> Any:
    """Convert Qiskit, NumPy, and standard Python values to JSON-safe values."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        return to_jsonable(value.tolist())
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if hasattr(value, "__dict__"):
        return to_jsonable(vars(value))
    return str(value)


def extract_qubit_calibrations(backend: Any, num_qubits: int) -> dict[str, dict[str, Any]]:
    """Return per-qubit calibration values exposed by BackendProperties."""
    calibrations = {str(index): {} for index in range(num_qubits)}
    properties = backend.properties()
    if properties is None:
        return calibrations

    for qubit_index, parameters in enumerate(properties.qubits):
        if qubit_index >= num_qubits:
            break
        for parameter in parameters:
            name = getattr(parameter, "name", None)
            if name is None:
                continue
            calibrations[str(qubit_index)][name] = to_jsonable(
                getattr(parameter, "value", None)
            )
            unit = getattr(parameter, "unit", None)
            if unit:
                calibrations[str(qubit_index)][f"{name}_unit"] = unit
    return calibrations


def extract_target(backend: Any) -> tuple[dict[str, Any], list[list[int]]]:
    """Collect BackendV2 target operation properties and directed couplings."""
    target = backend.target
    operations = []
    coupling_map = []
    seen_edges = set()

    for operation_name, qarg_properties in target.items():
        operation = target.operation_from_name(operation_name)
        num_operation_qubits = next(
            (len(qargs) for qargs in qarg_properties if qargs is not None), None
        )
        if num_operation_qubits is None:
            operation_width = getattr(operation, "num_qubits", None)
            if isinstance(operation_width, int):
                num_operation_qubits = operation_width
        qargs_data = []

        for qargs, instruction_properties in qarg_properties.items():
            qargs_list = list(qargs) if qargs is not None else None
            properties = {}
            for attribute in ("duration", "error"):
                value = getattr(instruction_properties, attribute, None)
                if value is not None:
                    properties[attribute] = to_jsonable(value)

            qargs_data.append({"qubits": qargs_list, **properties})
            if num_operation_qubits == 2 and qargs_list is not None:
                edge = tuple(qargs_list)
                if edge not in seen_edges:
                    seen_edges.add(edge)
                    coupling_map.append(qargs_list)

        operations.append(
            {
                "name": operation_name,
                "num_qubits": num_operation_qubits,
                "qargs": qargs_data,
            }
        )

    target_data = {
        "num_qubits": getattr(target, "num_qubits", None),
        "dt_seconds": to_jsonable(getattr(target, "dt", None)),
        "operations": operations,
    }
    return target_data, coupling_map


def extract_backend_configuration(backend: Any) -> dict[str, Any]:
    """Collect common backend configuration attributes when exposed."""
    attribute_names = (
        "backend_version",
        "description",
        "online_date",
        "simulator",
        "local",
        "memory",
        "max_shots",
        "max_experiments",
        "basis_gates",
        "coupling_map",
        "dt",
        "dtm",
        "meas_map",
        "max_memory_mb",
        "processor_type",
        "rep_times",
        "supported_instructions",
    )
    configuration = {}
    for name in attribute_names:
        try:
            value = getattr(backend, name, None)
        except (AttributeError, NotImplementedError):
            continue
        if value is not None:
            configuration[name] = to_jsonable(value)
    return configuration


def build_snapshot() -> dict[str, Any]:
    try:
        from qiskit_aer.noise import NoiseModel
        from qiskit_ibm_runtime.fake_provider import FakeTorino
    except ImportError as error:
        raise RuntimeError(
            "FakeTorino extraction requires qiskit-ibm-runtime and qiskit-aer. "
            "Install them in a separate environment so the repository's legacy "
            "Qiskit dependencies remain unchanged."
        ) from error

    backend = FakeTorino()
    num_qubits = backend.num_qubits
    target_data, coupling_map = extract_target(backend)
    noise_model = NoiseModel.from_backend(backend)

    return {
        "backend_name": str(backend.name),
        "backend_version": to_jsonable(getattr(backend, "backend_version", None)),
        "num_qubits": num_qubits,
        "configuration": extract_backend_configuration(backend),
        "coupling_map_directed": coupling_map,
        "qubit_calibrations": extract_qubit_calibrations(backend, num_qubits),
        "target": target_data,
        "noise_model": to_jsonable(noise_model.to_dict()),
        "noise_model_basis_gates": list(noise_model.basis_gates),
        "source": "qiskit_ibm_runtime.fake_provider.FakeTorino",
        "note": "Static fake-backend calibration snapshot; not live device data.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export FakeTorino hardware and noise data to JSON."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("fake_torino_properties.json"),
        help="output JSON path (default: fake_torino_properties.json)",
    )
    args = parser.parse_args()

    try:
        snapshot = build_snapshot()
        args.output.write_text(
            json.dumps(snapshot, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    except (ImportError, RuntimeError, OSError, TypeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print(f"Wrote FakeTorino hardware and noise data to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
