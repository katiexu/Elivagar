"""Export Elivagar trial/run metrics and gate counts to a task-level CSV.

Example:
    python export_trial_results.py --search_dir \
        experiments/mnist_4_0123/40_params/ibm_FakeTorino/elivagar/search_100_ibm_FakeTorino

Noisy metric rows are assumed to remain in run_1, run_2, ... order. If inference
was run with --pick_best, its output order may not identify the original run.
"""

import argparse
import ast
import csv
import math
import re
import warnings
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_SEARCH_DIR = Path(
    "experiments/mnist_4_0123/40_params/ibm_FakeTorino/elivagar/"
    "search_100_ibm_FakeTorino"
)

CSV_COLUMNS = [
    "trial_number",
    "run_number",
    "num_params",
    "num_embed",
    "tq_train_loss",
    "tq_train_acc",
    "tq_test_loss",
    "tq_test_acc",
    "noisy_test_loss",
    "noisy_test_acc",
    "total_gate_count",
    "single_gate_count",
    "enta_gate_count",
]

METRIC_FILES = {
    "tq_train_loss": "train_losses.txt",
    "tq_train_acc": "train_accs.txt",
    "tq_test_loss": "val_losses.txt",
    "tq_test_acc": "accs.txt",
}


def _resolve_path(path: str | Path, project_root: Path) -> Path:
    path = Path(path).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def _task_name(search_dir: Path, project_root: Path, task_name: str | None = None) -> str:
    if task_name is not None:
        if not task_name or Path(task_name).name != task_name:
            raise ValueError("task_name must be a non-empty directory name")
        return task_name

    experiments_dir = (project_root / "experiments").resolve()
    try:
        relative_path = search_dir.relative_to(experiments_dir)
    except ValueError as exc:
        raise ValueError(
            f"Cannot infer task name: {search_dir} is not under {experiments_dir}; "
            "pass --task_name explicitly."
        ) from exc

    if not relative_path.parts:
        raise ValueError("Cannot infer task name from the experiments directory itself")
    return relative_path.parts[0]


def _noise_device(search_dir: Path, noise_device: str | None = None) -> str:
    if noise_device:
        return noise_device

    candidates = []
    search_match = re.fullmatch(r"search_\d+_(.+)", search_dir.name)
    if search_match:
        candidates.append(search_match.group(1))

    # Standard layout: <device>/elivagar/search_<ratio>_<device>.
    if search_dir.parent.name == "elivagar":
        candidates.append(search_dir.parent.parent.name)

    candidates = list(dict.fromkeys(candidates))
    if not candidates:
        raise ValueError(
            f"Cannot infer the noisy-evaluation device from {search_dir}; "
            "pass --noise_device explicitly."
        )
    if len(candidates) > 1:
        raise ValueError(
            f"Ambiguous noisy-evaluation device for {search_dir}: {candidates}; "
            "pass --noise_device explicitly."
        )
    return candidates[0]


def _read_metric_series(path: Path) -> list[str]:
    if not path.is_file():
        warnings.warn(f"Missing metric file: {path}; corresponding CSV values will be blank.",
                      RuntimeWarning, stacklevel=2)
        return []

    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    while lines and not lines[-1]:
        lines.pop()

    values = []
    for line_number, line in enumerate(lines, start=1):
        if not line:
            values.append("")
            continue
        parts = line.split()
        if len(parts) != 1:
            raise ValueError(f"Expected one metric per line in {path}:{line_number}")
        try:
            float(parts[0])
        except ValueError as exc:
            raise ValueError(f"Invalid metric in {path}:{line_number}: {parts[0]!r}") from exc
        values.append(parts[0])
    return values


def _read_bounds_count(path: Path) -> tuple[int, int]:
    if not path.is_file():
        raise FileNotFoundError(f"Required parameter-bound file is missing: {path}")

    values = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split()
        if len(parts) != 1:
            raise ValueError(f"Expected one bound per line in {path}:{line_number}")
        try:
            value = float(parts[0])
        except ValueError as exc:
            raise ValueError(f"Invalid bound in {path}:{line_number}: {parts[0]!r}") from exc
        if not math.isfinite(value) or not math.isclose(value, round(value), abs_tol=1e-8):
            raise ValueError(f"Expected a finite integer boundary in {path}:{line_number}")
        values.append(int(round(value)))

    if len(values) < 2:
        raise ValueError(f"Expected at least two cumulative boundaries in {path}")
    if any(current < previous for previous, current in zip(values, values[1:])):
        raise ValueError(f"Cumulative boundaries must be nondecreasing in {path}")
    return values[-1] - values[0], len(values)


def _gate_counts(trial_dir: Path) -> tuple[int, int, int]:
    gates_path = trial_dir / "gates.txt"
    positions_path = trial_dir / "gate_params.txt"
    if not gates_path.is_file():
        raise FileNotFoundError(f"Required gate list is missing: {gates_path}")
    if not positions_path.is_file():
        raise FileNotFoundError(f"Required gate positions are missing: {positions_path}")

    gates = [line.strip() for line in gates_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    position_lines = [
        line.strip() for line in positions_path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    if len(gates) != len(position_lines):
        raise ValueError(
            f"Gate/position row count mismatch in {trial_dir}: "
            f"{len(gates)} gates vs {len(position_lines)} positions"
        )

    single_gate_count = 0
    enta_gate_count = 0
    for row_number, position_text in enumerate(position_lines, start=1):
        try:
            positions = ast.literal_eval(position_text)
        except (SyntaxError, ValueError) as exc:
            raise ValueError(
                f"Invalid gate position in {positions_path}:{row_number}: {position_text!r}"
            ) from exc
        if isinstance(positions, int):
            positions = [positions]
        if not isinstance(positions, (list, tuple)) or len(positions) not in (1, 2):
            raise ValueError(
                f"Expected a one- or two-qubit position in {positions_path}:{row_number}"
            )
        if any(not isinstance(position, int) or isinstance(position, bool) for position in positions):
            raise ValueError(f"Expected integer qubit indices in {positions_path}:{row_number}")
        if len(positions) == 1:
            single_gate_count += 1
        else:
            enta_gate_count += 1

    return len(gates), single_gate_count, enta_gate_count


def _numbered_directories(parent: Path, prefix: str) -> list[int]:
    numbered = []
    for path in parent.iterdir():
        match = re.fullmatch(rf"{re.escape(prefix)}_(\d+)", path.name)
        if path.is_dir() and match:
            numbered.append(int(match.group(1)))
    return sorted(set(numbered))


def _metric_value(series: list[str], run_number: int, path: Path) -> str:
    if not path.is_file():
        # _read_metric_series already warned once for this missing file.
        return ""
    if run_number > len(series) or not series[run_number - 1]:
        warnings.warn(
            f"No metric value for run_{run_number} in {path}; the CSV value will be blank.",
            RuntimeWarning,
            stacklevel=2,
        )
        return ""
    return series[run_number - 1]


def export_results(
    search_dir: str | Path,
    project_root: Path = PROJECT_ROOT,
    task_name: str | None = None,
    noise_device: str | None = None,
) -> Path:
    """Export one CSV row per trial/run and return the output path."""
    project_root = Path(project_root).expanduser().resolve()
    search_dir = _resolve_path(search_dir, project_root)
    if not search_dir.is_dir():
        raise FileNotFoundError(f"Search directory does not exist: {search_dir}")

    resolved_task_name = _task_name(search_dir, project_root, task_name)
    resolved_noise_device = _noise_device(search_dir, noise_device)
    trial_numbers = _numbered_directories(search_dir, "trial")
    if not trial_numbers:
        raise ValueError(f"No trial_N directories found in {search_dir}")

    rows = []
    for trial_number in trial_numbers:
        trial_dir = search_dir / f"trial_{trial_number}"
        num_params, weights_bounds_length = _read_bounds_count(trial_dir / "weights_bounds.txt")
        num_embed, inputs_bounds_length = _read_bounds_count(trial_dir / "inputs_bounds.txt")
        total_gate_count, single_gate_count, enta_gate_count = _gate_counts(trial_dir)
        expected_bounds_length = total_gate_count + 1
        if weights_bounds_length != expected_bounds_length:
            raise ValueError(
                f"weights_bounds.txt in {trial_dir} has {weights_bounds_length} values; "
                f"expected {expected_bounds_length} for {total_gate_count} gates"
            )
        if inputs_bounds_length != expected_bounds_length:
            raise ValueError(
                f"inputs_bounds.txt in {trial_dir} has {inputs_bounds_length} values; "
                f"expected {expected_bounds_length} for {total_gate_count} gates"
            )

        path_param_match = next(
            (re.fullmatch(r"(\d+)_params", part) for part in search_dir.parts
             if re.fullmatch(r"\d+_params", part)),
            None,
        )
        if path_param_match and int(path_param_match.group(1)) != num_params:
            raise ValueError(
                f"Path specifies {path_param_match.group(1)}_params but {trial_dir} "
                f"contains {num_params} trainable parameters"
            )

        metric_paths = {
            column: trial_dir / filename for column, filename in METRIC_FILES.items()
        }
        noisy_dir = trial_dir / "noise_sim" / resolved_noise_device
        metric_paths["noisy_test_loss"] = noisy_dir / "val_losses_inference_only.txt"
        metric_paths["noisy_test_acc"] = noisy_dir / "accs_inference_only.txt"
        metric_series = {
            column: _read_metric_series(path) for column, path in metric_paths.items()
        }

        run_numbers = _numbered_directories(trial_dir, "run")
        if not run_numbers:
            max_metric_rows = max((len(values) for values in metric_series.values()), default=0)
            run_numbers = list(range(1, max_metric_rows + 1))
        if not run_numbers:
            warnings.warn(
                f"No run_N directories or metric rows found in {trial_dir}; no rows exported.",
                RuntimeWarning,
                stacklevel=2,
            )
            continue

        for run_number in run_numbers:
            row = {
                "trial_number": f"trial_{trial_number}",
                "run_number": f"run_{run_number}",
                "num_params": num_params,
                "num_embed": num_embed,
                "total_gate_count": total_gate_count,
                "single_gate_count": single_gate_count,
                "enta_gate_count": enta_gate_count,
            }
            for column, series in metric_series.items():
                row[column] = _metric_value(series, run_number, metric_paths[column])
            rows.append(row)

    if not rows:
        raise ValueError(f"No trial/run result rows found in {search_dir}")

    results_dir = project_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / f"results_{resolved_task_name}.csv"
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--search_dir",
        "--search-dir",
        type=Path,
        default=DEFAULT_SEARCH_DIR,
        help="trial search directory (relative paths are resolved from the project root)",
    )
    parser.add_argument("--task_name", "--task-name", help="override task name used in the CSV filename")
    parser.add_argument(
        "--noise_device",
        "--noise-device",
        help="override the device subdirectory under each trial's noise_sim folder",
    )
    args = parser.parse_args(argv)

    try:
        output_path = export_results(
            args.search_dir,
            project_root=PROJECT_ROOT,
            task_name=args.task_name,
            noise_device=args.noise_device,
        )
    except (FileNotFoundError, OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())