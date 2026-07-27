from __future__ import annotations

import awkward as ak
import os
from fastml.utils.image import sliding_window
from fastml.utils_egamma.efex import eFex_slidingwindow_mask
import numpy as np
from sklearn.model_selection import train_test_split
from time import time
import time
import ROOT

from data.opendataloader import OpenDataLoader
from data.opendatahelper import *
from data.trainingdatahelper import *


class OpenDataSet():
    def __init__(self):
        self.x_train = None
        self.y_train = None
        self.x_val   = None
        self.y_val   = None
        self.w_train = None
        self.w_val = None

        self.bin_edges = np.arange(10.0, 40.0 + 0.5, 0.5)
        self.thresholds = [(i, i+0.5) for i in np.arange(10, 40,0.5)]
    
    def load_raw_data(self):
        name_map = {
            "Zee": "zee",
            "JZ": "jz",
        }
        loaders = {}
        towers_dict = {}

        for sample_name, _ in name_map.items():
            dl = OpenDataLoader(sample_name)
            stop = 3 if 'Zee' in sample_name else 2

            dl.load(n_start=0, n_stop=stop)

            w = ak.broadcast_arrays(dl.weight, dl.seed_vectors.eta)[0]

            _, (ev_ids, e0, p0) = eFex_slidingwindow_mask(dl.seed_vectors)

            towers = sliding_window(dl.towers, 3)[ev_ids, e0, p0]

            dl.seed_vectors = ak.flatten(dl.seed_vectors)
            dl.weight = ak.to_numpy(ak.flatten(w))
            towers_dict[sample_name] = towers
            loaders[sample_name] = dl
        return loaders, towers_dict

    def load(self):
        loaders, towers_dict = self.load_raw_data()

        training_tower_dist, training_pt_dist = filter_training(loaders, towers_dict, self.thresholds)
        # for sample_name in ["Zee", "JZ"]:
        #     print(f"[OPEN] [{sample_name}] training_tower_dist shape: {training_tower_dist[sample_name].shape}")

        X_bkg = training_tower_dist['JZ'] 

        total_bkg = training_pt_dist['JZ']
        
        X_sig = training_tower_dist['Zee']
        
        y_bkg = np.zeros((len(X_bkg), 1, 1, 1), dtype=np.int8)
        y_sig = np.ones((len(X_sig), 1, 1, 1), dtype=np.int8)

        w_bkg = calculate_weights(total_bkg, self.bin_edges)
        w_sig = calculate_weights(training_pt_dist['Zee'], self.bin_edges)
        
        X = np.concatenate([X_sig, X_bkg], axis=0)
        y = np.concatenate([y_sig, y_bkg], axis=0)
        w = np.concatenate([w_sig, w_bkg], axis=0)
        # print("y:", y.shape)
        # print("w:", w.shape)

        X_cnn_train, X_cnn_test, y_train, y_test, w_train, w_test = train_test_split(
            X,
            y,
            w,
            test_size=0.2,
            random_state=101,
        )
        # print("x_train:", X_cnn_train.shape)

        self.x_train = X_cnn_train
        self.y_train = y_train
        self.x_val   = X_cnn_test
        self.y_val   = y_test
        self.w_train = w_train 
        self.w_val   = w_test

    def save_to_parquet(self, train_path, val_path):
        train_data = {
        "x_train": self.x_train,
        "y_train": self.y_train,
        "w_train": self.w_train
        }

        train_data = ak.zip(train_data, depth_limit=1)
        ak.to_parquet(train_data, train_path)
        
        val_data = {
        "x_val": self.x_val,
        "y_val": self.y_val,
        "w_val": self.w_val,
        }

        val_data = ak.zip(val_data, depth_limit=1)
        ak.to_parquet(val_data, val_path)

    def save_to_root(self, train_path, val_path):
        safe_train_columns = write_root_input_diagnostics(
            self.x_train,
            self.y_train,
            self.w_train,
            log_directory="root_diagnostics/train",
        )

        if safe_train_columns is None:
            raise RuntimeError(
                "Input preparation failed. See root_diagnostics/train/diagnosis.log"
            )
        
        result = isolated_root_write(
            safe_train_columns,
            output_path=train_path,
            tree_name="train",
            work_directory="root_diagnostics/train",
        )

        if result.returncode < 0:
            # On Unix, a negative return code indicates termination by a signal.
            signal_number = -result.returncode
            print(f"ROOT child process was killed by signal {signal_number}.")
        elif result.returncode != 0:
            print(f"ROOT child process failed with code {result.returncode}.")
        else:
            print("ROOT write process completed.")

        safe_val_columns = write_root_input_diagnostics(
            self.x_val,
            self.y_val,
            self.w_val,
            log_directory="root_diagnostics/val",
        )

        if safe_val_columns is None:
            raise RuntimeError(
                "Input preparation failed. See root_diagnostics/val/diagnosis.log"
            )
        
        result = isolated_root_write(
            safe_val_columns,
            output_path=val_path,
            tree_name="val",
            work_directory="root_diagnostics/val",
        )

        if result.returncode < 0:
            # On Unix, a negative return code indicates termination by a signal.
            signal_number = -result.returncode
            print(f"ROOT child process was killed by signal {signal_number}.")
        elif result.returncode != 0:
            print(f"ROOT child process failed with code {result.returncode}.")
        else:
            print("ROOT write process completed.")

import datetime
import faulthandler
import gc
import json
import logging
import os
import platform
import sys
import traceback
from pathlib import Path
from typing import Any

import awkward as ak
import numpy as np


def _json_safe(value: Any) -> Any:
    """Convert NumPy/Python values into JSON-serializable values."""
    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, tuple):
        return list(value)

    return value


def _flush_file(file_object) -> None:
    """Flush Python and operating-system buffers."""
    file_object.flush()

    try:
        os.fsync(file_object.fileno())
    except (AttributeError, OSError):
        pass


def inspect_numpy_array(
    name: str,
    array: Any,
    *,
    sample_rows: int = 3,
) -> dict[str, Any]:
    """
    Inspect an array without constructing an RDataFrame.

    This function does not modify the original array.
    """
    result: dict[str, Any] = {
        "name": name,
        "python_type": type(array).__name__,
        "python_module": type(array).__module__,
    }

    try:
        arr = np.asarray(array)
    except Exception as error:
        result["numpy_conversion_error"] = repr(error)
        return result

    result.update(
        {
            "shape": list(arr.shape),
            "ndim": int(arr.ndim),
            "length": int(len(arr)) if arr.ndim > 0 else None,
            "dtype": str(arr.dtype),
            "dtype_kind": arr.dtype.kind,
            "itemsize": int(arr.dtype.itemsize),
            "byteorder": arr.dtype.byteorder,
            "nbytes": int(arr.nbytes),
            "c_contiguous": bool(arr.flags.c_contiguous),
            "f_contiguous": bool(arr.flags.f_contiguous),
            "aligned": bool(arr.flags.aligned),
            "writeable": bool(arr.flags.writeable),
            "owns_data": bool(arr.flags.owndata),
            "has_object_dtype": bool(arr.dtype == object),
            "strides": list(arr.strides),
        }
    )

    if arr.ndim == 0:
        result["scalar_value"] = _json_safe(arr.item())
        return result

    # Check numeric data without producing large temporary arrays.
    if np.issubdtype(arr.dtype, np.number):
        try:
            result["has_nan"] = bool(np.isnan(arr).any())
        except TypeError:
            result["has_nan"] = None

        try:
            result["has_inf"] = bool(np.isinf(arr).any())
        except TypeError:
            result["has_inf"] = None

        if arr.size > 0:
            try:
                result["minimum"] = _json_safe(np.min(arr))
                result["maximum"] = _json_safe(np.max(arr))
            except Exception as error:
                result["min_max_error"] = repr(error)

    # Record a few rows, but avoid writing the full dataset.
    samples = []

    for index in range(min(sample_rows, len(arr))):
        value = arr[index]

        sample = {
            "index": index,
            "python_type": type(value).__name__,
        }

        if isinstance(value, np.ndarray):
            sample.update(
                {
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                    "c_contiguous": bool(value.flags.c_contiguous),
                    "strides": list(value.strides),
                }
            )

            # Keep previews intentionally small.
            sample["preview"] = value.reshape(-1)[:10].tolist()
        else:
            try:
                sample["value"] = _json_safe(value)
            except Exception:
                sample["repr"] = repr(value)

        samples.append(sample)

    result["samples"] = samples

    return result


def inspect_awkward_conversion(
    name: str,
    array: Any,
) -> dict[str, Any]:
    """
    Convert one input column to Awkward and inspect its layout.

    This calls ak.Array, but does not call ak.to_rdataframe or any ROOT
    RDataFrame operation.
    """
    result: dict[str, Any] = {"name": name}

    try:
        awkward_array = ak.Array(array)

        result.update(
            {
                "awkward_type": str(ak.type(awkward_array)),
                "fields": list(ak.fields(awkward_array)),
                "length": len(awkward_array),
                "layout_class": type(ak.to_layout(awkward_array)).__name__,
                "form": awkward_array.layout.form.to_dict(),
            }
        )

        # Validate buffers/layout consistency.
        try:
            error = ak.validity_error(awkward_array)
            result["validity_error"] = error or None
            result["is_valid"] = not bool(error)
        except Exception as validity_exception:
            result["validity_check_error"] = repr(validity_exception)

        # Test conversion of a tiny slice only.
        try:
            result["first_rows"] = ak.to_list(awkward_array[:3])
        except Exception as preview_exception:
            result["preview_error"] = repr(preview_exception)

    except Exception as error:
        result["conversion_error"] = repr(error)
        result["traceback"] = traceback.format_exc()

    return result


def predicted_root_rdf_type(array: Any) -> str:
    """
    Give an approximate type that ak.to_rdataframe is likely to expose.

    This is diagnostic only. It does not query ROOT and is not guaranteed
    to reproduce every Awkward/ROOT type-mapping detail.
    """
    arr = np.asarray(array)

    dtype_map = {
        np.dtype("bool"): "bool",
        np.dtype("int8"): "int8_t",
        np.dtype("uint8"): "uint8_t",
        np.dtype("int16"): "int16_t",
        np.dtype("uint16"): "uint16_t",
        np.dtype("int32"): "int32_t",
        np.dtype("uint32"): "uint32_t",
        np.dtype("int64"): "int64_t",
        np.dtype("uint64"): "uint64_t",
        np.dtype("float32"): "float",
        np.dtype("float64"): "double",
    }

    scalar_type = dtype_map.get(arr.dtype, f"unmapped<{arr.dtype}>")

    if arr.ndim <= 1:
        return scalar_type

    result = scalar_type

    # Every dimension after the event dimension may become a nested RVec
    # or another collection representation.
    for _ in arr.shape[1:]:
        result = f"ROOT::VecOps::RVec<{result}>"

    return result


def propose_safe_columns(
    x: Any,
    y: Any,
    w: Any,
) -> tuple[dict[str, np.ndarray], list[str]]:
    """
    Build conservative candidate arrays without touching RDataFrame.

    The returned arrays remain alive as ordinary NumPy objects.
    """
    notes: list[str] = []

    x_arr = np.asarray(x)
    y_arr = np.asarray(y)
    w_arr = np.asarray(w)

    if x_arr.ndim < 2:
        notes.append(
            f"x has ndim={x_arr.ndim}; expected an event dimension plus features."
        )

    if y_arr.ndim == 0:
        raise ValueError("y is scalar and has no event dimension.")

    if w_arr.ndim == 0:
        raise ValueError("w is scalar and has no event dimension.")

    lengths = {
        "x": len(x_arr),
        "y": len(y_arr),
        "w": len(w_arr),
    }

    if len(set(lengths.values())) != 1:
        raise ValueError(f"Outer lengths differ: {lengths}")

    # Flatten all feature dimensions into one fixed-size vector per event.
    x_safe = np.ascontiguousarray(
        x_arr.reshape(len(x_arr), -1),
        dtype=np.float64,
    )

    # Require exactly one label value per event.
    y_matrix = y_arr.reshape(len(y_arr), -1)

    if y_matrix.shape[1] != 1:
        raise ValueError(
            "y does not contain exactly one value per event after flattening: "
            f"original shape={y_arr.shape}, flattened shape={y_matrix.shape}"
        )

    # int32 avoids the ambiguity around 8-bit integer branch handling.
    y_safe = np.ascontiguousarray(
        y_matrix[:, 0],
        dtype=np.int32,
    )

    # Require exactly one weight per event.
    w_matrix = w_arr.reshape(len(w_arr), -1)

    if w_matrix.shape[1] != 1:
        raise ValueError(
            "w does not contain exactly one value per event after flattening: "
            f"original shape={w_arr.shape}, flattened shape={w_matrix.shape}"
        )

    w_safe = np.ascontiguousarray(
        w_matrix[:, 0],
        dtype=np.float64,
    )

    if y_arr.dtype != y_safe.dtype:
        notes.append(f"Converted y from {y_arr.dtype} to {y_safe.dtype}.")

    if w_arr.dtype != w_safe.dtype:
        notes.append(f"Converted w from {w_arr.dtype} to {w_safe.dtype}.")

    if x_arr.shape != x_safe.shape:
        notes.append(f"Flattened x from {x_arr.shape} to {x_safe.shape}.")

    if y_arr.shape != y_safe.shape:
        notes.append(f"Collapsed y from {y_arr.shape} to {y_safe.shape}.")

    return (
        {
            "x_train": x_safe,
            "y_train": y_safe,
            "w_train": w_safe,
        },
        notes,
    )


def write_root_input_diagnostics(
    x: Any,
    y: Any,
    w: Any,
    *,
    log_directory: str | Path = "root_diagnostics",
) -> dict[str, np.ndarray] | None:
    """
    Write detailed diagnostics before any RDataFrame operation.

    Returns conservative candidate arrays when preparation succeeds.
    """
    log_directory = Path(log_directory)
    log_directory.mkdir(parents=True, exist_ok=True)

    text_path = log_directory / "diagnosis.log"
    json_path = log_directory / "diagnosis.json"
    fault_path = log_directory / "python_fault.log"

    # Line buffering ensures each completed line is written promptly.
    text_file = text_path.open("w", encoding="utf-8", buffering=1)
    fault_file = fault_path.open("w", encoding="utf-8", buffering=1)

    # This can record Python-level stack information for fatal signals.
    faulthandler.enable(file=fault_file, all_threads=True)

    logger = logging.getLogger("root-diagnostics")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    logger.propagate = False

    handler = logging.StreamHandler(text_file)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s"
        )
    )
    logger.addHandler(handler)

    report: dict[str, Any] = {
        "timestamp_utc": datetime.datetime.now(
            datetime.timezone.utc
        ).isoformat(),
        "environment": {},
        "original_columns": {},
        "candidate_columns": {},
        "notes": [],
        "exceptions": [],
    }

    candidate_columns: dict[str, np.ndarray] | None = None

    try:
        logger.info("Starting crash-safe diagnostics.")
        _flush_file(text_file)

        environment = {
            "python_version": sys.version,
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "numpy_version": np.__version__,
            "awkward_version": ak.__version__,
            "pid": os.getpid(),
            "working_directory": os.getcwd(),
        }

        # Importing ROOT is normally safe and does not construct an RDF.
        try:
            import ROOT

            environment.update(
                {
                    "root_version": ROOT.gROOT.GetVersion(),
                    "root_version_int": int(ROOT.gROOT.GetVersionInt()),
                    "implicit_mt_enabled": bool(ROOT.IsImplicitMTEnabled()),
                }
            )
        except Exception as error:
            environment["root_import_error"] = repr(error)

        report["environment"] = environment

        for key, value in environment.items():
            logger.info("Environment %s: %s", key, value)

        _flush_file(text_file)

        original = {
            "x_train": x,
            "y_train": y,
            "w_train": w,
        }

        for name, values in original.items():
            numpy_info = inspect_numpy_array(name, values)
            awkward_info = inspect_awkward_conversion(name, values)

            entry = {
                "numpy": numpy_info,
                "awkward": awkward_info,
            }

            try:
                entry["predicted_rdf_type"] = predicted_root_rdf_type(values)
            except Exception as error:
                entry["predicted_rdf_type_error"] = repr(error)

            report["original_columns"][name] = entry

            logger.info(
                "Original %s: shape=%s dtype=%s predicted_type=%s",
                name,
                numpy_info.get("shape"),
                numpy_info.get("dtype"),
                entry.get("predicted_rdf_type"),
            )

            logger.info(
                "Original %s Awkward type: %s",
                name,
                awkward_info.get("awkward_type"),
            )

            _flush_file(text_file)

        logger.info("Preparing conservative candidate columns.")

        candidate_columns, notes = propose_safe_columns(x, y, w)
        report["notes"].extend(notes)

        for note in notes:
            logger.info("Preparation note: %s", note)

        for name, values in candidate_columns.items():
            numpy_info = inspect_numpy_array(name, values)
            awkward_info = inspect_awkward_conversion(name, values)

            entry = {
                "numpy": numpy_info,
                "awkward": awkward_info,
                "predicted_rdf_type": predicted_root_rdf_type(values),
            }

            report["candidate_columns"][name] = entry

            logger.info(
                "Candidate %s: shape=%s dtype=%s predicted_type=%s",
                name,
                numpy_info.get("shape"),
                numpy_info.get("dtype"),
                entry["predicted_rdf_type"],
            )

            logger.info(
                "Candidate %s Awkward type: %s",
                name,
                awkward_info.get("awkward_type"),
            )

            _flush_file(text_file)

        # Additional cross-column checks.
        candidate_lengths = {
            name: len(values)
            for name, values in candidate_columns.items()
        }

        report["candidate_lengths"] = candidate_lengths
        report["candidate_lengths_equal"] = (
            len(set(candidate_lengths.values())) == 1
        )

        logger.info("Candidate lengths: %s", candidate_lengths)

        if not report["candidate_lengths_equal"]:
            logger.error("Candidate outer lengths are not equal.")

        # Explicitly record data-lifetime information.
        report["candidate_reference_counts"] = {
            name: sys.getrefcount(values)
            for name, values in candidate_columns.items()
        }

        report["garbage_collector_enabled"] = gc.isenabled()

        logger.info(
            "No RDataFrame was constructed during this diagnostic run."
        )

    except Exception as error:
        exception_info = {
            "type": type(error).__name__,
            "message": str(error),
            "repr": repr(error),
            "traceback": traceback.format_exc(),
        }

        report["exceptions"].append(exception_info)

        logger.exception("Diagnostic preparation failed.")

    finally:
        # Write JSON atomically, so an interrupted write does not leave the
        # only report partially overwritten.
        temporary_json_path = json_path.with_suffix(".json.tmp")

        with temporary_json_path.open(
            "w",
            encoding="utf-8",
        ) as json_file:
            json.dump(
                report,
                json_file,
                indent=2,
                default=_json_safe,
            )
            json_file.write("\n")
            _flush_file(json_file)

        os.replace(temporary_json_path, json_path)

        logger.info("JSON report written to %s", json_path)
        logger.info("Text report written to %s", text_path)
        logger.info("Fault report configured at %s", fault_path)
        logger.info("Diagnostics complete.")

        _flush_file(text_file)
        _flush_file(fault_file)

        # Keep faulthandler active only while its output file is open.
        faulthandler.disable()

        for active_handler in list(logger.handlers):
            active_handler.flush()
            active_handler.close()
            logger.removeHandler(active_handler)

        text_file.close()
        fault_file.close()

    return candidate_columns


import subprocess


def isolated_root_write(
    columns: dict[str, np.ndarray],
    *,
    output_path: str,
    tree_name: str = "train",
    work_directory: str = "root_diagnostics/train",
) -> subprocess.CompletedProcess:
    work_directory = Path(work_directory)
    work_directory.mkdir(parents=True, exist_ok=True)

    input_path = work_directory / "prepared_columns.npz"
    child_log_path = work_directory / "root_write_child.log"
    parent_log_path = work_directory / "root_write_parent.json"

    # Save independent copies, eliminating dependence on the lifetime or
    # ownership of arrays in the parent process.
    np.savez(
        input_path,
        x_train=np.ascontiguousarray(columns["x_train"]),
        y_train=np.ascontiguousarray(columns["y_train"]),
        w_train=np.ascontiguousarray(columns["w_train"]),
    )

    command = [
        sys.executable,
        "-X",
        "faulthandler",
        "root_write_child.py",
        "--input",
        str(input_path),
        "--output",
        str(output_path),
        "--tree",
        tree_name,
        "--log",
        str(child_log_path),
    ]

    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )

    parent_report = {
        "command": command,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "child_log": str(child_log_path),
        "output_path": output_path,
        "output_exists": Path(output_path).exists(),
        "output_size": (
            Path(output_path).stat().st_size
            if Path(output_path).exists()
            else None
        ),
    }

    temporary_path = parent_log_path.with_suffix(".json.tmp")

    with temporary_path.open("w", encoding="utf-8") as file_object:
        json.dump(parent_report, file_object, indent=2)
        file_object.write("\n")
        file_object.flush()
        os.fsync(file_object.fileno())

    os.replace(temporary_path, parent_log_path)

    return result

if __name__ == "__main__":
    x = time.time()
    ods = OpenDataSet()
    ods.load()
    ods.save_to_root(
        "/home/student2/ofml_workspace/OpenFastML/train_data/ak_big_train_data.root", 
        "/home/student2/ofml_workspace/OpenFastML/train_data/ak_big_val_data.root"
        )
    print(time.time()-x)