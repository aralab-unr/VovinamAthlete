#!/usr/bin/env python3
"""Convert GMR-retargeted motion pickle(s) to numeric CSV for csv_to_npz.py.

Column layout written: [root_pos(3), root_rot(4), dof_pos(D)], no header —
matches what scripts/csv_to_npz.py's MotionLoader expects (root_rot defaults
to xyzw order, converted to wxyz internally by csv_to_npz.py).

Single file:

.. code-block:: bash

    python scripts/pkl_to_csv.py path/to/motion.pkl \\
      -o vovinamathlete_mjlab/assets/motions/vd03/csvdata/motion.csv

Whole directory (batch), converting every '*.pkl' file found -- output
filenames keep the input stem exactly (pass --glob to filter, e.g. only
files matching a specific naming pattern):

.. code-block:: bash

    python scripts/pkl_to_csv.py \\
      --input_dir path/to/gmr_output \\
      --output_dir vovinamathlete_mjlab/assets/motions/vd03/csvdata

Inspect a pickle's structure without converting:

.. code-block:: bash

    python scripts/pkl_to_csv.py path/to/motion.pkl --inspect
"""

import sys
import pickle
import argparse
from pathlib import Path

import numpy as np


class _NumpyCompatUnpickler(pickle.Unpickler):
    """Load pickles written with numpy>=2.0 (module 'numpy._core') under numpy<2.0."""

    def find_class(self, module, name):
        if module.startswith("numpy._core"):
            module = "numpy.core" + module[len("numpy._core"):]
        return super().find_class(module, name)


def load_pickle(path):
    """Unpickle, tolerating numpy 2.x <-> 1.x module path differences."""
    with open(path, "rb") as f:
        return _NumpyCompatUnpickler(f).load()


def inspect_object(obj):
    print(f"type={type(obj)}")
    if isinstance(obj, dict):
        print("keys =", list(obj.keys()))
        for k, v in obj.items():
            if isinstance(v, np.ndarray):
                print(f"  {k}: ndarray shape={v.shape}, dtype={v.dtype}")
            else:
                print(f"  {k}: {type(v)} -> {v}")


def convert_one(input_path, output_path, quat_order, output_quat_order, fmt, cut_start=0, cut_end=0):
    """Convert a single motion pickle to a numeric CSV. Returns the frame count."""
    obj = load_pickle(input_path)

    if not isinstance(obj, dict):
        raise ValueError(f"{input_path}: expected pickle to contain a dict.")

    required_keys = ["root_pos", "root_rot", "dof_pos"]
    for k in required_keys:
        if k not in obj:
            raise ValueError(f"{input_path}: missing required key '{k}'")

    root_pos = np.asarray(obj["root_pos"], dtype=np.float64)
    root_rot = np.asarray(obj["root_rot"], dtype=np.float64)
    dof_pos = np.asarray(obj["dof_pos"], dtype=np.float64)

    if root_pos.ndim != 2 or root_pos.shape[1] != 3:
        raise ValueError(f"{input_path}: root_pos should have shape [N, 3], got {root_pos.shape}")
    if root_rot.ndim != 2 or root_rot.shape[1] != 4:
        raise ValueError(f"{input_path}: root_rot should have shape [N, 4], got {root_rot.shape}")
    if dof_pos.ndim != 2:
        raise ValueError(f"{input_path}: dof_pos should have shape [N, D], got {dof_pos.shape}")

    n = root_pos.shape[0]
    if root_rot.shape[0] != n or dof_pos.shape[0] != n:
        raise ValueError(
            f"{input_path}: frame count mismatch: "
            f"root_pos={root_pos.shape[0]}, root_rot={root_rot.shape[0]}, dof_pos={dof_pos.shape[0]}"
        )

    # Apply frame cuts
    end_idx = n - cut_end if cut_end > 0 else n
    root_pos = root_pos[cut_start:end_idx]
    root_rot = root_rot[cut_start:end_idx]
    dof_pos  = dof_pos[cut_start:end_idx]
    n = root_pos.shape[0]
    if cut_start or cut_end:
        print(f"  Cut: -{cut_start} start frames, -{cut_end} end frames → {n} frames remaining")

    # Reorder quaternion if needed
    # xyzw -> wxyz  : [x,y,z,w] -> [w,x,y,z]
    # wxyz -> xyzw  : [w,x,y,z] -> [x,y,z,w]
    if quat_order != output_quat_order:
        if quat_order == "xyzw" and output_quat_order == "wxyz":
            root_rot = root_rot[:, [3, 0, 1, 2]]
        elif quat_order == "wxyz" and output_quat_order == "xyzw":
            root_rot = root_rot[:, [1, 2, 3, 0]]

    motion = np.concatenate([root_pos, root_rot, dof_pos], axis=1)
    np.savetxt(output_path, motion, delimiter=",", fmt=fmt)
    return n


def main():
    parser = argparse.ArgumentParser(
        description="Convert GMR-retargeted motion pickle(s) to a csv_to_npz.py-compatible numeric CSV."
    )
    parser.add_argument("input_pkl", nargs="?", help="Input pickle file (single-file mode)")
    parser.add_argument("-o", "--output_csv", help="Output CSV file")
    parser.add_argument(
        "--input_dir",
        help="Batch mode: directory searched recursively for pickle files (see --glob).",
    )
    parser.add_argument(
        "--glob",
        default="*.pkl",
        help="Batch mode: filename glob to match under --input_dir (default: '*.pkl').",
    )
    parser.add_argument(
        "--output_dir",
        help="Batch mode: directory to write '<stem>.csv' files into (created if missing).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Batch mode: re-convert files whose output CSV already exists.",
    )
    parser.add_argument("--inspect", action="store_true", help="Inspect pickle structure only")
    parser.add_argument(
        "--quat-order",
        choices=["xyzw", "wxyz"],
        default="xyzw",
        help="Quaternion order stored in pickle root_rot (default: xyzw)",
    )
    parser.add_argument(
        "--output-quat-order",
        choices=["xyzw", "wxyz"],
        default="xyzw",
        help="Quaternion order to write into CSV (default: xyzw, matching csv_to_npz.py's expected input)",
    )
    parser.add_argument(
        "--fmt",
        default="%.10f",
        help="Float format for CSV output (default: %%.10f)",
    )
    parser.add_argument(
        "--cut_start",
        type=int,
        default=0,
        metavar="FRAMES",
        help="Number of frames to remove from the beginning (e.g. 660 for 5.5s at 120fps).",
    )
    parser.add_argument(
        "--cut_end",
        type=int,
        default=0,
        metavar="FRAMES",
        help="Number of frames to remove from the end (e.g. 300 for 2.5s at 120fps).",
    )
    args = parser.parse_args()

    # ---- batch mode: convert every file matching --glob under --input_dir,
    # keeping each output CSV's name exactly the same as its input stem ----
    if args.input_dir:
        if not args.output_dir:
            print("Error: --output_dir is required with --input_dir")
            sys.exit(1)
        input_dir = Path(args.input_dir).expanduser().resolve()
        output_dir = Path(args.output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        pkl_files = sorted(input_dir.rglob(args.glob))
        print(f"Found {len(pkl_files)} '{args.glob}' files under {input_dir}")
        converted = skipped = failed = 0
        for i, pkl_path in enumerate(pkl_files, 1):
            stem = pkl_path.stem
            csv_path = output_dir / f"{stem}.csv"
            if csv_path.exists() and not args.overwrite:
                skipped += 1
                continue
            try:
                n = convert_one(pkl_path, csv_path, args.quat_order, args.output_quat_order, args.fmt,
                                cut_start=args.cut_start, cut_end=args.cut_end)
                converted += 1
                print(f"[{i}/{len(pkl_files)}] {stem}.csv ({n} frames)")
            except Exception as exc:  # one bad file must not abort the batch
                failed += 1
                print(f"[{i}/{len(pkl_files)}] FAILED {pkl_path.name}: {exc}")
        print(f"\nDone. converted={converted}, skipped(existing)={skipped}, failed={failed}")
        print(f"CSV column layout: [root_pos(3), root_rot(4), dof_pos(D)], quaternion={args.output_quat_order}")
        return

    # ---- single-file mode ----
    if not args.input_pkl:
        print("Error: provide an input pickle, or use --input_dir/--output_dir for batch mode.")
        sys.exit(1)

    input_path = Path(args.input_pkl).expanduser().resolve()
    if not input_path.exists():
        print(f"Error: file not found: {input_path}")
        sys.exit(1)

    output_path = Path(args.output_csv).expanduser().resolve() if args.output_csv else input_path.with_suffix(".csv")

    if args.inspect:
        obj = load_pickle(input_path)
        print(f"Loaded: {input_path}")
        inspect_object(obj)
        return

    n = convert_one(input_path, output_path, args.quat_order, args.output_quat_order, args.fmt,
                    cut_start=args.cut_start, cut_end=args.cut_end)

    print(f"Loaded: {input_path}")
    print(f"final motion frames: {n}")
    print(f"Saved numeric CSV: {output_path}")
    print("Column layout: [root_pos(3), root_rot(4), dof_pos(D)]")
    print("No header written.")
    print(f"Quaternion written as: {args.output_quat_order}")


if __name__ == "__main__":
    main()
