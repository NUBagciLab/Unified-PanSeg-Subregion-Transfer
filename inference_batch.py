"""
Batch nnUNet inference: segment every NIfTI in an input folder and write to an output folder
with the same filenames. Uses CleanInference from simple_inference (nnUNetSubregionTrainer).

CUDA only: select GPU by integer index (--device).
"""

import argparse
import os
import sys
from typing import List

import torch

_toolkit_dir = os.path.dirname(os.path.abspath(__file__))
if _toolkit_dir not in sys.path:
    sys.path.insert(0, _toolkit_dir)

from simple_inference import CleanInference


def cuda_device(gpu_id: int) -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required but not available.")
    if gpu_id < 0 or gpu_id >= torch.cuda.device_count():
        raise ValueError(
            f"Invalid GPU id {gpu_id}; valid range is 0..{torch.cuda.device_count() - 1}"
        )
    return torch.device(f"cuda:{gpu_id}")


def list_nifti_files(input_dir: str) -> List[str]:
    """Sorted paths to .nii / .nii.gz files in a directory (non-recursive)."""
    paths: List[str] = []
    for name in sorted(os.listdir(input_dir)):
        p = os.path.join(input_dir, name)
        if not os.path.isfile(p):
            continue
        lower = name.lower()
        if lower.endswith(".nii.gz") or lower.endswith(".nii"):
            paths.append(p)
    return paths


def run_batch_inference(
    input_dir: str,
    output_dir: str,
    model_folder: str,
    gpu_id: int = 0,
    verbose: bool = True,
) -> None:
    """
    Segment each .nii / .nii.gz in ``input_dir`` and save under ``output_dir`` with the same basename.

    Args:
        input_dir: Folder of input volumes
        output_dir: Folder for outputs (created if missing)
        model_folder: nnUNet model directory (checkpoint_best.pth, dataset.json, plans.json)
        gpu_id: CUDA device index
        verbose: Print per-file progress
    """
    if not os.path.isdir(input_dir):
        raise NotADirectoryError(f"Not a directory: {input_dir}")

    files = list_nifti_files(input_dir)
    if not files:
        raise FileNotFoundError(f"No .nii or .nii.gz files found in: {input_dir}")

    os.makedirs(output_dir, exist_ok=True)

    device = cuda_device(gpu_id)
    inference = CleanInference(
        model_folder=model_folder,
        device=device,
        verbose=verbose,
    )

    n = len(files)
    for i, input_path in enumerate(files, start=1):
        out_path = os.path.join(output_dir, os.path.basename(input_path))
        if verbose:
            print(f"\n[{i}/{n}] {input_path} -> {out_path}")
        inference.predict_single_file(
            input_file=input_path,
            output_file=out_path,
            save_probabilities=False,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Batch nnUNet segmentation: all .nii/.nii.gz in input dir -> output dir (same names)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python inference_batch.py -i /data/in -o /data/out -m Model -d 0
        """,
    )
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="Input directory containing NIfTI files",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Output directory for segmentations (created if needed)",
    )
    parser.add_argument(
        "-m",
        "--model",
        default="Model",
        help="Model folder (checkpoint_best.pth, dataset.json, plans.json)",
    )
    parser.add_argument(
        "-d",
        "--device",
        type=int,
        default=0,
        metavar="GPU_ID",
        help="CUDA GPU index (e.g. 0 for the first GPU)",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    if os.path.exists(args.output) and not os.path.isdir(args.output):
        parser.error("--output must be a directory path (not a file).")

    run_batch_inference(
        input_dir=args.input,
        output_dir=args.output,
        model_folder=args.model,
        gpu_id=args.device,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
