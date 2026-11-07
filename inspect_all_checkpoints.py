"""
Batch PyTorch checkpoint inspector for the Mars models.

Run:
    python inspect_checkpoint.py

By default, all paths in CHECKPOINT_PATHS are inspected and one combined
checkpoint_inspection_report.txt is created.

You may also pass paths:
    python inspect_checkpoint.py "D:\\mars_app\\model1.pth" "D:\\mars_app\\model2.pth"

Optional:
    python inspect_checkpoint.py --output "D:\\mars_app\\report.txt"
"""
import os

# Temporary workaround for the duplicate OpenMP DLL error in this environment.
# This is not a true fix and may cause instability. A clean Conda environment
# is preferable. Set before importing torch.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
from collections.abc import Mapping
from pathlib import Path
import sys
import torch

# Edit this list if you add/remove checkpoint files.
CHECKPOINT_PATHS = [
    r"D:\mars_app\checkpoints\adaptive_y_net_mars.pth",
    r"D:\mars_app\checkpoints\binary_y_net_mars.pth",
    r"D:\mars_app\checkpoints\CCHNet_FP16.pth",
    r"D:\mars_app\checkpoints\unet_cbam.pth",
    r"D:\mars_app\checkpoints\y_net_mars.pth",
    r"D:\mars_app\checkpoints\unet.pth",
]
DEFAULT_REPORT = "checkpoint_inspection_report.txt"


def is_state_dict(obj):
    return isinstance(obj, Mapping) and len(obj) > 0 and all(
        isinstance(v, torch.Tensor) for v in obj.values()
    )


def find_state_dict(obj):
    if is_state_dict(obj):
        return "<root>", obj
    if isinstance(obj, Mapping):
        for key in ("model_state_dict", "state_dict", "model", "net",
                    "network", "weights", "ema_state_dict"):
            if is_state_dict(obj.get(key)):
                return str(key), obj[key]
        for key, value in obj.items():
            if is_state_dict(value):
                return str(key), value
    return None, None


def describe(value):
    if isinstance(value, torch.Tensor):
        return f"Tensor shape={tuple(value.shape)}, dtype={value.dtype}"
    if isinstance(value, Mapping):
        return f"{type(value).__name__} with {len(value)} entries"
    if isinstance(value, (list, tuple)):
        return f"{type(value).__name__} with {len(value)} items"
    return f"{type(value).__name__}: {str(value)[:160]}"


def class_metadata(checkpoint):
    found = []
    if isinstance(checkpoint, Mapping):
        for key in ("num_classes", "n_classes", "class_names",
                    "classes_names", "classes", "label_names"):
            if key in checkpoint:
                value = checkpoint[key]
                found.append(f"{key}: {value!r}")
    return found


def likely_output_layers(state_dict):
    if state_dict is None:
        return []
    keywords = ("out", "output", "classifier", "segmentation_head",
                "seg_head", "head", "logits", "final", "last")
    found = []
    for name, tensor in state_dict.items():
        low = str(name).lower()
        if (isinstance(tensor, torch.Tensor) and tensor.ndim >= 1
                and low.endswith("weight")
                and any(word in low for word in keywords)):
            found.append((str(name), tuple(tensor.shape), int(tensor.shape[0])))
    return found


def inspect_checkpoint(path_string):
    result = []
    add = result.append
    path = Path(path_string).expanduser()

    add("=" * 88)
    add(f"CHECKPOINT: {path}")
    add("=" * 88)

    if not path.is_file():
        add("STATUS: FILE NOT FOUND")
        return result

    add(f"Resolved path: {path.resolve()}")
    add(f"File size: {path.stat().st_size:,} bytes")
    add(f"PyTorch version: {torch.__version__}")
    add("Loading device: CPU")

    try:
        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            add("STATUS: PyTorch lacks weights_only support; upgrade PyTorch.")
            return result
        except Exception as exc:
            add(f"STATUS: SAFE LOAD FAILED: {type(exc).__name__}: {exc}")
            add("No unrestricted pickle-loading fallback was attempted.")
            return result

        add("\n[1] TOP-LEVEL STRUCTURE")
        add(f"Object type: {type(checkpoint).__name__}")
        if isinstance(checkpoint, Mapping):
            add(f"Top-level entries: {len(checkpoint)}")
            for key, value in checkpoint.items():
                add(f"  {key}: {describe(value)}")
        else:
            add(f"Value preview: {str(checkpoint)[:500]}")

        state_key, state_dict = find_state_dict(checkpoint)
        add("\n[2] STATE DICTIONARY / TENSOR SHAPES")
        if state_dict is None:
            add("No conventional tensor-only state_dict detected.")
        else:
            add(f"State dictionary found under: {state_key}")
            add(f"Tensor count: {len(state_dict)}")
            add(f"Total tensor elements: {sum(t.numel() for t in state_dict.values()):,}")
            for name, tensor in state_dict.items():
                add(f"  {name}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}")

        add("\n[3] CLASS COUNT")
        metadata = class_metadata(checkpoint)
        if metadata:
            add("Explicit class-related metadata:")
            for item in metadata:
                add(f"  {item}")
        else:
            add("No explicit standard class-count metadata found.")

        candidates = likely_output_layers(state_dict)
        if candidates:
            add("Likely output-head candidates (heuristic):")
            for name, shape, count in candidates:
                add(f"  {name}: shape={shape} -> likely {count} output channels/classes")
            counts = sorted({item[2] for item in candidates})
            if len(counts) == 1:
                add(f"LIKELY CLASS COUNT: {counts[0]}")
            else:
                add(f"AMBIGUOUS OUTPUT SIZES: {counts}")
                add("Check which candidate is the actual final output layer.")
        else:
            add("LIKELY CLASS COUNT: UNKNOWN")
            add("Could not identify an output layer from tensor names.")

        add("\n[4] INTERPRETATION")
        add("- Conv2d weight shape [C_out, C_in, kH, kW]: C_out is usually output channels.")
        add("- Linear weight shape [C_out, C_in]: C_out is usually output units.")
        add("- Output channels do not always equal semantic classes.")
        add("- Class names/order generally require the training label mapping.")
        add("- Tensor shapes alone may not uniquely identify the architecture.")
        add("STATUS: INSPECTION COMPLETED")

    except Exception as exc:
        add(f"STATUS: INSPECTION ERROR: {type(exc).__name__}: {exc}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Inspect multiple Mars .pth/.pt checkpoints.")
    parser.add_argument("checkpoints", nargs="*", help="Optional checkpoint paths.")
    parser.add_argument("--output", default=DEFAULT_REPORT, help="Combined text report path.")
    args = parser.parse_args()

    paths = args.checkpoints or CHECKPOINT_PATHS
    report = [
        "MARS MODEL CHECKPOINT INSPECTION REPORT",
        f"PyTorch version: {torch.__version__}",
        f"Requested checkpoint count: {len(paths)}",
        "Class counts inferred from layer shapes are heuristics; verify against model code.",
        "",
    ]

    for index, path in enumerate(paths, 1):
        print(f"[{index}/{len(paths)}] Inspecting {path}")
        report.extend(inspect_checkpoint(path))
        report.append("")

    output = Path(args.output).expanduser()
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("\n".join(report), encoding="utf-8")
    except OSError as exc:
        print(f"Could not save report: {exc}")
        sys.exit(1)

    print(f"\nDone. Combined report saved to: {output.resolve()}")


if __name__ == "__main__":
    main()
