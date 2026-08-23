#!/usr/bin/env python3
"""Measure cross-expert shared-basis energy in a Granite safetensors checkpoint.

This is an offline candidate-generation tool. It does not modify model weights
or claim that a low-rank reconstruction preserves model quality.
"""

import argparse
import json
import struct
from pathlib import Path

import numpy as np


LAYERS = 24
EXPERTS = 32


def load_header(path: Path):
    with path.open("rb") as handle:
        size = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(size))
    return header, 8 + size


def bf16_array(raw: memoryview, offset: int, shape):
    count = int(np.prod(shape))
    values = np.frombuffer(raw, dtype="<u2", count=count, offset=offset)
    widened = (values.astype(np.uint32) << 16).view(np.float32)
    return widened.reshape(shape)


def basis_fractions(weights: np.ndarray):
    matrix = weights.reshape(EXPERTS, -1).astype(np.float32, copy=False)
    mean = matrix.mean(axis=0, dtype=np.float32)
    centered = matrix - mean
    total = float(np.sum(matrix * matrix, dtype=np.float64))
    shared = float(EXPERTS * np.sum(mean * mean, dtype=np.float64))
    covariance = centered.astype(np.float64) @ centered.astype(np.float64).T
    eigenvalues = np.linalg.eigvalsh(covariance)[::-1]
    fractions = {"shared": shared / total if total else 1.0}
    for rank in (1, 2, 4, 8, 16):
        captured = shared + float(np.sum(eigenvalues[:rank]))
        fractions[f"shared_plus_rank_{rank}"] = captured / total if total else 1.0
    fractions["centered_variance_fraction"] = (
        float(np.sum(eigenvalues)) / total if total else 0.0
    )
    fractions["residual_fraction_rank_4"] = 1.0 - fractions["shared_plus_rank_4"]
    return fractions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    safetensors = args.model / "model.safetensors"
    header, data_start = load_header(safetensors)
    raw = memoryview(safetensors.read_bytes())[data_start:]
    records = []
    for layer in range(LAYERS):
        for kind, suffix in (
            ("gate_up", "input_linear.weight"),
            ("down", "output_linear.weight"),
        ):
            name = f"model.layers.{layer}.block_sparse_moe.{suffix}"
            info = header[name]
            if info["dtype"] != "BF16" or info["shape"][0] != EXPERTS:
                raise SystemExit(f"unsupported {name}: {info['dtype']} {info['shape']}")
            start, end = info["data_offsets"]
            weights = bf16_array(raw, start, info["shape"])
            fractions = basis_fractions(weights)
            records.append(
                {
                    "layer": layer,
                    "tensor": kind,
                    "shape": info["shape"],
                    **fractions,
                }
            )
            print(
                f"layer={layer:02d} tensor={kind:7s} "
                f"shared={fractions['shared']:.4f} "
                f"rank4={fractions['shared_plus_rank_4']:.4f} "
                f"residual_rank4={fractions['residual_fraction_rank_4']:.4f}"
            )

    aggregate = {}
    for tensor in ("gate_up", "down"):
        selected = [row for row in records if row["tensor"] == tensor]
        aggregate[tensor] = {
            key: float(np.mean([row[key] for row in selected]))
            for key in (
                "shared",
                "shared_plus_rank_1",
                "shared_plus_rank_2",
                "shared_plus_rank_4",
                "shared_plus_rank_8",
                "shared_plus_rank_16",
                "residual_fraction_rank_4",
            )
        }
    report = {
        "model": str(args.model),
        "experts": EXPERTS,
        "layers": LAYERS,
        "method": "cross-expert covariance eigenvalues on centered BF16 weights",
        "quality_status": "candidate generation only; no teacher-forced validation",
        "aggregate": aggregate,
        "layers": records,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
