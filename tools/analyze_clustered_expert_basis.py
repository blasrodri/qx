#!/usr/bin/env python3
"""Screen clustered shared-base expert approximations offline.

This is a deterministic BF16-weight pre-screen only. It does not produce a
runtime artifact or establish teacher-forced quality.
"""

import argparse
import json
import struct
from pathlib import Path

import numpy as np

LAYERS = 24
EXPERTS = 32


def pairwise_dot(left, right):
    """Compute small expert-by-center dot products without a huge BLAS matmul."""
    return np.einsum("ij,kj->ik", left, right, optimize=True)


def load_header(path):
    with path.open("rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


def load_bf16(raw, offset, shape):
    count = int(np.prod(shape))
    u16 = np.frombuffer(raw, dtype="<u2", count=count, offset=offset)
    return ((u16.astype(np.uint32) << 16).view(np.float32)).reshape(shape)


def greedy_centers(matrix, count):
    matrix64 = matrix.astype(np.float64, copy=False)
    norms = np.sum(matrix64 * matrix64, axis=1, dtype=np.float64)
    centers = [0]
    for _ in range(1, count):
        chosen = matrix64[centers]
        distances = norms.copy()
        for center, center_norm in zip(chosen, norms[centers]):
            distances = np.minimum(
                distances,
                norms + center_norm - 2.0 * pairwise_dot(matrix64, center[None, :])[:, 0],
            )
        distances[centers] = -1.0
        centers.append(int(np.argmax(distances)))
    return centers


def cluster(matrix, count):
    matrix64 = matrix.astype(np.float64, copy=False)
    centers = greedy_centers(matrix, count)
    labels = np.zeros(EXPERTS, dtype=np.int32)
    for _ in range(8):
        centroid_matrix = matrix64[centers]
        distances = (
            np.sum(matrix64 * matrix64, axis=1)[:, None]
            + np.sum(centroid_matrix * centroid_matrix, axis=1)[None, :]
            - 2.0 * pairwise_dot(matrix64, centroid_matrix)
        )
        new_labels = np.argmin(distances, axis=1)
        if np.array_equal(labels, new_labels):
            break
        labels = new_labels
        centers = [
            int(np.flatnonzero(labels == c)[0])
            if np.any(labels == c)
            else centers[c]
            for c in range(count)
        ]
        centroid_matrix = np.vstack(
            [matrix64[labels == c].mean(axis=0) for c in range(count)]
        )
        distances = (
            np.sum(matrix64 * matrix64, axis=1)[:, None]
            + np.sum(centroid_matrix * centroid_matrix, axis=1)[None, :]
            - 2.0 * pairwise_dot(matrix64, centroid_matrix)
        )
        centers = [int(np.argmin(distances[:, c])) for c in range(count)]
    return labels


def explained(matrix, labels, clusters, rank):
    total = float(np.sum(matrix * matrix, dtype=np.float64))
    captured = 0.0
    for c in range(clusters):
        members = matrix[labels == c]
        if len(members) == 0:
            continue
        mean = members.mean(axis=0, dtype=np.float32)
        captured += len(members) * float(np.sum(mean * mean, dtype=np.float64))
        centered = members - mean
        centered64 = centered.astype(np.float64, copy=False)
        covariance = pairwise_dot(centered64, centered64)
        eigenvalues = np.linalg.eigvalsh(covariance)[::-1]
        captured += float(np.sum(eigenvalues[: min(rank, len(eigenvalues))]))
    return captured / total if total else 1.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    path = args.model / "model.safetensors"
    header, data_start = load_header(path)
    raw = memoryview(path.read_bytes())[data_start:]
    records = []
    for layer in range(LAYERS):
        for tensor, suffix in (("gate_up", "input_linear.weight"), ("down", "output_linear.weight")):
            name = f"model.layers.{layer}.block_sparse_moe.{suffix}"
            info = header[name]
            start, _ = info["data_offsets"]
            matrix = load_bf16(raw, start, info["shape"]).reshape(EXPERTS, -1)
            row = {"layer": layer, "tensor": tensor, "shape": info["shape"]}
            for clusters in (1, 2, 4):
                labels = cluster(matrix, clusters)
                for rank in (4, 8, 16):
                    row[f"c{clusters}_r{rank}"] = explained(matrix, labels, clusters, rank)
            records.append(row)
            print(
                f"layer={layer:02d} tensor={tensor:7s} "
                + " ".join(f"c{c}r{r}={row[f'c{c}_r{r}']:.4f}" for c in (1, 2, 4) for r in (4, 8, 16))
            )
    aggregate = {}
    for tensor in ("gate_up", "down"):
        rows = [r for r in records if r["tensor"] == tensor]
        aggregate[tensor] = {
            f"c{c}_r{r}": float(np.mean([x[f"c{c}_r{r}"] for x in rows]))
            for c in (1, 2, 4)
            for r in (4, 8, 16)
        }
    args.output.write_text(json.dumps({
        "model": str(args.model),
        "method": "deterministic greedy expert clustering plus within-cluster covariance basis",
        "quality_status": "candidate generation only; no activation-weighted or teacher-forced validation",
        "aggregate": aggregate,
        "records": records,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
