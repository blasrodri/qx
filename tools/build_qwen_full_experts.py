#!/usr/bin/env python3
"""Resumable, bounded capture and training for the Qwen3-30B-A3B expert tier.

One capture shard replays the GGUF once and writes training samples for every
MoE layer.  Completed shard markers and per-layer manifests are the resume
boundary.  By default, one invocation performs at most one missing work unit,
which keeps interactive development from accidentally starting a many-hour
job.
"""

import argparse
import hashlib
import json
import math
import os
import pathlib
import subprocess
import sys
import time


MODEL_LAYERS = 48
MODEL_EXPERTS = 128
MODEL_TOP_K = 8
EXPERT_NEURONS = 768
HIDDEN_SIZE = 2048


def parse_layer_set(value):
    layers = set()
    for item in value.split(","):
        item = item.strip()
        if not item:
            raise argparse.ArgumentTypeError("layer list contains an empty item")
        if "-" in item:
            first_text, last_text = item.split("-", 1)
            first, last = int(first_text), int(last_text)
            if first > last:
                raise argparse.ArgumentTypeError(f"descending layer range: {item}")
            layers.update(range(first, last + 1))
        else:
            layers.add(int(item))
    if not layers or min(layers) < 0 or max(layers) >= MODEL_LAYERS:
        raise argparse.ArgumentTypeError("layers must be within 0..47")
    return sorted(layers)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Advance a resumable full-layer Qwen expert build."
    )
    parser.add_argument("--model", type=pathlib.Path, required=True)
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument("--model-root", type=pathlib.Path, required=True)
    parser.add_argument("--train-corpus", type=pathlib.Path, action="append", required=True)
    parser.add_argument(
        "--heldout-corpus", type=pathlib.Path, action="append", required=True
    )
    parser.add_argument(
        "--heldout-extension-corpus",
        type=pathlib.Path,
        action="append",
        default=[],
        help="Additional held-out corpus captured in append-only shards.",
    )
    parser.add_argument("--work-dir", type=pathlib.Path, required=True)
    parser.add_argument(
        "--route-cache",
        type=pathlib.Path,
        default=pathlib.Path("target/release/qx-route-cache"),
    )
    parser.add_argument(
        "--trainer",
        type=pathlib.Path,
        default=pathlib.Path("tools/train_qwen_expert_layer.py"),
    )
    parser.add_argument("--layers", type=parse_layer_set, default=parse_layer_set("0-47"))
    parser.add_argument(
        "--reuse-manifest",
        type=pathlib.Path,
        action="append",
        default=[],
        help="A compatible completed layer manifest that should not be retrained.",
    )
    parser.add_argument("--phase", choices=("all", "capture", "train", "status"), default="all")
    parser.add_argument("--max-work-units", type=int, default=1)
    parser.add_argument("--train-sequences", type=int, default=32)
    parser.add_argument("--heldout-sequences", type=int, default=8)
    parser.add_argument("--heldout-extension-sequences", type=int, default=0)
    parser.add_argument("--sequences-per-shard", type=int, default=8)
    parser.add_argument("--max-tokens-per-sequence", type=int, default=96)
    parser.add_argument("--context", type=int, default=128)
    parser.add_argument("--train-samples-per-expert", type=int, default=32)
    parser.add_argument("--heldout-samples-per-expert", type=int, default=8)
    parser.add_argument("--heldout-extension-samples-per-expert", type=int, default=8)
    parser.add_argument("--min-train-samples", type=int, default=4)
    parser.add_argument("--min-heldout-samples", type=int, default=2)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--heldout-gpu-layers",
        type=int,
        help="Override GPU offload only for held-out capture; default keeps all layers on GPU.",
    )
    parser.add_argument("--nice", type=int, default=15)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def count_corpus_paragraphs(paths):
    count = 0
    inside = False
    for path in paths:
        for line in path.read_text().splitlines():
            if line.strip():
                inside = True
            elif inside:
                count += 1
                inside = False
        if inside:
            count += 1
            inside = False
    return count


def validate_args(args):
    inputs = [
        args.model,
        args.model_root / "model.safetensors.index.json",
        args.route_cache,
        args.trainer,
        *args.train_corpus,
        *args.heldout_corpus,
        *args.heldout_extension_corpus,
        *args.reuse_manifest,
    ]
    missing = [str(path) for path in inputs if not path.exists()]
    if missing:
        raise ValueError("missing input(s): " + ", ".join(missing))
    if len(args.model_sha256) != 64 or any(
        character not in "0123456789abcdefABCDEF" for character in args.model_sha256
    ):
        raise ValueError("model SHA-256 must contain exactly 64 hexadecimal characters")
    if args.max_work_units < 0:
        raise ValueError("max work units cannot be negative; zero means unlimited")
    if args.train_sequences <= 0 or args.heldout_sequences <= 0:
        raise ValueError("train and held-out sequence counts must be positive")
    if args.heldout_extension_sequences < 0:
        raise ValueError("held-out extension sequence count cannot be negative")
    if bool(args.heldout_extension_corpus) != bool(args.heldout_extension_sequences):
        raise ValueError(
            "held-out extension corpus and positive sequence count must be provided together"
        )
    train_paragraphs = count_corpus_paragraphs(args.train_corpus)
    heldout_paragraphs = count_corpus_paragraphs(args.heldout_corpus)
    heldout_extension_paragraphs = count_corpus_paragraphs(
        args.heldout_extension_corpus
    )
    if args.train_sequences > train_paragraphs:
        raise ValueError(
            f"requested {args.train_sequences} train sequences from only "
            f"{train_paragraphs} corpus paragraphs"
        )
    if args.heldout_sequences > heldout_paragraphs:
        raise ValueError(
            f"requested {args.heldout_sequences} held-out sequences from only "
            f"{heldout_paragraphs} corpus paragraphs"
        )
    if args.heldout_extension_sequences > heldout_extension_paragraphs:
        raise ValueError(
            f"requested {args.heldout_extension_sequences} held-out extension sequences "
            f"from only {heldout_extension_paragraphs} corpus paragraphs"
        )
    if args.sequences_per_shard <= 0:
        raise ValueError("sequences per shard must be positive")
    if not 2 <= args.max_tokens_per_sequence <= args.context:
        raise ValueError("max tokens per sequence must be within 2..context")
    if (
        args.train_samples_per_expert <= 0
        or args.heldout_samples_per_expert <= 0
        or args.heldout_extension_samples_per_expert <= 0
    ):
        raise ValueError("per-expert evidence caps must be positive")
    if args.nice < 0 or args.nice > 20:
        raise ValueError("nice level must be within 0..20")
    if args.heldout_gpu_layers is not None and not 0 <= args.heldout_gpu_layers <= MODEL_LAYERS:
        raise ValueError("held-out GPU layers must be within 0..48")


def stable_fingerprint(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def shard_specs(kind, corpora, sequences, sequences_per_shard, sample_cap, args):
    count = math.ceil(sequences / sequences_per_shard)
    cap_per_shard = math.ceil(sample_cap / count)
    specs = []
    for index in range(count):
        skip = index * sequences_per_shard
        count_for_shard = min(sequences_per_shard, sequences - skip)
        directory = args.work_dir / "captures" / kind / f"{index:03d}"
        contract = {
            "schema_version": 1,
            "kind": kind,
            "model": str(args.model.resolve()),
            "model_sha256": args.model_sha256.lower(),
            "corpora": [str(path.resolve()) for path in corpora],
            "skip_sequences": skip,
            "sequences": count_for_shard,
            "max_tokens_per_sequence": args.max_tokens_per_sequence,
            "context": args.context,
            "max_samples_per_expert": cap_per_shard,
            "layers": MODEL_LAYERS,
            "experts": MODEL_EXPERTS,
            "top_k": MODEL_TOP_K,
            "expert_neurons": EXPERT_NEURONS,
            "hidden_size": HIDDEN_SIZE,
            "sample_format": "qxds-v1",
        }
        if kind.startswith("heldout") and args.heldout_gpu_layers is not None:
            contract["gpu_layers"] = args.heldout_gpu_layers
        contract["fingerprint"] = stable_fingerprint(contract)
        specs.append((directory, contract))
    return specs


def marker_status(directory, expected):
    marker = directory / "complete.json"
    if not marker.is_file():
        return False
    actual = json.loads(marker.read_text())
    if actual.get("fingerprint") != expected["fingerprint"]:
        raise ValueError(f"capture contract changed for completed shard {directory}")
    route_trace = directory / "routes.jsonl"
    layer_dir = directory / "layers"
    layer_files = [layer_dir / f"layer{layer}.qxds" for layer in range(MODEL_LAYERS)]
    if not route_trace.is_file() or any(
        not path.is_file() or path.stat().st_size == 0 for path in layer_files
    ):
        raise ValueError(f"completed capture shard is incomplete on disk: {directory}")
    return True


def display_command(command):
    print("+ " + " ".join(subprocess.list2cmdline([item]) for item in command), flush=True)


def run_command(command, dry_run):
    display_command(command)
    if not dry_run:
        subprocess.run(command, check=True)


def capture_command(directory, contract, corpora, args):
    command = [
        "nice",
        "-n",
        str(args.nice),
        str(args.route_cache),
        "capture-corpus",
        "--model",
        str(args.model),
    ]
    for corpus in corpora:
        command.extend(("--corpus", str(corpus)))
    command.extend(
        (
            "--output",
            str(directory / "routes.jsonl"),
            "--sample-all-layers",
            "--split-layer-binary-dir",
            str(directory / "layers"),
            "--skip-sequences",
            str(contract["skip_sequences"]),
            "--max-sequences",
            str(contract["sequences"]),
            "--max-tokens-per-sequence",
            str(contract["max_tokens_per_sequence"]),
            "--max-samples-per-expert",
            str(contract["max_samples_per_expert"]),
            "--context",
            str(contract["context"]),
            "--layers",
            str(MODEL_LAYERS),
            "--experts",
            str(MODEL_EXPERTS),
            "--top-k",
            str(MODEL_TOP_K),
            f"--gpu-layers={contract.get('gpu_layers', -1)}",
            "--threads",
            str(args.threads),
            "--expert-neurons",
            str(EXPERT_NEURONS),
            "--hidden-size",
            str(HIDDEN_SIZE),
            "--model-sha256",
            args.model_sha256.lower(),
        )
    )
    return command


def load_manifest(path):
    value = json.loads(path.read_text())
    required = {
        "classification": "qwen-layer-batched-width-transformation",
        "student_width": EXPERT_NEURONS,
        "teacher_target": "captured-q4",
    }
    for key, expected in required.items():
        if value.get(key) != expected:
            raise ValueError(f"incompatible reused manifest {path}: {key}")
    codecs = value.get("projection_codecs", {})
    if any(codecs.get(projection) != "q2s4" for projection in ("gate", "up", "down")):
        raise ValueError(f"reused manifest is not uniform Q2S4: {path}")
    layer = value.get("layer")
    if not isinstance(layer, int) or not 0 <= layer < MODEL_LAYERS:
        raise ValueError(f"reused manifest has invalid layer: {path}")
    for student in value.get("students", []):
        sidecar = pathlib.Path(student.get("sidecar", ""))
        if not sidecar.is_file():
            raise ValueError(f"manifest references a missing student sidecar: {sidecar}")
    return layer


def completed_training(args):
    completed = {}
    for manifest in args.reuse_manifest:
        layer = load_manifest(manifest)
        if layer in completed:
            raise ValueError(f"duplicate completed manifest for layer {layer}")
        completed[layer] = manifest
    for layer in range(MODEL_LAYERS):
        manifest = args.work_dir / "manifests" / f"layer{layer}.json"
        if not manifest.is_file():
            continue
        manifest_layer = load_manifest(manifest)
        if manifest_layer != layer:
            raise ValueError(f"manifest path/layer mismatch: {manifest}")
        if layer in completed:
            raise ValueError(f"layer {layer} has both reused and local manifests")
        completed[layer] = manifest
    return completed


def training_command(layer, train_specs, heldout_specs, args):
    command = [
        "nice",
        "-n",
        str(args.nice),
        sys.executable,
        str(args.trainer),
        "--model-root",
        str(args.model_root),
    ]
    for directory, _ in train_specs:
        command.extend(("--train-trace", str(directory / "layers" / f"layer{layer}.qxds")))
    for directory, _ in heldout_specs:
        command.extend(
            ("--heldout-trace", str(directory / "layers" / f"layer{layer}.qxds"))
        )
    command.extend(
        (
            "--layer",
            str(layer),
            "--experts",
            str(MODEL_EXPERTS),
            "--width",
            str(EXPERT_NEURONS),
            "--min-train-samples",
            str(args.min_train_samples),
            "--min-heldout-samples",
            str(args.min_heldout_samples),
            "--gate-codec",
            "q2s4",
            "--up-codec",
            "q2s4",
            "--down-codec",
            "q2s4",
            "--teacher-target",
            "captured-q4",
            "--output-dir",
            str(args.work_dir / "students" / f"layer{layer}"),
            "--manifest",
            str(args.work_dir / "manifests" / f"layer{layer}.json"),
        )
    )
    return command


def main():
    args = parse_args()
    validate_args(args)
    train_specs = shard_specs(
        "train",
        args.train_corpus,
        args.train_sequences,
        args.sequences_per_shard,
        args.train_samples_per_expert,
        args,
    )
    heldout_specs = shard_specs(
        "heldout",
        args.heldout_corpus,
        args.heldout_sequences,
        args.sequences_per_shard,
        args.heldout_samples_per_expert,
        args,
    )
    heldout_extension_specs = []
    if args.heldout_extension_sequences:
        heldout_extension_specs = shard_specs(
            "heldout-extension",
            args.heldout_extension_corpus,
            args.heldout_extension_sequences,
            args.sequences_per_shard,
            args.heldout_extension_samples_per_expert,
            args,
        )
    all_heldout_specs = [*heldout_specs, *heldout_extension_specs]
    all_specs = [
        *((directory, contract, args.train_corpus) for directory, contract in train_specs),
        *((directory, contract, args.heldout_corpus) for directory, contract in heldout_specs),
        *(
            (directory, contract, args.heldout_extension_corpus)
            for directory, contract in heldout_extension_specs
        ),
    ]
    capture_complete = []
    capture_pending = []
    for directory, contract, corpora in all_specs:
        if marker_status(directory, contract):
            capture_complete.append(directory)
        else:
            capture_pending.append((directory, contract, corpora))
    trained = completed_training(args)
    pending_layers = [layer for layer in args.layers if layer not in trained]

    print(
        f"capture shards: {len(capture_complete)}/{len(all_specs)} complete; "
        f"requested layers: {len(trained.keys() & set(args.layers))}/{len(args.layers)} trained",
        flush=True,
    )
    if args.phase == "status":
        if capture_pending:
            print("next capture: " + str(capture_pending[0][0]), flush=True)
        elif pending_layers:
            print(f"next training layer: {pending_layers[0]}", flush=True)
        else:
            print("requested build is complete", flush=True)
        return

    units = 0
    unlimited = args.max_work_units == 0
    if args.phase in ("all", "capture"):
        for directory, contract, corpora in capture_pending:
            if not unlimited and units >= args.max_work_units:
                break
            directory.mkdir(parents=True, exist_ok=True)
            started = time.monotonic()
            run_command(capture_command(directory, contract, corpora, args), args.dry_run)
            if not args.dry_run:
                completed = dict(contract)
                completed["elapsed_seconds"] = time.monotonic() - started
                atomic_json(directory / "complete.json", completed)
            units += 1

    captures_ready = not capture_pending or (
        not args.dry_run and all(marker_status(directory, contract) for directory, contract, _ in all_specs)
    )
    if args.phase in ("all", "train") and not captures_ready:
        print("training waits until every capture shard is complete", flush=True)
    elif args.phase in ("all", "train"):
        for layer in pending_layers:
            if not unlimited and units >= args.max_work_units:
                break
            run_command(
                training_command(layer, train_specs, all_heldout_specs, args), args.dry_run
            )
            units += 1

    remaining_captures = sum(
        not marker_status(directory, contract) for directory, contract, _ in all_specs
    )
    trained = completed_training(args)
    remaining_layers = sum(layer not in trained for layer in args.layers)
    print(
        f"remaining: {remaining_captures} capture shard(s), {remaining_layers} layer(s); "
        f"work units this invocation: {units}",
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
