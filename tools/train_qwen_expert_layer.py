#!/usr/bin/env python3
"""Transform every evidenced expert in one Qwen MoE layer in a single run.

The single-expert trainer is the numerical reference. This driver removes the
scaling pathologies around it: traces and the safetensors index are read once,
all experts share one process and MPS context, and under-evidenced experts are
reported as explicit full-width exceptions instead of being silently trained.
"""

import argparse
import collections
import gc
import hashlib
import json
import pathlib
import time

import torch

import train_qwen_expert_student as expert_student


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", type=pathlib.Path, required=True)
    parser.add_argument("--train-trace", type=pathlib.Path, action="append", required=True)
    parser.add_argument("--heldout-trace", type=pathlib.Path, action="append", required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--experts", type=int, default=128)
    parser.add_argument(
        "--expert",
        type=int,
        action="append",
        help="Restrict the run to repeated expert IDs; the default is the complete layer.",
    )
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--min-train-samples", type=int, default=4)
    parser.add_argument("--min-heldout-samples", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--qat-epochs", type=int, default=400)
    parser.add_argument("--qat-learning-rate", type=float, default=2e-5)
    parser.add_argument("--qat-patience", type=int, default=80)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    parser.add_argument("--gate-codec", choices=("q2s4", "q3"), default="q2s4")
    parser.add_argument("--up-codec", choices=("q2s4", "q3"), default="q3")
    parser.add_argument("--down-codec", choices=("q2s4", "q3"), default="q2s4")
    parser.add_argument(
        "--teacher-target",
        choices=("bf16", "captured-q4"),
        default="captured-q4",
        help="Output target for distillation; captured-q4 matches the paired GGUF control.",
    )
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--manifest", type=pathlib.Path, required=True)
    parser.add_argument(
        "--retain-master",
        action="store_true",
        help="Also retain the pre-dequantization F32 student sidecar.",
    )
    return parser.parse_args()


def load_grouped_trace(paths, layer, experts):
    grouped_inputs = collections.defaultdict(list)
    grouped_outputs = collections.defaultdict(list)
    positions = 0
    samples = 0
    for path in paths:
        with path.open() as stream:
            for line_number, line in enumerate(stream, 1):
                record = json.loads(line)
                record_samples = record.get("expert_distill_samples", [])
                single = record.get("expert_distill_sample")
                if single is not None:
                    record_samples = [single, *record_samples]
                if record_samples:
                    positions += 1
                for sample in record_samples:
                    if sample["layer"] != layer:
                        continue
                    expert = sample["expert"]
                    if expert < 0 or expert >= experts:
                        raise ValueError(
                            f"{path}:{line_number} has out-of-range expert {expert}"
                        )
                    values = sample.get("input", [])
                    if not values:
                        raise ValueError(
                            f"{path}:{line_number} predates normalized expert-input capture"
                        )
                    grouped_inputs[expert].append(values)
                    grouped_outputs[expert].append(sample["teacher_down_output"])
                    samples += 1
    return grouped_inputs, grouped_outputs, {"positions": positions, "samples": samples}


def as_tensors(grouped_inputs, grouped_outputs, expert):
    return (
        torch.tensor(grouped_inputs[expert], dtype=torch.float32),
        torch.tensor(grouped_outputs[expert], dtype=torch.float32),
    )


def weighted_metric(results, key):
    samples = sum(result["heldout_samples"] for result in results)
    if samples == 0:
        return None
    return sum(
        result["quantized_trained_heldout"][key] * result["heldout_samples"]
        for result in results
    ) / samples


def train_one(args, weight_map, expert, train_inputs, captured_train_outputs,
              heldout_inputs, captured_heldout_outputs):
    teacher = expert_student.load_teacher(
        args.model_root, args.layer, expert, weight_map=weight_map
    )
    hidden = teacher["gate"].shape[1]
    if train_inputs.shape[1] != hidden or heldout_inputs.shape[1] != hidden:
        raise ValueError(f"expert {expert} captured input width does not match teacher")
    with torch.no_grad():
        bf16_train_targets, _ = expert_student.expert_forward(
            train_inputs, teacher["gate"], teacher["up"], teacher["down"]
        )
        bf16_heldout_targets, _ = expert_student.expert_forward(
            heldout_inputs, teacher["gate"], teacher["up"], teacher["down"]
        )
    train_targets = (
        captured_train_outputs
        if args.teacher_target == "captured-q4"
        else bf16_train_targets
    )
    heldout_targets = (
        captured_heldout_outputs
        if args.teacher_target == "captured-q4"
        else bf16_heldout_targets
    )
    local_args = argparse.Namespace(**vars(args))
    local_args.expert = expert
    student, initial, indices, training = expert_student.train(
        local_args,
        train_inputs,
        train_targets,
        heldout_inputs,
        heldout_targets,
        teacher,
    )
    codecs = (args.gate_codec, args.up_codec, args.down_codec)
    with torch.no_grad():
        initial_predictions, _ = expert_student.expert_forward(
            heldout_inputs, initial["gate"], initial["up"], initial["down"]
        )
        pre_qat_predictions = student(heldout_inputs, codecs=codecs)
    qat_training = expert_student.quantization_aware_train(
        local_args, student, train_inputs, train_targets
    )
    quantized_student = expert_student.Student(
        expert_student.quantize_dequantized(student.gate, args.gate_codec),
        expert_student.quantize_dequantized(student.up, args.up_codec),
        expert_student.quantize_dequantized(student.down, args.down_codec),
    )
    with torch.no_grad():
        trained_predictions = student(heldout_inputs)
        quantized_predictions = quantized_student(heldout_inputs)

    stem = f"layer{args.layer}-expert{expert}"
    sidecar = args.output_dir / f"{stem}.qxejs"
    master = args.output_dir / f"{stem}-master.qxejs" if args.retain_master else None
    report_path = args.output_dir / f"{stem}-report.json"
    expert_student.write_sidecar(sidecar, local_args, quantized_student)
    if master is not None:
        expert_student.write_sidecar(master, local_args, student)
    report = {
        "schema_version": 1,
        "classification": "layer-batched-expert-local-width-student",
        "promotion_evidence": False,
        "layer": args.layer,
        "expert": expert,
        "teacher_width": 768,
        "student_width": args.width,
        "projection_codecs": {
            "gate": args.gate_codec,
            "up": args.up_codec,
            "down": args.down_codec,
        },
        "teacher_target": args.teacher_target,
        "train_samples": int(train_inputs.shape[0]),
        "heldout_samples": int(heldout_inputs.shape[0]),
        "training": training,
        "qat_training": qat_training,
        "initial_heldout": expert_student.metrics(initial_predictions, heldout_targets),
        "pre_qat_quantized_heldout": expert_student.metrics(
            pre_qat_predictions, heldout_targets
        ),
        "trained_heldout": expert_student.metrics(trained_predictions, heldout_targets),
        "quantized_trained_heldout": expert_student.metrics(
            quantized_predictions, heldout_targets
        ),
        "bf16_vs_captured_q4_heldout": expert_student.metrics(
            bf16_heldout_targets, captured_heldout_outputs
        ),
        "selected_neurons_sha256": hashlib.sha256(
            indices.numpy().astype("<u2", copy=False).tobytes()
        ).hexdigest(),
        "student_parameters": sum(value.numel() for value in student.parameters()),
        "sidecar": str(sidecar),
        "master_sidecar": str(master) if master is not None else None,
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    report["report"] = str(report_path)
    del teacher, train_targets, heldout_targets, student, quantized_student
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    return report


def validate(args):
    if args.layer < 0 or args.experts <= 0:
        raise ValueError("layer must be non-negative and experts positive")
    if args.width <= 0 or args.width > 768 or args.width % 8:
        raise ValueError("width must be an eight-neuron multiple in 8..=768")
    if args.min_train_samples < 2 or args.min_heldout_samples < 1:
        raise ValueError("minimum evidence must be at least two train and one heldout sample")
    if args.epochs <= 0 or args.patience <= 0:
        raise ValueError("epochs and patience must be positive")
    if args.qat_epochs < 0 or args.qat_patience <= 0:
        raise ValueError("QAT epochs must be non-negative and QAT patience positive")
    requested = args.expert if args.expert is not None else range(args.experts)
    requested = sorted(set(requested))
    if not requested or requested[0] < 0 or requested[-1] >= args.experts:
        raise ValueError("requested experts must be within the configured layer")
    for path in [args.model_root, *args.train_trace, *args.heldout_trace]:
        if not path.exists():
            raise ValueError(f"input does not exist: {path}")
    return requested


def main():
    args = parse_args()
    requested = validate(args)
    started = time.monotonic()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    train_inputs, train_outputs, train_capture = load_grouped_trace(
        args.train_trace, args.layer, args.experts
    )
    heldout_inputs, heldout_outputs, heldout_capture = load_grouped_trace(
        args.heldout_trace, args.layer, args.experts
    )
    weight_map = json.loads(
        (args.model_root / "model.safetensors.index.json").read_text()
    )["weight_map"]

    students = []
    exceptions = []
    for index, expert in enumerate(requested, 1):
        train_count = len(train_inputs[expert])
        heldout_count = len(heldout_inputs[expert])
        reasons = []
        if train_count < args.min_train_samples:
            reasons.append("insufficient_train_samples")
        if heldout_count < args.min_heldout_samples:
            reasons.append("insufficient_heldout_samples")
        if reasons:
            exceptions.append(
                {
                    "expert": expert,
                    "policy": "retain_teacher_width",
                    "train_samples": train_count,
                    "heldout_samples": heldout_count,
                    "reasons": reasons,
                }
            )
            print(
                f"[{index}/{len(requested)}] expert {expert}: retain width 768 "
                f"(train={train_count}, heldout={heldout_count})",
                flush=True,
            )
            continue
        train_x, captured_train_y = as_tensors(train_inputs, train_outputs, expert)
        heldout_x, captured_heldout_y = as_tensors(
            heldout_inputs, heldout_outputs, expert
        )
        result = train_one(
            args,
            weight_map,
            expert,
            train_x,
            captured_train_y,
            heldout_x,
            captured_heldout_y,
        )
        students.append(result)
        print(
            f"[{index}/{len(requested)}] expert {expert}: width {args.width}, "
            f"train={train_count}, heldout={heldout_count}, "
            f"quantized L2={result['quantized_trained_heldout']['mean_relative_l2']:.4f}",
            flush=True,
        )

    train_total = sum(len(train_inputs[expert]) for expert in requested)
    heldout_total = sum(len(heldout_inputs[expert]) for expert in requested)
    student_ids = {result["expert"] for result in students}
    train_student_hits = sum(len(train_inputs[expert]) for expert in student_ids)
    heldout_student_hits = sum(len(heldout_inputs[expert]) for expert in student_ids)
    manifest = {
        "schema_version": 1,
        "classification": "qwen-layer-batched-width-transformation",
        "promotion_evidence": False,
        "layer": args.layer,
        "teacher_width": 768,
        "student_width": args.width,
        "projection_codecs": {
            "gate": args.gate_codec,
            "up": args.up_codec,
            "down": args.down_codec,
        },
        "teacher_target": args.teacher_target,
        "evidence_policy": {
            "min_train_samples": args.min_train_samples,
            "min_heldout_samples": args.min_heldout_samples,
            "under_evidenced_policy": "retain_teacher_width",
        },
        "requested_experts": len(requested),
        "transformed_experts": len(students),
        "teacher_width_exceptions": len(exceptions),
        "train_capture": train_capture,
        "heldout_capture": heldout_capture,
        "train_route_coverage": train_student_hits / max(train_total, 1),
        "heldout_route_coverage": heldout_student_hits / max(heldout_total, 1),
        "weighted_quantized_heldout": {
            "mean_relative_l2": weighted_metric(students, "mean_relative_l2"),
            "mean_cosine": weighted_metric(students, "mean_cosine"),
        },
        "maximum_expert_mean_relative_l2": max(
            (
                result["quantized_trained_heldout"]["mean_relative_l2"]
                for result in students
            ),
            default=None,
        ),
        "elapsed_seconds": time.monotonic() - started,
        "students": students,
        "exceptions": exceptions,
        "limitations": [
            "expert-local metrics are diagnostics rather than the promotion oracle",
            "teacher-width exceptions require later evidence or remain heterogeneous",
            "sidecars contain diagnostic dequantized F32 weights rather than packed payloads",
        ],
    }
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        f"layer {args.layer}: transformed {len(students)}/{len(requested)} experts, "
        f"heldout route coverage {manifest['heldout_route_coverage']:.2%}, "
        f"elapsed {manifest['elapsed_seconds']:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
