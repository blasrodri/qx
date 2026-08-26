#!/usr/bin/env python3
"""Train one narrower Qwen MoE expert from routed teacher inputs.

The trainer is deliberately expert-local: it maps only three BF16 tensors,
loads bounded JSONL capture traces, and optimizes one gate/up/down student on
MPS or CPU. This lets the first Mac validate model co-design without loading a
complete BF16 checkpoint or optimizer state.
"""

import argparse
import json
import math
import pathlib
import struct
import time

import torch
from safetensors import safe_open


MAGIC = b"QXEJS001"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", type=pathlib.Path, required=True)
    parser.add_argument("--train-trace", type=pathlib.Path, action="append", required=True)
    parser.add_argument("--heldout-trace", type=pathlib.Path, action="append", required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--expert", type=int, required=True)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--qat-epochs", type=int, default=0)
    parser.add_argument("--qat-learning-rate", type=float, default=5e-6)
    parser.add_argument("--qat-patience", type=int, default=30)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--initial-output", type=pathlib.Path)
    parser.add_argument("--quantized-output", type=pathlib.Path)
    parser.add_argument("--gate-codec", choices=("q2s4", "q3"), default="q2s4")
    parser.add_argument("--up-codec", choices=("q2s4", "q3"), default="q2s4")
    parser.add_argument("--down-codec", choices=("q2s4", "q3"), default="q2s4")
    parser.add_argument(
        "--teacher-target",
        choices=("bf16", "captured-q4"),
        default="bf16",
        help="Output target for distillation; captured-q4 matches the paired GGUF control.",
    )
    parser.add_argument("--report", type=pathlib.Path, required=True)
    return parser.parse_args()


def load_trace(paths, layer, expert):
    inputs = []
    captured_outputs = []
    for path in paths:
        with path.open() as stream:
            for line in stream:
                record = json.loads(line)
                samples = record.get("expert_distill_samples", [])
                single = record.get("expert_distill_sample")
                if single is not None:
                    samples = [single, *samples]
                for sample in samples:
                    if sample["layer"] != layer or sample["expert"] != expert:
                        continue
                    values = sample.get("input", [])
                    if not values:
                        raise ValueError(
                            f"{path} predates normalized expert-input capture"
                        )
                    inputs.append(values)
                    captured_outputs.append(sample["teacher_down_output"])
    if not inputs:
        raise ValueError("trace set contains no target-expert hits")
    return torch.tensor(inputs, dtype=torch.float32), torch.tensor(
        captured_outputs, dtype=torch.float32
    )


def load_teacher(root, layer, expert, weight_map=None):
    if weight_map is None:
        weight_map = json.loads(
            (root / "model.safetensors.index.json").read_text()
        )["weight_map"]
    names_by_shard = {}
    for projection in ("gate", "up", "down"):
        name = f"model.layers.{layer}.mlp.experts.{expert}.{projection}_proj.weight"
        names_by_shard.setdefault(weight_map[name], []).append((projection, name))
    tensors = {}
    for shard_name, names in names_by_shard.items():
        with safe_open(root / shard_name, framework="pt", device="cpu") as stream:
            for projection, name in names:
                tensors[projection] = stream.get_tensor(name).float().contiguous()
    return tensors


def expert_forward(inputs, gate, up, down):
    activation = torch.nn.functional.silu(inputs @ gate.T) * (inputs @ up.T)
    return activation @ down.T, activation


def relative_l2(predicted, expected):
    numerator = (predicted - expected).square().sum(dim=1).sqrt()
    denominator = expected.square().sum(dim=1).sqrt().clamp_min(torch.finfo(torch.float32).tiny)
    return numerator / denominator


def metrics(predicted, expected):
    errors = relative_l2(predicted, expected).detach().cpu().sort().values
    cosine = torch.nn.functional.cosine_similarity(predicted, expected, dim=1).mean()
    return {
        "mean_relative_l2": float(errors.mean()),
        "p50_relative_l2": float(torch.quantile(errors, 0.50)),
        "p95_relative_l2": float(torch.quantile(errors, 0.95)),
        "mean_cosine": float(cosine.detach().cpu()),
    }


class Student(torch.nn.Module):
    def __init__(self, gate, up, down):
        super().__init__()
        self.gate = torch.nn.Parameter(gate.clone())
        self.up = torch.nn.Parameter(up.clone())
        self.down = torch.nn.Parameter(down.clone())

    def forward(self, inputs, codecs=None):
        if codecs is not None:
            return expert_forward(
                inputs,
                quantize_ste(self.gate, codecs[0]),
                quantize_ste(self.up, codecs[1]),
                quantize_ste(self.down, codecs[2]),
            )[0]
        return expert_forward(inputs, self.gate, self.up, self.down)[0]


def choose_neurons(inputs, teacher, width):
    with torch.no_grad():
        _, activation = expert_forward(
            inputs, teacher["gate"], teacher["up"], teacher["down"]
        )
        activation_energy = activation.square().mean(dim=0)
        down_energy = teacher["down"].square().sum(dim=0)
        score = activation_energy * down_energy
        return torch.topk(score, width, sorted=True).indices.sort().values


def normalized_loss(predicted, expected):
    return (predicted - expected).square().mean() / expected.square().mean().clamp_min(1e-12)


def q2s4_dequantized(values):
    if values.shape[1] % 32:
        raise ValueError("Q2S4 tensor rows must be divisible into group-32 blocks")
    shape = values.shape
    groups = values.detach().float().reshape(-1, 32)
    ratio = 0.32
    scale = groups.abs().amax(dim=1, keepdim=True)
    zero = scale == 0
    scale = torch.where(zero, torch.ones_like(scale), scale)
    for _ in range(8):
        magnitude = torch.where(
            groups.abs() < scale * ((1.0 + ratio) * 0.5), ratio, 1.0
        )
        level = torch.where(groups < 0, -magnitude, magnitude)
        scale = (groups * level).sum(dim=1, keepdim=True) / level.square().sum(
            dim=1, keepdim=True
        )
    scale = scale.to(torch.float16).to(torch.float32)
    magnitude = torch.where(
        groups.abs() < scale * ((1.0 + ratio) * 0.5), ratio, 1.0
    )
    level = torch.where(groups < 0, -magnitude, magnitude)
    return (level * scale).reshape(shape).contiguous()


def q3_dequantized(values):
    if values.shape[1] % 32:
        raise ValueError("Q3 tensor rows must be divisible into group-32 blocks")
    shape = values.shape
    groups = values.detach().float().reshape(-1, 32)
    scale = groups.abs().amax(dim=1, keepdim=True) / 3.0
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    scale = scale.to(torch.float16).to(torch.float32)
    return ((groups / scale).round().clamp(-3, 3) * scale).reshape(shape).contiguous()


def quantize_dequantized(values, codec):
    if codec == "q2s4":
        return q2s4_dequantized(values)
    if codec == "q3":
        return q3_dequantized(values)
    raise ValueError(f"unsupported codec {codec}")


def quantize_ste(values, codec):
    quantized = quantize_dequantized(values, codec)
    return values + (quantized - values).detach()


def train(args, train_inputs, train_targets, heldout_inputs, heldout_targets, teacher):
    torch.manual_seed(args.seed)
    indices = choose_neurons(train_inputs, teacher, args.width)
    student = Student(
        teacher["gate"][indices],
        teacher["up"][indices],
        teacher["down"][:, indices],
    )
    initial = {name: value.detach().clone() for name, value in student.state_dict().items()}

    validation_count = max(1, min(train_inputs.shape[0] // 5, 64))
    generator = torch.Generator().manual_seed(args.seed)
    order = torch.randperm(train_inputs.shape[0], generator=generator)
    validation_indices = order[:validation_count]
    fitting_indices = order[validation_count:]
    if fitting_indices.numel() == 0:
        raise ValueError("training trace needs at least two expert hits")

    device = args.device
    if device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    student.to(device)
    fit_x = train_inputs[fitting_indices].to(device)
    fit_y = train_targets[fitting_indices].to(device)
    val_x = train_inputs[validation_indices].to(device)
    val_y = train_targets[validation_indices].to(device)
    optimizer = torch.optim.AdamW(
        student.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    best = None
    best_epoch = 0
    stale = 0
    started = time.monotonic()
    for epoch in range(args.epochs):
        permutation = torch.randperm(fit_x.shape[0], device=device)
        student.train()
        for start in range(0, fit_x.shape[0], args.batch_size):
            batch = permutation[start : start + args.batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = normalized_loss(student(fit_x[batch]), fit_y[batch])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
        student.eval()
        with torch.no_grad():
            validation_loss = float(normalized_loss(student(val_x), val_y).cpu())
        if best is None or validation_loss < best[0] - 1e-7:
            best = (
                validation_loss,
                {name: value.detach().cpu().clone() for name, value in student.state_dict().items()},
            )
            best_epoch = epoch + 1
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break
    student.load_state_dict(best[1])
    student.cpu().eval()
    return student, initial, indices, {
        "device": device,
        "epochs_completed": epoch + 1,
        "best_epoch": best_epoch,
        "best_validation_normalized_mse": best[0],
        "training_seconds": time.monotonic() - started,
        "fitting_samples": int(fitting_indices.numel()),
        "validation_samples": int(validation_indices.numel()),
    }


def quantization_aware_train(args, student, train_inputs, train_targets):
    if args.qat_epochs == 0:
        return None

    validation_count = max(1, min(train_inputs.shape[0] // 5, 64))
    generator = torch.Generator().manual_seed(args.seed)
    order = torch.randperm(train_inputs.shape[0], generator=generator)
    validation_indices = order[:validation_count]
    fitting_indices = order[validation_count:]
    device = args.device
    if device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    student.to(device)
    fit_x = train_inputs[fitting_indices].to(device)
    fit_y = train_targets[fitting_indices].to(device)
    val_x = train_inputs[validation_indices].to(device)
    val_y = train_targets[validation_indices].to(device)
    optimizer = torch.optim.AdamW(
        student.parameters(), lr=args.qat_learning_rate, weight_decay=args.weight_decay
    )
    codecs = (args.gate_codec, args.up_codec, args.down_codec)

    with torch.no_grad():
        initial_validation_loss = float(
            normalized_loss(student(val_x, codecs=codecs), val_y).cpu()
        )
    best = (
        initial_validation_loss,
        {name: value.detach().cpu().clone() for name, value in student.state_dict().items()},
    )
    best_epoch = 0
    stale = 0
    started = time.monotonic()
    for epoch in range(args.qat_epochs):
        permutation = torch.randperm(fit_x.shape[0], device=device)
        student.train()
        for start in range(0, fit_x.shape[0], args.batch_size):
            batch = permutation[start : start + args.batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = normalized_loss(student(fit_x[batch], codecs=codecs), fit_y[batch])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
        student.eval()
        with torch.no_grad():
            validation_loss = float(
                normalized_loss(student(val_x, codecs=codecs), val_y).cpu()
            )
        if validation_loss < best[0] - 1e-7:
            best = (
                validation_loss,
                {
                    name: value.detach().cpu().clone()
                    for name, value in student.state_dict().items()
                },
            )
            best_epoch = epoch + 1
            stale = 0
        else:
            stale += 1
            if stale >= args.qat_patience:
                break
    student.load_state_dict(best[1])
    student.cpu().eval()
    return {
        "device": device,
        "epochs_completed": epoch + 1,
        "best_epoch": best_epoch,
        "initial_validation_normalized_mse": initial_validation_loss,
        "best_validation_normalized_mse": best[0],
        "training_seconds": time.monotonic() - started,
    }


def write_sidecar(path, args, student):
    gate = student.gate.detach().cpu().contiguous()
    up = student.up.detach().cpu().contiguous()
    down = student.down.detach().cpu().contiguous()
    hidden = gate.shape[1]
    with path.open("wb") as stream:
        stream.write(MAGIC)
        stream.write(struct.pack("<IIIII", args.layer, args.expert, hidden, args.width, 768))
        for tensor in (gate, up, down):
            stream.write(tensor.numpy().astype("<f4", copy=False).tobytes())


def main():
    args = parse_args()
    if args.width <= 0 or args.width > 768 or args.width % 8:
        raise ValueError("width must be an eight-neuron multiple in 8..=768")
    if args.epochs <= 0 or args.patience <= 0:
        raise ValueError("epochs and patience must be positive")
    if args.qat_epochs < 0 or args.qat_patience <= 0:
        raise ValueError("QAT epochs must be non-negative and QAT patience positive")
    train_inputs, captured_train_outputs = load_trace(
        args.train_trace, args.layer, args.expert
    )
    heldout_inputs, captured_heldout_outputs = load_trace(
        args.heldout_trace, args.layer, args.expert
    )
    teacher = load_teacher(args.model_root, args.layer, args.expert)
    hidden = teacher["gate"].shape[1]
    if train_inputs.shape[1] != hidden or heldout_inputs.shape[1] != hidden:
        raise ValueError("captured input width does not match BF16 teacher")
    with torch.no_grad():
        bf16_train_targets, _ = expert_forward(
            train_inputs, teacher["gate"], teacher["up"], teacher["down"]
        )
        bf16_heldout_targets, _ = expert_forward(
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
    student, initial, indices, training = train(
        args, train_inputs, train_targets, heldout_inputs, heldout_targets, teacher
    )
    with torch.no_grad():
        initial_predictions, _ = expert_forward(
            heldout_inputs, initial["gate"], initial["up"], initial["down"]
        )
        trained_predictions = student(heldout_inputs)
        codecs = (args.gate_codec, args.up_codec, args.down_codec)
        pre_qat_quantized_predictions = student(heldout_inputs, codecs=codecs)
    qat_training = quantization_aware_train(
        args, student, train_inputs, train_targets
    )
    with torch.no_grad():
        qat_float_predictions = student(heldout_inputs)
        qat_quantized_predictions = student(heldout_inputs, codecs=codecs)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.initial_output is not None:
        args.initial_output.parent.mkdir(parents=True, exist_ok=True)
    if args.quantized_output is not None:
        args.quantized_output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    write_sidecar(args.output, args, student)
    if args.initial_output is not None:
        initial_student = Student(initial["gate"], initial["up"], initial["down"])
        write_sidecar(args.initial_output, args, initial_student)
    quantized_student = None
    if args.quantized_output is not None:
        quantized_student = Student(
            quantize_dequantized(student.gate, args.gate_codec),
            quantize_dequantized(student.up, args.up_codec),
            quantize_dequantized(student.down, args.down_codec),
        )
        write_sidecar(args.quantized_output, args, quantized_student)
    report = {
        "schema_version": 1,
        "classification": "expert-local-joint-gate-up-down-width-student-smoke",
        "promotion_evidence": False,
        "teacher_semantics": "pinned-bf16-expert-on-real-q4-llama-routed-input-states",
        "teacher_target": args.teacher_target,
        "training_semantics": "joint-gate-up-down-output-distillation-with-activation-contribution-initialization-and-optional-projection-codec-ste-qat",
        "layer": args.layer,
        "expert": args.expert,
        "teacher_width": 768,
        "student_width": args.width,
        "projection_codecs": {
            "gate": args.gate_codec,
            "up": args.up_codec,
            "down": args.down_codec,
        },
        "retained_fraction": args.width / 768,
        "train_samples": int(train_inputs.shape[0]),
        "heldout_samples": int(heldout_inputs.shape[0]),
        "training": training,
        "qat_training": qat_training,
        "initial_heldout": metrics(initial_predictions, heldout_targets),
        "pre_qat_trained_heldout": metrics(trained_predictions, heldout_targets),
        "pre_qat_quantized_heldout": metrics(
            pre_qat_quantized_predictions, heldout_targets
        ),
        "trained_heldout": metrics(qat_float_predictions, heldout_targets),
        "quantized_trained_heldout": metrics(quantized_student(heldout_inputs), heldout_targets)
        if quantized_student is not None
        else metrics(qat_quantized_predictions, heldout_targets),
        "bf16_vs_captured_q4_heldout": metrics(
            bf16_heldout_targets, captured_heldout_outputs
        ),
        "selected_neurons_sha256": __import__("hashlib").sha256(
            indices.numpy().astype("<u2", copy=False).tobytes()
        ).hexdigest(),
        "student_parameters": sum(value.numel() for value in student.parameters()),
        "projected_complete_weight_bytes": (
            6_907_387_904
            + 805_306_368
            * sum(codec == "q3" for codec in codecs)
            if args.width == 512
            else None
        ),
        "sidecar": str(args.output),
        "initial_sidecar": str(args.initial_output) if args.initial_output else None,
        "quantized_sidecar": str(args.quantized_output) if args.quantized_output else None,
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
