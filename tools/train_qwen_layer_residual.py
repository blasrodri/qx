#!/usr/bin/env python3
"""Fit a low-rank correction to an aggregated transformed Qwen MoE layer."""

import argparse
import json
import pathlib
import struct
import time

import torch
from safetensors import safe_open

import train_qwen_expert_student as expert_student


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", type=pathlib.Path, required=True)
    parser.add_argument("--train-trace", type=pathlib.Path, required=True)
    parser.add_argument("--heldout-trace", type=pathlib.Path, required=True)
    parser.add_argument("--student-manifest", type=pathlib.Path, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--ranks", type=int, action="append", default=[])
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--report", type=pathlib.Path, required=True)
    return parser.parse_args()


def read_complete_trace(path, layer, top_k):
    records = []
    with path.open() as stream:
        for line in stream:
            record = json.loads(line)
            samples = record.get("expert_distill_samples", [])
            samples = [sample for sample in samples if sample["layer"] == layer]
            if len(samples) != top_k:
                continue
            routes = record["routes"][layer]
            if [sample["expert"] for sample in samples] != routes:
                raise ValueError("expert sample order does not match routed slots")
            first_input = samples[0]["input"]
            if any(sample["input"] != first_input for sample in samples[1:]):
                raise ValueError("routed experts do not share one normalized input")
            records.append(
                {
                    "input": first_input,
                    "routes": routes,
                    "teacher": [sample["teacher_down_output"] for sample in samples],
                }
            )
    if not records:
        raise ValueError(f"{path} contains no complete top-{top_k} layer records")
    return records


def load_router(root, layer):
    weight_map = json.loads((root / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    name = f"model.layers.{layer}.mlp.gate.weight"
    with safe_open(root / weight_map[name], framework="pt", device="cpu") as stream:
        return stream.get_tensor(name).float().contiguous()


def read_student(path):
    with path.open("rb") as stream:
        if stream.read(8) != b"QXEJS001":
            raise ValueError(f"invalid student sidecar {path}")
        layer, expert, hidden, width, teacher_width = struct.unpack("<IIIII", stream.read(20))
        count = hidden * width
        tensors = []
        for _ in range(3):
            values = torch.frombuffer(bytearray(stream.read(count * 4)), dtype=torch.float32)
            tensors.append(values.reshape(width, hidden).clone())
        if stream.read(1):
            raise ValueError(f"student sidecar has trailing bytes: {path}")
    gate, up, down_flat = tensors
    return {
        "layer": layer,
        "expert": expert,
        "hidden": hidden,
        "width": width,
        "teacher_width": teacher_width,
        "gate": gate,
        "up": up,
        "down": down_flat.reshape(hidden, width),
    }


def layer_dataset(records, router, sidecars, device):
    inputs = torch.tensor([record["input"] for record in records], dtype=torch.float32)
    routes = torch.tensor([record["routes"] for record in records], dtype=torch.int64)
    teacher = torch.tensor([record["teacher"] for record in records], dtype=torch.float32)
    with torch.no_grad():
        selected_logits = (inputs @ router.T).gather(1, routes)
        weights = torch.softmax(selected_logits, dim=1)
        exact = torch.einsum("nk,nkh->nh", weights, teacher)
    student_outputs = teacher.clone()
    for expert, path in sidecars.items():
        occurrences = (routes == expert).nonzero(as_tuple=False)
        if occurrences.numel() == 0:
            continue
        student = read_student(path)
        if student["layer"] != sidecars.layer or student["expert"] != expert:
            raise ValueError(f"student metadata mismatch: {path}")
        rows = occurrences[:, 0]
        batch = inputs[rows].to(device)
        with torch.no_grad():
            output, _ = expert_student.expert_forward(
                batch,
                student["gate"].to(device),
                student["up"].to(device),
                student["down"].to(device),
            )
        student_outputs[rows, occurrences[:, 1]] = output.cpu()
    student_aggregate = torch.einsum("nk,nkh->nh", weights, student_outputs)
    return inputs, exact, student_aggregate


class Sidecars(dict):
    layer: int


class ResidualAdapter(torch.nn.Module):
    def __init__(self, hidden, rank):
        super().__init__()
        self.down = torch.nn.Linear(hidden, rank, bias=False)
        self.up = torch.nn.Linear(rank, hidden, bias=False)
        torch.nn.init.normal_(self.down.weight, std=1.0 / hidden**0.5)
        torch.nn.init.zeros_(self.up.weight)

    def forward(self, values):
        return self.up(self.down(values))


def aggregate_metrics(predicted, expected):
    result = expert_student.metrics(predicted, expected)
    result["normalized_mse"] = float(
        expert_student.normalized_loss(predicted, expected)
    )
    return result


def fit_rank(args, rank, train_x, train_residual, heldout_x, heldout_exact,
             heldout_student, device):
    torch.manual_seed(args.seed + rank)
    model = ResidualAdapter(train_x.shape[1], rank).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    permutation = torch.randperm(train_x.shape[0], generator=torch.Generator().manual_seed(args.seed))
    validation_count = max(8, train_x.shape[0] // 5)
    validation = permutation[:validation_count]
    fitting = permutation[validation_count:]
    x = train_x.to(device)
    residual = train_residual.to(device)
    best = None
    stale = 0
    started = time.monotonic()
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        prediction = model(x[fitting])
        loss = expert_student.normalized_loss(prediction, residual[fitting])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_loss = float(
                expert_student.normalized_loss(model(x[validation]), residual[validation]).cpu()
            )
        if best is None or validation_loss < best[0] - 1e-7:
            best = (
                validation_loss,
                {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
            )
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break
    model.load_state_dict(best[1])
    model.cpu().eval()
    with torch.no_grad():
        corrected = heldout_student + model(heldout_x)
    return model, {
        "rank": rank,
        "parameters": sum(value.numel() for value in model.parameters()),
        "fp16_bytes_per_layer": sum(value.numel() for value in model.parameters()) * 2,
        "epochs_completed": epoch + 1,
        "best_validation_normalized_residual_mse": best[0],
        "training_seconds": time.monotonic() - started,
        "heldout_corrected": aggregate_metrics(corrected, heldout_exact),
    }


def write_adapter(path, layer, model):
    down = model.down.weight.detach().contiguous()
    up = model.up.weight.detach().contiguous()
    with path.open("wb") as stream:
        stream.write(b"QXLRA001")
        stream.write(struct.pack("<III", layer, down.shape[0], down.shape[1]))
        stream.write(down.numpy().astype("<f4", copy=False).tobytes())
        stream.write(up.numpy().astype("<f4", copy=False).tobytes())


def main():
    args = parse_args()
    ranks = sorted(set(args.ranks or [8, 16, 32, 64]))
    if args.layer < 0 or args.top_k <= 0 or any(rank <= 0 for rank in ranks):
        raise ValueError("layer, top-k, and ranks must be positive")
    device = args.device
    if device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    manifest = json.loads(args.student_manifest.read_text())
    sidecars = Sidecars(
        (student["expert"], pathlib.Path(student["sidecar"]))
        for student in manifest["students"]
    )
    sidecars.layer = args.layer
    router = load_router(args.model_root, args.layer)
    train_records = read_complete_trace(args.train_trace, args.layer, args.top_k)
    heldout_records = read_complete_trace(args.heldout_trace, args.layer, args.top_k)
    train_x, train_exact, train_student = layer_dataset(
        train_records, router, sidecars, device
    )
    heldout_x, heldout_exact, heldout_student = layer_dataset(
        heldout_records, router, sidecars, device
    )
    train_residual = train_exact - train_student
    results = []
    models = []
    for rank in ranks:
        model, result = fit_rank(
            args,
            rank,
            train_x,
            train_residual,
            heldout_x,
            heldout_exact,
            heldout_student,
            device,
        )
        models.append(model)
        results.append(result)
        print(
            f"rank {rank}: heldout corrected L2 "
            f"{result['heldout_corrected']['mean_relative_l2']:.4f}",
            flush=True,
        )
    best_index = min(
        range(len(results)),
        key=lambda index: results[index]["heldout_corrected"]["mean_relative_l2"],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    write_adapter(args.output, args.layer, models[best_index])
    report = {
        "schema_version": 1,
        "classification": "qwen-transformed-layer-low-rank-residual-smoke",
        "promotion_evidence": False,
        "layer": args.layer,
        "router_semantics": "BF16-router-softmax-over-exact-Q4-selected-experts",
        "train_complete_positions": len(train_records),
        "heldout_complete_positions": len(heldout_records),
        "transformed_experts": len(sidecars),
        "heldout_uncorrected": aggregate_metrics(heldout_student, heldout_exact),
        "rank_results": results,
        "selected_rank": results[best_index]["rank"],
        "adapter": str(args.output),
        "limitations": [
            "offline aggregate-output reconstruction",
            "BF16 router weights approximate Q4 route weights",
            "adapter rank selected on the reported heldout slice",
            "requires live full-model injection before promotion",
        ],
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
