# QX specialization compiler architecture

The unique-packet result identifies the abstraction boundary QX needs:

```text
logical model
    -> physical tensor representation
    -> quantized consumer/dataflow
    -> sealed executable plan
    -> pre-resolved Metal dispatch graph
```

Result #1 selected the physical representation (`group32_q4_row_inline`).
Result #2 selected the consumer (`unique-packet`) for that representation.
They are independent compiler decisions and must remain independently
searchable.

## Current implementation

Artifact-v2 now seals `moe_consumer` and the independent `moe_down_consumer`
alongside representations, topology, scheduling, memory ownership, and
provenance. Runtime activation resolves those fields once at model load and
sets the Metal pipeline selection before decode; the token loop does not make a
per-token string or plan decision. The split is intentional: gate/up and down
have different shapes and need not select the same packet consumer.

`qx-autotune` now preserves randomized per-fixture samples and emits paired
wall-time deltas, bootstrap confidence intervals, and win rates in its
selection report. The suite contract is configured for 20 rounds; `--rounds`
is available only as an explicit smoke-test override.

## Next IR refactor

The current string consumer fields are a compatibility bridge, not the final
compiler IR. The next stable representation should separate:

- `QuantizedStorage`: row-major, row-inline, Q4_K, or future formats;
- `QuantizedConsumer`: canonical row, cooperative packet, or archived candidates;
- fixed-width descriptors for group size, packet width, scale distribution,
  strides, and artifact slices;
- target capability constraints and workload shape;
- executable pipeline/function ID and dispatch geometry.

The planner should generate compatible storage × consumer candidates and let
complete-model evidence select them. It must not encode a model/target special
case such as `if Granite && M5 { unique_packet }`.

The planning IR now also carries an optional `MoeExecutionUnit` descriptor:
`Matrix`, `RowTile`, `QuantPacket`, or `MoeNeuronCluster`. This is deliberately
only a planning surface for now. Cluster candidates must not be emitted as
executable artifacts until their packed representation, exact consumer, and
full sequential parity gate exist.

## What is deliberately deferred

The following are valuable but should follow statistical sealing of Result #2:

1. Replace persistent JSON/string plan fields with explicit versioned
   little-endian artifact sections and fixed-width IDs.
2. Introduce a typed `StorageBacking`/tensor-slice ownership model for mmap,
   Metal-private, and arena allocations.
3. Flatten the loaded plan into a pre-resolved `DispatchNode[]` so the token
   path contains only buffer bindings, constants, and dispatch calls.
4. Split compiler IR into model, physical, and backend executable levels so
   cooperative packet consumption can map to non-Metal backends later.
5. Add candidate status and tuning-key provenance (model fingerprint, target,
   workload, context, compiler/kernel revision) to prevent rediscovering
   rejected plans under the same conditions.

These changes improve product architecture, but none should alter the sealed
unique-packet execution path before its 20-pair release/quality campaign is
complete.
