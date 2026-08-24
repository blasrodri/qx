# QX speculative decoding engineering plan

## Why this is a separate product lane

The exact QX engine currently runs one autoregressive token graph at a time.
The refreshed control is approximately 223 tok/s versus 326 tok/s for the
same-session llama decode benchmark. The remaining exact gap is no longer
supported by a single obvious kernel hypothesis.

Speculative decoding attacks a different cost model: a draft model proposes a
short sequence, and the target verifier evaluates several candidate positions
with shared target weights. For QX's routed, weight-consumption-heavy MoE,
that can turn repeated batch-1 matvec work into a more efficient multi-token
execution shape.

## Current blocker

`GraniteV2Generator` exposes `step(token)` and `step_logits(token)`. Each call
advances one KV position and submits a one-token graph. Adding a draft model
alone would not create a speedup: the target would still verify every proposed
token through the same one-token graph.

The required primitive is therefore:

```text
verify(candidates[K], state)
    -> accepted_prefix_length
    -> target_next_token
    -> committed KV state
```

The verifier must keep target weights and intermediate state GPU-resident,
handle causal attention for K candidate positions, and commit only the
accepted prefix. Rejected suffix state must not contaminate the next decode.

## Implementation phases

### Phase 1: correctness-only verifier contract

Add a verifier state model and test it against sequential target decoding:

- candidate widths `K=2,4,8`;
- exact greedy target logits;
- accepted prefix and rejection cases;
- KV position/rollback invariants;
- parity through long sequential fixtures.

This phase may use sequential execution internally. It establishes the API and
state semantics, not a throughput claim.

### Phase 2: multi-token target graph

Extend the Metal graph around a fixed speculative width. The first target
should be `K=4` for Granite/M5:

- QKV and RMS consume `[K, hidden]` activations;
- RoPE writes K candidate positions;
- attention uses causal masks for the candidate block;
- MoE runs selected routes for all candidates;
- LM head returns target logits for all K positions;
- only the accepted prefix updates the committed state.

The planner should treat speculative width, verifier topology, KV policy, and
draft source as workload-specific plan dimensions. Do not add a runtime string
branch inside the token loop.

### Phase 3: product benchmark

Compare QX speculative decoding against llama speculative decoding using the
same draft model, candidate width, prompt fixtures, acceptance accounting,
quality settings, and context range. Report:

- user-visible accepted tokens/sec;
- verifier target tokens/sec;
- acceptance rate and accepted run length;
- draft cost;
- GPU and wall latency;
- peak physical footprint;
- quality and exact output agreement.

Speculative throughput must be reported separately from exact one-token
throughput. It is a system/product result, not evidence that QX's exact
single-token kernel has reached llama parity.

## First implementation gate

Do not build a draft-model optimizer or mixed precision into the first version.
The first useful milestone is a correct fixed-width `K=4` verifier with a
CPU/reference draft adapter and a measured acceptance trace. Only after the
verifier is correct should the compiler search draft model, width, route
policy, and memory plan.
