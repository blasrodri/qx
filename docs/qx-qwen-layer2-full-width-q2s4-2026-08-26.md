# Qwen layer-2 full-width Q2S4 gate

Date: 2026-08-26  
Status: NLL pass, behavior failure; not promotion evidence

## Outcome

Keeping all 768 expert neurons avoids the catastrophic loss caused by the
width-512 transformation. Seventy-nine evidenced layer-2 experts were trained
against captured Q4 outputs with Q2S4 in gate, up, and down. They covered
97.37% of routes on the held-out capture.

On the 47-token live kill slice, perplexity changed by -2.72%, but top-1
agreement was only 65.96%. The negative loss delta is not an improvement claim:
the large top-1 disagreement proves that uniform full-width Q2S4 changes model
behavior materially.

| Metric | Result |
|---|---:|
| Exact mean NLL | 3.644961 |
| Q2S4 mean NLL | 3.617421 |
| Relative perplexity delta | -2.72% |
| Top-1 agreement | 65.96% |

## Capacity

A full-width Q2S4 expert requires 1,474,560 packed bytes. The complete model
projection is 9.246 GiB of weights, or 9.621 GiB after a 4K FP16 KV allowance.
That is too large to load completely on a 12 GB phone once Metal and application
memory are included, but it is a viable storage artifact for a bounded expert
residency runtime.

## Decision

Uniform Q2S4 is rejected for behavior quality. Before spending another training
iteration on selective Q3 promotion, QX must prove that a 3.0–4.5 GiB expert
cache can keep storage traffic within a useful iPhone throughput envelope. If
residency fails even with compact Q2S4 experts, increasing expert precision can
only make the cache problem harder.
