# Qwen layers 0-3 Q3 composition

Date: 2026-09-03
Status: promoted with composition-selected exact anchor `2:76`

## Outcome

Qwen3-30B-A3B layers 0 through 3 now pass the strict 1% causal perplexity gate
as one composed Q3 policy. The promoted policy contains 454 packed Q3 students
and preserves these exact causal anchors:

- layer 1: `1:68`;
- layer 2: `2:92` and composition-selected `2:76`;
- layer 3: `3:82`.

All evidence-retained under-evidenced experts also remain on their source GGUF
payloads. The per-layer transformed counts are 119 students in layer 0, 118 in
layer 1, 106 in layer 2, and 111 in layer 3.

Both corpus partitions pass independently at context 128 and GPU20, and their
paired 343-token aggregate improves mean NLL relative to the exact control:

| Corpus slice | Scored tokens | PPL delta | Top-1 agreement |
|---|---:|---:|---:|
| selection prompts 1-4 | 125 | +0.2152% | 97.60% |
| frozen prompts 5-12 | 218 | -0.4612% | 94.50% |
| **aggregate** | **343** | **-0.2153%** | **95.63%** |

The aggregate exact mean NLL is `3.8784066855846877`, student mean NLL is
`3.8762517398280214`, and mean-NLL delta is `-0.0021549457566658317`.
The negative aggregate is finite-corpus evidence of gate clearance, not a claim
that the compact model is generally better than the teacher.

## Composition-aware selection

The initial four-layer policy kept `2:92` exact but encoded `2:76` as Q3. Its
455-student selection result narrowly failed:

| Policy | Scored tokens | PPL delta | Top-1 agreement |
|---|---:|---:|---:|
| four-layer baseline, exact `2:92` | 125 | +1.5191% | 98.40% |

A two-prompt, 58-token screen then restored six layer-2 experts independently
against the same complete four-layer baseline. The values below are each
candidate policy's delta versus the exact teacher, not a cumulative sequence:

| Exact candidate | PPL delta | Gain vs +1.3063% screen baseline | Top-1 |
|---|---:|---:|---:|
| `2:76` | **-1.8283%** | **3.1346 pp** | 98.28% |
| `2:115` | -1.0083% | 2.3146 pp | 96.55% |
| `2:47` | -0.0166% | 1.3229 pp | 98.28% |
| `2:44` | +0.2354% | 1.0709 pp | 98.28% |
| `2:117` | +1.0870% | 0.2193 pp | 98.28% |
| `2:66` | +1.3063% | 0.0000 pp | 98.28% |

`2:76` was the second-most-routed layer-2 student on the full selection slice,
but route count alone did not explain the result: `2:66` had the strongest
combined route-exposure and Q3-error score and produced no measurable causal
gain. The older Q2 causal control `2:117` also transferred weakly. Direct
downstream NLL attribution under the actual representation and composed policy
was therefore necessary.

Restoring `2:76` exact reduced the full selection result from +1.5191% to
+0.2152%, a 1.3039 percentage-point improvement. It also passed the untouched
frozen partition without further selection.

## Memory result

The diagnostic representation uses 1,021.50 MiB for 454 full-width packed Q3
students plus one shared 18 MiB route-local F32 expansion scratch. It executed
successfully at GPU20 alongside the approximately 17 GiB teacher GGUF.

At product FP16-scale Q3 density, the 454 transformed experts occupy about
893.81 MiB. Moving layer 2 from its Q2S4 policy to 106 Q3 students costs about
59.63 MiB; preserving `2:76` exact costs another approximately 0.95 MiB over
Q3. The layer-2 increment is therefore about 60.57 MiB, and the estimated Q3
precision premium across layers 0-3 is about 259.34 MiB. The complete compiled
artifact measurement below supersedes the earlier 7-8 GiB projection: retaining
all under-evidenced experts at source Q4 costs substantially more than that
projection allowed.

## Compiled artifact result

The compiler now accepts a checkpoint-specific expert policy plus the completed
student-build directory. It validates both causal evidence hashes, every layer
manifest, every student target and shape, and the declared quantization lattice.
It then writes Q2S4, Q3, and exact Q4 expert tensors into their native typed
artifact slabs. The execution plan records the explicit per-layer student and
source-exact sets, causal anchors, manifest hashes, evidence hashes, and policy
ID. `qx-inspect` verifies that plan against every expert tensor-directory entry.

The first complete encode produced:

| Measurement | Result |
|---|---:|
| Artifact file | 11,538,169,856 bytes (10.746 GiB) |
| Logical weights | 11,537,506,304 bytes (10.745 GiB) |
| Modeled total at context 4096 | 11,941,398,016 bytes (11.121 GiB) |
| Q2S4 tensor entries | 9,104 |
| Q3 tensor entries | 908 |
| Q4 tensor entries | 2,518 (2,276 expert + 242 dense) |
| Raw vectors | 193 |
| Artifact SHA-256 | `32be704346b44573da7cd5e218bd60e2f13f33be174d3048a0c5528a29f9b7e2` |

The artifact contains 4,552 Q2S4 students, 454 Q3 students, and 1,138
source-exact experts. It is 1,610,252,288 bytes (1.500 GiB) larger than the
uniform Q2S4 capacity candidate. This result is mechanically complete but
**misses the 7-8 GiB installed-weight target** and is not safe to claim as a
12 GiB iPhone deployment: iOS, Metal allocations, and application memory need
headroom beyond the modeled 11.121 GiB total.

The artifact is at
`external/models/qwen3-30b-a3b-a19pro-q2s4-q3-early4-v1.qx`; its audit is next
to it as `.audit.json`. Both large outputs remain outside Git. Reproduce the
policy preflight and encode with:

```sh
cargo run --release -p qx-compile-qwen -- \
  external/models/qwen3-30b-a3b-hf-ad44e777 \
  --source-audit benchmarks/results/qwen3-30b-a3b-bf16-source-audit-2026-08-25.json \
  --expert-policy benchmarks/qwen3-30b-a3b-four-layer-q3-policy.json \
  --expert-build-dir external/results/qwen3-full-q2s4-build \
  --output external/models/qwen3-30b-a3b-a19pro-q2s4-q3-early4-v1.qx \
  --context 4096 \
  --plan-id qwen3-30b-a3b-a19pro-q2s4-q3-early4-v1
```

This is a capacity artifact, not an execution promotion. The Qwen runtime still
needs consumer-native mixed Q2S4/Q3/Q4 dispatch, and the complete composed model
still needs a much broader quality gate. Reaching the phone memory target now
requires reducing the exact-Q4 fallback population and/or a validated narrower
student tier; further packaging optimization cannot recover the missing 2.7+
GiB by itself.

## Evidence

- failed four-layer selection baseline:
  `external/results/qwen3-full-q2s4-build/causal-layers00-01-02-03-q3-anchors68-92-82-general4-packed-gpu20.json`
- independent two-prompt causal screen:
  `external/results/qwen3-full-q2s4-build/causal-layers00-01-02-03-q3-layer2-panel-44-47-66-76-115-117-general2-independent-gpu20.json`
- promoted selection slice:
  `external/results/qwen3-full-q2s4-build/causal-layers00-01-02-03-q3-anchors68-92-76-82-general4-packed-gpu20.json`
- promoted frozen slice:
  `external/results/qwen3-full-q2s4-build/causal-layers00-01-02-03-q3-anchors68-92-76-82-untouched8-packed-gpu20.json`

The large evidence and sidecar files remain outside Git. Their model identity
is `0d003f48c75bd2f6448692cfe917e64f253f4e0e9b6fa89a50fc8c9f88f5f8f5`.

## Promotion decision

Promote the anchored layers-0/1/2/3 Q3 composition as the current compact Qwen
frontier. Preserve exact `1:68`, `2:92`, `2:76`, and `3:82`, plus all
evidence-retained source experts. This promotion is specific to the composed
policy: do not infer that the isolated layer-2 Q3 policy passes, and do not
choose exact experts from reconstruction error alone.

The heterogeneous policy is now encoded and measured. The next bounded step is
not another early-layer scan: it is to attack the 1,138 source-exact experts as
a distinct memory frontier, while preserving this artifact as the reproducible
quality/capacity control.
