# QX security and review

QX is an optimistic verification protocol. Do not put real value at risk based
only on the maintainers' tests.

Report vulnerabilities privately to the repository owner. Include the commit,
program ID, affected opcode, transaction or witness bytes, expected invariant,
and observed result. Do not publish an exploit against a funded deployment
before coordinating remediation.

An independent review should reproduce these gates from a clean checkout:

```sh
cargo test -p qx-commit -p qx-consensus-math -p qx-verifier-core -p qx-gateway
cargo build-sbf --manifest-path programs/qx-terminal-verifier/Cargo.toml
cargo test -p qx-terminal-verifier --test sbf_v2
cargo clippy -p qx-commit -p qx-gateway -p qx-devnet-adversary --all-targets --no-deps -- -D warnings
```

The review must additionally check the deployed program-data hash and authority,
repeat real Granite traces on each supported Apple GPU/OS/compiler tuple, fuzz
all instruction decoders and Merkle depths, and independently model collateral,
bond, timeout, and one-time settlement invariants.

The current testnet profile intentionally retains an upgrade authority and a
single gateway deployment. Those are explicit trust assumptions, not hidden
claims of decentralization.
