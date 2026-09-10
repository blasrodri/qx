use std::path::PathBuf;

use anyhow::Result;
use clap::{Parser, Subcommand};

#[derive(Debug, Parser)]
#[command(
    name = "qx-verify",
    about = "Build and inspect QX verification commitments"
)]
struct Args {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Stream the canonical V1 weight root without retaining model weights.
    ArtifactRoot {
        #[arg(value_name = "ARTIFACT")]
        artifact: PathBuf,
    },
}

fn main() -> Result<()> {
    match Args::parse().command {
        Command::ArtifactRoot { artifact } => {
            let commitment = qx_commit::commit_q4_artifact(artifact)?;
            println!("weights_root={}", hex(&commitment.root));
            println!("weight_leaves={}", commitment.leaf_count);
            println!("tensors={}", commitment.tensor_count);
        }
    }
    Ok(())
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}
