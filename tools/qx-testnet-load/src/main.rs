use std::{
    path::PathBuf,
    str::FromStr,
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, AtomicUsize, Ordering},
    },
    thread,
    time::{Duration, Instant},
};

use anyhow::{Context, Result, bail};
use clap::Parser;
use solana_client::rpc_client::RpcClient;
use solana_commitment_config::CommitmentConfig;
use solana_sdk::{
    instruction::Instruction,
    pubkey::Pubkey,
    signature::{Signer, read_keypair_file},
    transaction::Transaction,
};

const LOAD_DOMAIN: &[u8] = b"QX:TESTNET:LOAD:V1";
const LOAD_OPCODE: u8 = 48;
const COMPUTE_BUDGET_PROGRAM: &str = "ComputeBudget111111111111111111111111111111";

#[derive(Debug, Parser)]
#[command(
    name = "qx-testnet-load",
    about = "Send bounded QX SBF load probes to Solana testnet"
)]
struct Args {
    #[arg(long, default_value = "https://api.testnet.solana.com")]
    rpc_url: String,
    #[arg(long)]
    program: String,
    #[arg(long)]
    keypair: PathBuf,
    #[arg(long, default_value_t = 100)]
    transactions: usize,
    #[arg(long, default_value_t = 4)]
    concurrency: usize,
    #[arg(long, default_value_t = 32)]
    hash_rounds: u16,
    #[arg(long, default_value_t = 256)]
    payload_bytes: usize,
    #[arg(long, default_value_t = 400_000)]
    compute_unit_limit: u32,
}

fn main() -> Result<()> {
    let args = Args::parse();
    if args.transactions == 0 || args.concurrency == 0 {
        bail!("transactions and concurrency must be positive");
    }
    if !(1..=64).contains(&args.hash_rounds) || args.payload_bytes < 8 || args.payload_bytes > 512 {
        bail!("hash-rounds must be 1..=64 and payload-bytes must be 8..=512");
    }
    if !(200_000..=1_400_000).contains(&args.compute_unit_limit) {
        bail!("compute-unit-limit must be 200000..=1400000");
    }
    let program = Pubkey::from_str(&args.program).context("invalid program public key")?;
    let payer = Arc::new(
        read_keypair_file(&args.keypair)
            .map_err(|error| anyhow::anyhow!("read {}: {error}", args.keypair.display()))?,
    );
    let rpc = rpc_client(args.rpc_url.clone());
    let balance = rpc
        .get_balance(&payer.pubkey())
        .context("read testnet payer balance")?;
    if balance == 0 {
        bail!("testnet payer {} has zero balance", payer.pubkey());
    }
    let program_account = rpc
        .get_account(&program)
        .with_context(|| format!("load testnet program {program}"))?;
    if !program_account.executable {
        bail!("{program} is not executable");
    }
    let probe = probe_transaction(
        &rpc,
        &payer,
        program,
        u64::MAX,
        args.hash_rounds,
        args.payload_bytes,
        args.compute_unit_limit,
    )?;
    let simulation = rpc
        .simulate_transaction(&probe)
        .context("simulate representative load probe")?;
    if let Some(error) = simulation.value.err {
        bail!("representative load probe simulation failed: {error:?}");
    }
    let simulation_compute_units = simulation.value.units_consumed.unwrap_or_default();
    let transaction_fee_lamports = rpc
        .get_fee_for_message(&probe.message)
        .context("quote representative transaction fee")?;

    let next = Arc::new(AtomicUsize::new(0));
    let succeeded = Arc::new(AtomicUsize::new(0));
    let failed = Arc::new(AtomicUsize::new(0));
    let total_micros = Arc::new(AtomicU64::new(0));
    let first_error = Arc::new(Mutex::new(None::<String>));
    let started = Instant::now();
    thread::scope(|scope| {
        for _ in 0..args.concurrency {
            let rpc_url = args.rpc_url.clone();
            let payer = Arc::clone(&payer);
            let next = Arc::clone(&next);
            let succeeded = Arc::clone(&succeeded);
            let failed = Arc::clone(&failed);
            let total_micros = Arc::clone(&total_micros);
            let first_error = Arc::clone(&first_error);
            scope.spawn(move || {
                let rpc = rpc_client(rpc_url);
                loop {
                    let sequence = next.fetch_add(1, Ordering::Relaxed);
                    if sequence >= args.transactions {
                        break;
                    }
                    let sent_at = Instant::now();
                    let result = send_probe(
                        &rpc,
                        &payer,
                        program,
                        sequence as u64,
                        args.hash_rounds,
                        args.payload_bytes,
                        args.compute_unit_limit,
                    );
                    total_micros.fetch_add(
                        sent_at.elapsed().as_micros().min(u128::from(u64::MAX)) as u64,
                        Ordering::Relaxed,
                    );
                    match result {
                        Ok(()) => {
                            succeeded.fetch_add(1, Ordering::Relaxed);
                        }
                        Err(error) => {
                            failed.fetch_add(1, Ordering::Relaxed);
                            if let Ok(mut slot) = first_error.lock() {
                                if slot.is_none() {
                                    *slot = Some(format!("{error:#}"));
                                }
                            }
                        }
                    }
                }
            });
        }
    });
    let elapsed = started.elapsed();
    let succeeded = succeeded.load(Ordering::Relaxed);
    let failed = failed.load(Ordering::Relaxed);
    let submitted = succeeded + failed;
    let mean = if submitted == 0 {
        Duration::ZERO
    } else {
        Duration::from_micros(total_micros.load(Ordering::Relaxed) / submitted as u64)
    };
    println!("program={program}");
    println!("payer={}", payer.pubkey());
    println!("simulation_compute_units={simulation_compute_units}");
    println!("requested_compute_unit_limit={}", args.compute_unit_limit);
    println!("transaction_fee_lamports={transaction_fee_lamports}");
    println!("submitted={submitted}");
    println!("succeeded={succeeded}");
    println!("failed={failed}");
    println!("elapsed_seconds={:.6}", elapsed.as_secs_f64());
    println!(
        "confirmed_transactions_per_second={:.6}",
        succeeded as f64 / elapsed.as_secs_f64().max(f64::MIN_POSITIVE)
    );
    println!(
        "mean_confirmation_milliseconds={:.3}",
        mean.as_secs_f64() * 1_000.0
    );
    if failed != 0 {
        if let Ok(first_error) = first_error.lock() {
            if let Some(error) = first_error.as_deref() {
                eprintln!("first_error={error}");
            }
        }
        bail!("{failed} load transactions failed");
    }
    Ok(())
}

fn rpc_client(url: String) -> RpcClient {
    RpcClient::new_with_commitment(url, CommitmentConfig::confirmed())
}

fn send_probe(
    rpc: &RpcClient,
    payer: &solana_sdk::signature::Keypair,
    program: Pubkey,
    sequence: u64,
    rounds: u16,
    payload_bytes: usize,
    compute_unit_limit: u32,
) -> Result<()> {
    let transaction = probe_transaction(
        rpc,
        payer,
        program,
        sequence,
        rounds,
        payload_bytes,
        compute_unit_limit,
    )?;
    rpc.send_and_confirm_transaction(&transaction)
        .context("send and confirm load probe")?;
    Ok(())
}

fn probe_transaction(
    rpc: &RpcClient,
    payer: &solana_sdk::signature::Keypair,
    program: Pubkey,
    sequence: u64,
    rounds: u16,
    payload_bytes: usize,
    compute_unit_limit: u32,
) -> Result<Transaction> {
    let mut payload = vec![0x5a; payload_bytes];
    payload[..8].copy_from_slice(&sequence.to_le_bytes());
    let mut digest = domain_hash(LOAD_DOMAIN, &[&payload]);
    for round in 1..rounds {
        digest = domain_hash(LOAD_DOMAIN, &[&round.to_le_bytes(), &digest]);
    }
    let mut data = Vec::with_capacity(1 + 2 + 2 + payload.len() + 32);
    data.push(LOAD_OPCODE);
    data.extend_from_slice(&rounds.to_le_bytes());
    data.extend_from_slice(&(payload_bytes as u16).to_le_bytes());
    data.extend_from_slice(&payload);
    data.extend_from_slice(&digest);
    let instruction = Instruction::new_with_bytes(program, &data, vec![]);
    let mut budget_data = Vec::with_capacity(5);
    budget_data.push(2);
    budget_data.extend_from_slice(&compute_unit_limit.to_le_bytes());
    let budget = Instruction::new_with_bytes(
        Pubkey::from_str(COMPUTE_BUDGET_PROGRAM).expect("valid compute-budget program"),
        &budget_data,
        vec![],
    );
    let blockhash = rpc.get_latest_blockhash().context("get latest blockhash")?;
    let transaction = Transaction::new_signed_with_payer(
        &[budget, instruction],
        Some(&payer.pubkey()),
        &[payer],
        blockhash,
    );
    Ok(transaction)
}

fn domain_hash(domain: &[u8], fields: &[&[u8]]) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(domain);
    for field in fields {
        hasher.update(&(field.len() as u64).to_le_bytes());
        hasher.update(field);
    }
    *hasher.finalize().as_bytes()
}
