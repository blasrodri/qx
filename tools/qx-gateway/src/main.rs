use std::{
    fs,
    io::Read,
    path::{Path, PathBuf},
    str::FromStr,
    time::{SystemTime, UNIX_EPOCH},
};

use anyhow::{Context, Result, anyhow, bail};
use clap::{Parser, Subcommand};
use ed25519_dalek::{Signature, Verifier, VerifyingKey};
use qx_commit::{
    D_BATCH, HashAlgorithm, JobEnvelopeV1, JobPayloadV1, MerkleTree, ResultLeafV1,
    audit_selected_v1, input_hash_v1, output_hash_v1,
};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use solana_client::rpc_client::RpcClient;
use solana_commitment_config::CommitmentConfig;
use solana_sdk::{
    instruction::{AccountMeta, Instruction},
    pubkey::Pubkey,
    signature::{Keypair, Signer, read_keypair_file, write_keypair_file},
    transaction::Transaction,
};
use solana_system_interface::instruction::create_account;
use tiny_http::{Header, Method, Request, Response, Server, StatusCode};

const MAX_REQUEST_BYTES: u64 = 4 * 1024 * 1024;
const EPOCH_ACCOUNT_BYTES: usize = 152;

#[derive(Debug, Parser)]
#[command(
    name = "qx-gateway",
    about = "Store QX jobs and serve self-authenticating audit packages"
)]
struct Args {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Build and requester-sign a canonical job submission JSON file.
    CreateJob {
        #[arg(long)]
        program: String,
        #[arg(long)]
        requester_keypair: PathBuf,
        #[arg(long)]
        job_id_hex: String,
        #[arg(long)]
        assigned_miner: String,
        #[arg(long)]
        model_id_hex: String,
        #[arg(long)]
        execution_spec_hex: String,
        #[arg(long)]
        model_registry: String,
        #[arg(long, value_delimiter = ',')]
        input_tokens: Vec<u32>,
        #[arg(long)]
        max_output_tokens: u32,
        #[arg(long, default_value_t = 0)]
        payment: u64,
        #[arg(long)]
        slashable_exposure: u64,
        #[arg(long)]
        deadline_slot: u64,
        #[arg(long)]
        nonce: u64,
        #[arg(long, default_value_t = 50_000)]
        audit_probability_ppm: u32,
        #[arg(long)]
        assigned_epoch: u64,
        #[arg(long)]
        output: PathBuf,
    },
    /// Build an unsigned canonical result package from committed inference output.
    CreateResult {
        #[arg(long)]
        job: PathBuf,
        #[arg(long, value_delimiter = ',')]
        output_tokens: Vec<u32>,
        #[arg(long)]
        trace_root_hex: String,
        #[arg(long)]
        output: PathBuf,
    },
    /// Execute a canonical job with Metal commitments and emit its unsigned result.
    RunJob {
        #[arg(long)]
        job: PathBuf,
        #[arg(long)]
        artifact: PathBuf,
        #[arg(long, default_value_t = 2)]
        checkpoint_interval: usize,
        #[arg(long, default_value_t = 512)]
        context: usize,
        #[arg(long)]
        eos_token: Option<u32>,
        #[arg(long)]
        output: PathBuf,
    },
    /// Register the exact Granite Q4 artifact and verifier schedule on Solana.
    RegisterModel {
        #[arg(long)]
        artifact: PathBuf,
        /// Override the deterministic registry identity (normally omitted).
        #[arg(long)]
        model_id_hex: Option<String>,
        #[arg(long, default_value_t = 2)]
        checkpoint_interval: u8,
        #[arg(long)]
        program: String,
        #[arg(long, default_value = "https://api.devnet.solana.com")]
        rpc_url: String,
        #[arg(long)]
        payer_keypair: PathBuf,
        /// Unique model registry account keypair; created if it does not exist.
        #[arg(long)]
        model_keypair: PathBuf,
    },
    /// Run the persistent HTTP package service.
    Serve {
        #[arg(long, default_value = "127.0.0.1:8787")]
        listen: String,
        /// Local record directory. Mutually exclusive with --dodil-bucket.
        #[arg(long)]
        data_dir: Option<PathBuf>,
        /// DODIL DataK3 bucket used as durable record storage.
        #[arg(long)]
        dodil_bucket: Option<String>,
        #[arg(long)]
        program: String,
        #[arg(long, default_value = "https://api.devnet.solana.com")]
        rpc_url: String,
        /// File containing the transport token used by the miner/operator write API.
        /// If omitted, QX_MINER_TOKEN is read from the environment.
        #[arg(long)]
        miner_token_file: Option<PathBuf>,
    },
    /// Create verifier-owned accounts and commit a locally closed epoch on Solana.
    SubmitEpoch {
        #[arg(long)]
        data_dir: PathBuf,
        #[arg(long)]
        epoch: u64,
        #[arg(long)]
        program: String,
        #[arg(long, default_value = "https://api.devnet.solana.com")]
        rpc_url: String,
        #[arg(long)]
        payer_keypair: PathBuf,
        /// Persistent miner identity keypair; created if it does not exist.
        #[arg(long)]
        miner_keypair: PathBuf,
        /// Unique epoch account keypair; created if it does not exist.
        #[arg(long)]
        epoch_keypair: PathBuf,
    },
    /// Finalize an on-chain epoch from the canonical successor block hash.
    FinalizeEpoch {
        #[arg(long)]
        program: String,
        #[arg(long, default_value = "https://api.devnet.solana.com")]
        rpc_url: String,
        #[arg(long)]
        payer_keypair: PathBuf,
        #[arg(long)]
        epoch_account: String,
    },
}

#[derive(Clone)]
struct App {
    store: GatewayStore,
    program: Pubkey,
    rpc_url: String,
    miner_token: String,
}

#[derive(Clone)]
enum GatewayStore {
    Files(PathBuf),
    Dodil(DodilStore),
}

#[derive(Clone)]
struct DodilStore {
    client: reqwest::blocking::Client,
    bucket_url: reqwest::Url,
    token: String,
}

#[derive(Deserialize)]
#[serde(rename = "ListBucketResult")]
struct ObjectList {
    #[serde(rename = "Contents", default)]
    contents: Vec<ObjectListEntry>,
    #[serde(rename = "IsTruncated", default)]
    is_truncated: bool,
    #[serde(rename = "NextContinuationToken")]
    next_continuation_token: Option<String>,
}

#[derive(Deserialize)]
struct ObjectListEntry {
    #[serde(rename = "Key")]
    key: String,
}

const JOBS: &str = "jobs";
const RESULTS: &str = "results";
const EPOCHS: &str = "epochs";

#[derive(Clone, Debug, Deserialize, Serialize)]
struct JobSubmission {
    payload_hex: String,
    input_tokens: Vec<u32>,
    requester_signature_hex: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct JobRecord {
    job_hash_hex: String,
    payload_hex: String,
    input_tokens: Vec<u32>,
    requester_signature_hex: String,
    assigned_epoch: u64,
    assigned_miner_hex: String,
    received_unix_seconds: u64,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct ResultSubmission {
    result_hex: String,
    output_tokens: Vec<u32>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct ResultRecord {
    job_hash_hex: String,
    result_hex: String,
    output_tokens: Vec<u32>,
    received_unix_seconds: u64,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct ResultIndexRecord {
    job_hash_hex: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct JobIndexRecord {
    job_hash_hex: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct CloseEpochRequest {
    miner_pubkey: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct SyncEpochRequest {
    epoch_account: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct OnchainEpoch {
    epoch_account: String,
    close_slot: u64,
    entropy_slot: u64,
    audit_seed_hex: String,
    challenge_deadline_slot: u64,
    response_window_slots: u64,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct EpochRecord {
    epoch: u64,
    miner_pubkey_hex: String,
    result_root_hex: String,
    result_count: u32,
    result_job_hashes: Vec<String>,
    onchain: Option<OnchainEpoch>,
}

#[derive(Clone, Debug, Serialize)]
struct ChallengePackage {
    epoch: u64,
    epoch_account: String,
    model_registry_hex: String,
    audit_seed_hex: String,
    payload_hex: String,
    input_tokens: Vec<u32>,
    requester_signature_hex: String,
    result_hex: String,
    output_tokens: Vec<u32>,
    leaf_index: u32,
    leaf_count: u32,
    result_root_hex: String,
    result_proof_siblings_hex: Vec<String>,
}

fn main() -> Result<()> {
    match Args::parse().command {
        Command::CreateJob {
            program,
            requester_keypair,
            job_id_hex,
            assigned_miner,
            model_id_hex,
            execution_spec_hex,
            model_registry,
            input_tokens,
            max_output_tokens,
            payment,
            slashable_exposure,
            deadline_slot,
            nonce,
            audit_probability_ppm,
            assigned_epoch,
            output,
        } => create_job(
            &program,
            &requester_keypair,
            &job_id_hex,
            &assigned_miner,
            &model_id_hex,
            &execution_spec_hex,
            &model_registry,
            input_tokens,
            max_output_tokens,
            payment,
            slashable_exposure,
            deadline_slot,
            nonce,
            audit_probability_ppm,
            assigned_epoch,
            &output,
        )?,
        Command::CreateResult {
            job,
            output_tokens,
            trace_root_hex,
            output,
        } => create_result(&job, output_tokens, &trace_root_hex, &output)?,
        Command::RunJob {
            job,
            artifact,
            checkpoint_interval,
            context,
            eos_token,
            output,
        } => run_job(
            &job,
            &artifact,
            checkpoint_interval,
            context,
            eos_token,
            &output,
        )?,
        Command::RegisterModel {
            artifact,
            model_id_hex,
            checkpoint_interval,
            program,
            rpc_url,
            payer_keypair,
            model_keypair,
        } => register_model(
            &artifact,
            model_id_hex.as_deref(),
            checkpoint_interval,
            &program,
            &rpc_url,
            &payer_keypair,
            &model_keypair,
        )?,
        Command::Serve {
            listen,
            data_dir,
            dodil_bucket,
            program,
            rpc_url,
            miner_token_file,
        } => {
            let program =
                Pubkey::from_str(&program).context("invalid verifier program public key")?;
            let miner_token = match miner_token_file {
                Some(path) => fs::read_to_string(&path)
                    .with_context(|| format!("read {}", path.display()))?
                    .trim()
                    .to_owned(),
                None => std::env::var("QX_MINER_TOKEN")
                    .context("set QX_MINER_TOKEN or pass --miner-token-file")?,
            };
            if miner_token.len() < 32 {
                bail!("miner token must contain at least 32 characters");
            }
            let store = match (data_dir, dodil_bucket) {
                (Some(data_dir), None) => GatewayStore::Files(data_dir),
                (None, Some(bucket)) => GatewayStore::Dodil(DodilStore::from_env(&bucket)?),
                (None, None) => bail!("pass exactly one of --data-dir or --dodil-bucket"),
                (Some(_), Some(_)) => {
                    bail!("--data-dir and --dodil-bucket are mutually exclusive")
                }
            };
            let app = App {
                store,
                program,
                rpc_url,
                miner_token,
            };
            app.prepare()?;
            let server = Server::http(&listen).map_err(|error| anyhow!(error.to_string()))?;
            println!("listen=http://{listen}");
            println!("rpc_url={}", app.rpc_url);
            println!("program={}", app.program);
            println!("storage={}", app.store.label());
            for request in server.incoming_requests() {
                let method = request.method().clone();
                let path = request
                    .url()
                    .split('?')
                    .next()
                    .unwrap_or(request.url())
                    .to_owned();
                if let Err(error) = handle(&app, request) {
                    eprintln!("request_error method={method:?} path={path} error={error:#}");
                }
            }
        }
        Command::SubmitEpoch {
            data_dir,
            epoch,
            program,
            rpc_url,
            payer_keypair,
            miner_keypair,
            epoch_keypair,
        } => submit_epoch(
            &data_dir,
            epoch,
            &program,
            &rpc_url,
            &payer_keypair,
            &miner_keypair,
            &epoch_keypair,
        )?,
        Command::FinalizeEpoch {
            program,
            rpc_url,
            payer_keypair,
            epoch_account,
        } => finalize_epoch(&program, &rpc_url, &payer_keypair, &epoch_account)?,
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn create_job(
    program: &str,
    requester_path: &Path,
    job_id_hex: &str,
    assigned_miner: &str,
    model_id_hex: &str,
    execution_spec_hex: &str,
    model_registry: &str,
    input_tokens: Vec<u32>,
    max_output_tokens: u32,
    payment: u64,
    slashable_exposure: u64,
    deadline_slot: u64,
    nonce: u64,
    audit_probability_ppm: u32,
    assigned_epoch: u64,
    output: &Path,
) -> Result<()> {
    let program = Pubkey::from_str(program).context("invalid verifier program public key")?;
    let requester = read_keypair(requester_path, "requester")?;
    let assigned_miner = Pubkey::from_str(assigned_miner).context("invalid miner public key")?;
    let model_registry =
        Pubkey::from_str(model_registry).context("invalid model registry public key")?;
    let model_id = decode_32(model_id_hex, "model id")?;
    if input_tokens.is_empty() {
        bail!("input tokens cannot be empty");
    }
    let payload = JobPayloadV1 {
        protocol_version: 1,
        protocol_instance: program.to_bytes(),
        job_id: decode_32(job_id_hex, "job id")?,
        requester: requester.pubkey().to_bytes(),
        assigned_miner: assigned_miner.to_bytes(),
        model_id,
        execution_spec: decode_32(execution_spec_hex, "execution spec")?,
        input_hash: input_hash_v1(&model_id, &input_tokens)?,
        input_token_count: u32::try_from(input_tokens.len())?,
        max_output_tokens,
        payment,
        slashable_exposure,
        deadline_slot,
        nonce,
        audit_probability_ppm,
        assigned_epoch,
        model_registry: model_registry.to_bytes(),
    };
    let payload_bytes = payload.canonical_bytes()?;
    let job_hash = payload.hash()?;
    let signature = requester.sign_message(&job_hash);
    let submission = JobSubmission {
        payload_hex: encode_hex(&payload_bytes),
        input_tokens,
        requester_signature_hex: encode_hex(signature.as_ref()),
    };
    write_record(output, &submission)?;
    println!("job_hash_hex={}", encode_hex(&job_hash));
    println!("output={}", output.display());
    Ok(())
}

fn create_result(
    job_path: &Path,
    output_tokens: Vec<u32>,
    trace_root_hex: &str,
    output_path: &Path,
) -> Result<()> {
    if output_tokens.is_empty() {
        bail!("output tokens cannot be empty");
    }
    let job: JobSubmission = read_json_file(job_path)?;
    let payload_bytes = decode_hex_exact(
        &job.payload_hex,
        JobPayloadV1::CANONICAL_SIZE,
        "job payload",
    )?;
    let payload = JobPayloadV1::from_canonical_bytes(&payload_bytes)?;
    let output_count = u32::try_from(output_tokens.len())?;
    if output_count > payload.max_output_tokens {
        bail!("output exceeds the job maximum");
    }
    let result = ResultLeafV1 {
        job_hash: payload.hash()?,
        output_hash: output_hash_v1(&output_tokens)?,
        trace_root: decode_32(trace_root_hex, "trace root")?,
        output_token_count: output_count,
        trace_length: payload
            .input_token_count
            .checked_add(output_count)
            .ok_or_else(|| anyhow!("trace length overflow"))?,
    };
    let submission = ResultSubmission {
        result_hex: encode_hex(&result.canonical_bytes()),
        output_tokens,
    };
    write_record(output_path, &submission)?;
    println!("job_hash_hex={}", encode_hex(&result.job_hash));
    println!("result_hash_hex={}", encode_hex(&result.hash()?));
    println!("output={}", output_path.display());
    Ok(())
}

#[cfg(target_os = "macos")]
fn verified_job(path: &Path) -> Result<(JobSubmission, JobPayloadV1)> {
    let job: JobSubmission = read_json_file(path)?;
    let payload_bytes = decode_hex_exact(
        &job.payload_hex,
        JobPayloadV1::CANONICAL_SIZE,
        "job payload",
    )?;
    let payload = JobPayloadV1::from_canonical_bytes(&payload_bytes)?;
    JobEnvelopeV1 {
        header: payload,
        input_tokens: job.input_tokens.clone(),
    }
    .validate()?;
    let signature_bytes =
        decode_hex_exact(&job.requester_signature_hex, 64, "requester signature")?;
    let signature = Signature::from_slice(&signature_bytes)?;
    VerifyingKey::from_bytes(&payload.requester)?.verify(&payload.hash()?, &signature)?;
    Ok((job, payload))
}

#[cfg(target_os = "macos")]
fn run_job(
    job_path: &Path,
    artifact: &Path,
    checkpoint_interval: usize,
    context: usize,
    eos_token: Option<u32>,
    output_path: &Path,
) -> Result<()> {
    let (job, payload) = verified_job(job_path)?;
    let expected_spec = qx_commit::granite_execution_spec_with_checkpoint_interval(u8::try_from(
        checkpoint_interval,
    )?)?;
    if payload.execution_spec != expected_spec {
        bail!("job execution spec does not match checkpoint interval");
    }
    let required_context = usize::try_from(
        payload
            .input_token_count
            .checked_add(payload.max_output_tokens)
            .ok_or_else(|| anyhow!("context length overflow"))?,
    )?;
    if required_context > context || context > 4_096 {
        bail!("job does not fit the requested registered context");
    }
    let mut generator =
        qx_runtime_v2::GraniteV2Generator::load(artifact, context).map_err(anyhow::Error::msg)?;
    generator
        .set_verification_checkpoint_interval(checkpoint_interval)
        .map_err(anyhow::Error::msg)?;
    let mut committer = qx_commit::ExecutionCommitterV1::new_with_checkpoint_interval(
        payload.job_id,
        24,
        context,
        checkpoint_interval,
    )?;
    let mut next = 0_u32;
    for token in &job.input_tokens {
        next = generator
            .step_verified(*token, &mut committer)
            .map_err(anyhow::Error::msg)?;
    }
    let mut output_tokens = Vec::with_capacity(payload.max_output_tokens as usize);
    for index in 0..payload.max_output_tokens {
        output_tokens.push(next);
        if Some(next) == eos_token || index + 1 == payload.max_output_tokens {
            break;
        }
        next = generator
            .step_verified(next, &mut committer)
            .map_err(anyhow::Error::msg)?;
    }
    let trace = committer.finish()?;
    let result = trace.result_leaf(payload.hash()?, &output_tokens)?;
    let submission = ResultSubmission {
        result_hex: encode_hex(&result.canonical_bytes()),
        output_tokens,
    };
    write_record(output_path, &submission)?;
    println!("job_hash_hex={}", encode_hex(&result.job_hash));
    println!("trace_root_hex={}", encode_hex(&trace.trace_root));
    println!("trace_length={}", trace.trace_length);
    println!("output_tokens={:?}", submission.output_tokens);
    println!("output={}", output_path.display());
    Ok(())
}

#[cfg(not(target_os = "macos"))]
fn run_job(
    _job_path: &Path,
    _artifact: &Path,
    _checkpoint_interval: usize,
    _context: usize,
    _eos_token: Option<u32>,
    _output_path: &Path,
) -> Result<()> {
    bail!("verified Granite job execution requires macOS Metal")
}

fn register_model(
    artifact: &Path,
    model_id_hex: Option<&str>,
    checkpoint_interval: u8,
    program: &str,
    rpc_url: &str,
    payer_path: &Path,
    model_path: &Path,
) -> Result<()> {
    if !matches!(checkpoint_interval, 1 | 2 | 4) {
        bail!("checkpoint interval must be 1, 2, or 4");
    }
    let program = Pubkey::from_str(program).context("invalid verifier program public key")?;
    let payer = read_keypair(payer_path, "payer")?;
    let (model, model_created) = read_or_create_keypair(model_path)?;
    let rpc = RpcClient::new_with_commitment(rpc_url.to_owned(), CommitmentConfig::confirmed());
    if rpc.get_account(&model.pubkey()).is_ok() {
        bail!("model registry account already exists");
    }
    qx_commit::validate_granite_macro_artifact_v1(artifact)?;
    let weights = qx_commit::commit_q4_artifact(artifact)?;
    let execution_spec =
        qx_commit::granite_execution_spec_with_checkpoint_interval(checkpoint_interval)?;
    let plan = qx_commit::granite_macro_plan_v1(checkpoint_interval)?;
    let model_id = match model_id_hex {
        Some(value) => decode_32(value, "model id")?,
        None => qx_commit::hash_domain(
            HashAlgorithm::Sha256V1,
            b"QX:MODEL:REGISTRY:V1",
            &[
                &weights.root,
                &weights.leaf_count.to_le_bytes(),
                &execution_spec,
                &plan.schedule.root(),
                &24_u16.to_le_bytes(),
                &u16::from(checkpoint_interval).to_le_bytes(),
                &4_096_u32.to_le_bytes(),
                &49_155_u32.to_le_bytes(),
            ],
        )?,
    };
    if model_id == [0; 32] {
        bail!("model id cannot be zero");
    }
    let checkpoint_count = 24_u32 / u32::from(checkpoint_interval) + 1;
    let mut body = Vec::with_capacity(156);
    body.push(6);
    body.extend_from_slice(&model_id);
    body.extend_from_slice(&weights.root);
    body.extend_from_slice(&weights.leaf_count.to_le_bytes());
    body.extend_from_slice(&execution_spec);
    body.extend_from_slice(&plan.schedule.root());
    body.extend_from_slice(&u32::try_from(plan.segments.len())?.to_le_bytes());
    body.extend_from_slice(&checkpoint_count.to_le_bytes());
    body.extend_from_slice(&24_u16.to_le_bytes());
    body.extend_from_slice(&u16::from(checkpoint_interval).to_le_bytes());
    body.extend_from_slice(&4_096_u32.to_le_bytes());
    body.extend_from_slice(&49_155_u32.to_le_bytes());
    let create = create_account(
        &payer.pubkey(),
        &model.pubkey(),
        rpc.get_minimum_balance_for_rent_exemption(156)?,
        156,
        &program,
    );
    let register =
        Instruction::new_with_bytes(program, &body, vec![AccountMeta::new(model.pubkey(), true)]);
    let blockhash = rpc.get_latest_blockhash()?;
    let transaction = Transaction::new_signed_with_payer(
        &[create, register],
        Some(&payer.pubkey()),
        &[&payer, &model],
        blockhash,
    );
    let signature = rpc.send_and_confirm_transaction(&transaction)?;
    println!("program={program}");
    println!("model_registry={}", model.pubkey());
    println!("model_id_hex={}", encode_hex(&model_id));
    println!("weights_root_hex={}", encode_hex(&weights.root));
    println!("weight_leaves={}", weights.leaf_count);
    println!("execution_spec_hex={}", encode_hex(&execution_spec));
    println!(
        "verification_schedule_root_hex={}",
        encode_hex(&plan.schedule.root())
    );
    println!("checkpoint_interval={checkpoint_interval}");
    println!("signature={signature}");
    println!("model_keypair_created={model_created}");
    Ok(())
}

fn submit_epoch(
    data_dir: &Path,
    epoch_number: u64,
    program: &str,
    rpc_url: &str,
    payer_path: &Path,
    miner_path: &Path,
    epoch_path: &Path,
) -> Result<()> {
    let program = Pubkey::from_str(program).context("invalid verifier program public key")?;
    let record: EpochRecord =
        read_json_file(&data_dir.join("epochs").join(format!("{epoch_number}.json")))?;
    if record.epoch != epoch_number || record.onchain.is_some() {
        bail!("local epoch is invalid or already synced");
    }
    let payer = read_keypair(payer_path, "payer")?;
    let (miner, miner_created) = read_or_create_keypair(miner_path)?;
    let (epoch, epoch_created) = read_or_create_keypair(epoch_path)?;
    if encode_hex(miner.pubkey().as_ref()) != record.miner_pubkey_hex {
        bail!("miner keypair does not match the locally closed epoch");
    }
    let rpc = RpcClient::new_with_commitment(rpc_url.to_owned(), CommitmentConfig::confirmed());
    let mut instructions = Vec::new();
    match rpc.get_account(&miner.pubkey()) {
        Ok(account) => {
            if account.owner != program || account.data.len() != 16 {
                bail!("existing miner account has the wrong owner or size");
            }
        }
        Err(_) => instructions.push(create_account(
            &payer.pubkey(),
            &miner.pubkey(),
            rpc.get_minimum_balance_for_rent_exemption(16)?,
            16,
            &program,
        )),
    }
    if rpc.get_account(&epoch.pubkey()).is_ok() {
        bail!("epoch account already exists; use a fresh epoch keypair path");
    }
    instructions.push(create_account(
        &payer.pubkey(),
        &epoch.pubkey(),
        rpc.get_minimum_balance_for_rent_exemption(EPOCH_ACCOUNT_BYTES)?,
        EPOCH_ACCOUNT_BYTES as u64,
        &program,
    ));
    let mut body = Vec::with_capacity(45);
    body.push(9);
    body.extend_from_slice(&record.epoch.to_le_bytes());
    body.extend_from_slice(&decode_32(&record.result_root_hex, "result root")?);
    body.extend_from_slice(&record.result_count.to_le_bytes());
    instructions.push(Instruction::new_with_bytes(
        program,
        &body,
        vec![
            AccountMeta::new(epoch.pubkey(), true),
            AccountMeta::new(miner.pubkey(), true),
        ],
    ));
    let blockhash = rpc.get_latest_blockhash()?;
    let transaction = Transaction::new_signed_with_payer(
        &instructions,
        Some(&payer.pubkey()),
        &[&payer, &miner, &epoch],
        blockhash,
    );
    let signature = rpc.send_and_confirm_transaction(&transaction)?;
    println!("program={program}");
    println!("epoch={epoch_number}");
    println!("epoch_account={}", epoch.pubkey());
    println!("miner_account={}", miner.pubkey());
    println!("result_root={}", record.result_root_hex);
    println!("result_count={}", record.result_count);
    println!("signature={signature}");
    println!("miner_keypair_created={miner_created}");
    println!("epoch_keypair_created={epoch_created}");
    Ok(())
}

fn finalize_epoch(program: &str, rpc_url: &str, payer_path: &Path, epoch: &str) -> Result<()> {
    let program = Pubkey::from_str(program).context("invalid verifier program public key")?;
    let epoch = Pubkey::from_str(epoch).context("invalid epoch account public key")?;
    let payer = read_keypair(payer_path, "payer")?;
    let rpc = RpcClient::new_with_commitment(rpc_url.to_owned(), CommitmentConfig::confirmed());
    let instruction =
        Instruction::new_with_bytes(program, &[10], vec![AccountMeta::new(epoch, false)]);
    let blockhash = rpc.get_latest_blockhash()?;
    let transaction = Transaction::new_signed_with_payer(
        &[instruction],
        Some(&payer.pubkey()),
        &[&payer],
        blockhash,
    );
    let signature = rpc.send_and_confirm_transaction(&transaction)?;
    println!("program={program}");
    println!("epoch_account={epoch}");
    println!("signature={signature}");
    Ok(())
}

fn read_keypair(path: &Path, label: &str) -> Result<Keypair> {
    read_keypair_file(path)
        .map_err(|error| anyhow!("read {label} keypair {}: {error}", path.display()))
}

fn read_or_create_keypair(path: &Path) -> Result<(Keypair, bool)> {
    if path.exists() {
        return Ok((read_keypair(path, "protocol")?, false));
    }
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let keypair = Keypair::new();
    write_keypair_file(&keypair, path)
        .map_err(|error| anyhow!("write keypair {}: {error}", path.display()))?;
    Ok((keypair, true))
}

fn handle(app: &App, mut request: Request) -> Result<()> {
    let method = request.method().clone();
    let path = request
        .url()
        .split('?')
        .next()
        .unwrap_or(request.url())
        .to_owned();
    let parts = path
        .trim_matches('/')
        .split('/')
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>();
    let response: HttpResult = (|| match (method, parts.as_slice()) {
        (Method::Get, ["health"]) => app.health(),
        (Method::Post, ["v1", "jobs"]) => {
            let body = read_json::<JobSubmission>(&mut request).map_err(HttpError::bad)?;
            app.put_job(body)
        }
        (Method::Get, ["v1", "jobs", job_hash]) => app.get_job(job_hash),
        (Method::Get, ["v1", "miners", miner, "jobs"]) => {
            require_miner(app, &request)?;
            app.pending_jobs(miner)
        }
        (Method::Post, ["v1", "results"]) => {
            require_miner(app, &request)?;
            let body = read_json::<ResultSubmission>(&mut request).map_err(HttpError::bad)?;
            app.put_result(body)
        }
        (Method::Get, ["v1", "results", job_hash]) => app.get_result(job_hash),
        (Method::Post, ["v1", "epochs", epoch, "close"]) => {
            require_miner(app, &request)?;
            let epoch = epoch.parse::<u64>().map_err(HttpError::bad)?;
            let body = read_json::<CloseEpochRequest>(&mut request).map_err(HttpError::bad)?;
            app.close_epoch(epoch, body)
        }
        (Method::Post, ["v1", "epochs", epoch, "sync"]) => {
            require_miner(app, &request)?;
            let epoch = epoch.parse::<u64>().map_err(HttpError::bad)?;
            let body = read_json::<SyncEpochRequest>(&mut request).map_err(HttpError::bad)?;
            app.sync_epoch(epoch, body)
        }
        (Method::Get, ["v1", "epochs", epoch]) => {
            let epoch = epoch.parse::<u64>().map_err(HttpError::bad)?;
            app.get_epoch(epoch)
        }
        (Method::Get, ["v1", "epochs", epoch, "audits"]) => {
            let epoch = epoch.parse::<u64>().map_err(HttpError::bad)?;
            app.audit_packages(epoch)
        }
        (Method::Get, ["v1", "challenges", job_hash]) => app.challenge_package(job_hash),
        _ => Err(HttpError::new(404, "not found")),
    })();
    match response {
        Ok(value) => send_json(request, 200, &value),
        Err(error) => send_json(
            request,
            error.status,
            &serde_json::json!({ "error": error.message }),
        ),
    }
}

#[derive(Debug)]
struct HttpError {
    status: u16,
    message: String,
}

impl HttpError {
    fn new(status: u16, message: impl Into<String>) -> Self {
        Self {
            status,
            message: message.into(),
        }
    }

    fn bad(error: impl std::fmt::Display) -> Self {
        Self::new(400, error.to_string())
    }

    fn internal(error: impl std::fmt::Display) -> Self {
        Self::new(500, error.to_string())
    }
}

type HttpResult = std::result::Result<serde_json::Value, HttpError>;

impl App {
    fn prepare(&self) -> Result<()> {
        self.store.prepare()
    }

    fn health(&self) -> HttpResult {
        Ok(serde_json::json!({
            "status": "ok",
            "storage": self.store.label(),
            "cluster_rpc": self.rpc_url,
            "program": self.program.to_string(),
        }))
    }

    fn put_job(&self, submission: JobSubmission) -> HttpResult {
        let payload_bytes = decode_hex_exact(
            &submission.payload_hex,
            JobPayloadV1::CANONICAL_SIZE,
            "payload_hex",
        )
        .map_err(HttpError::bad)?;
        let payload = JobPayloadV1::from_canonical_bytes(&payload_bytes).map_err(HttpError::bad)?;
        if payload.protocol_instance != self.program.to_bytes() {
            return Err(HttpError::new(
                409,
                "job targets a different verifier program",
            ));
        }
        let envelope = JobEnvelopeV1 {
            header: payload,
            input_tokens: submission.input_tokens.clone(),
        };
        envelope.validate().map_err(HttpError::bad)?;
        let signature_bytes = decode_hex_exact(
            &submission.requester_signature_hex,
            64,
            "requester_signature_hex",
        )
        .map_err(HttpError::bad)?;
        let signature = Signature::from_slice(&signature_bytes).map_err(HttpError::bad)?;
        let requester = VerifyingKey::from_bytes(&payload.requester).map_err(HttpError::bad)?;
        let job_hash = payload.hash().map_err(HttpError::bad)?;
        requester
            .verify(&job_hash, &signature)
            .map_err(|_| HttpError::new(401, "invalid requester signature"))?;
        let job_hash_hex = encode_hex(&job_hash);
        let record = JobRecord {
            job_hash_hex: job_hash_hex.clone(),
            payload_hex: encode_hex(&payload_bytes),
            input_tokens: submission.input_tokens,
            requester_signature_hex: encode_hex(&signature_bytes),
            assigned_epoch: payload.assigned_epoch,
            assigned_miner_hex: encode_hex(&payload.assigned_miner),
            received_unix_seconds: unix_seconds().map_err(HttpError::bad)?,
        };
        if let Some(existing) = self
            .store
            .get::<JobRecord>(JOBS, &job_hash_hex)
            .map_err(HttpError::internal)?
        {
            if existing.payload_hex != record.payload_hex
                || existing.input_tokens != record.input_tokens
                || existing.requester_signature_hex != record.requester_signature_hex
            {
                return Err(HttpError::new(409, "conflicting job already exists"));
            }
        } else {
            self.store
                .put_if_absent(JOBS, &job_hash_hex, &record)
                .map_err(HttpError::internal)?;
        }
        let index_kind = job_index_kind(&record.assigned_miner_hex);
        self.store
            .put_if_absent(
                &index_kind,
                &job_hash_hex,
                &JobIndexRecord {
                    job_hash_hex: job_hash_hex.clone(),
                },
            )
            .map_err(HttpError::internal)?;
        Ok(serde_json::json!({ "job_hash_hex": job_hash_hex, "status": "accepted" }))
    }

    fn get_job(&self, job_hash: &str) -> HttpResult {
        validate_hash_name(job_hash)?;
        let record: JobRecord = self.record(JOBS, job_hash)?;
        serde_json::to_value(record).map_err(HttpError::bad)
    }

    fn pending_jobs(&self, miner: &str) -> HttpResult {
        let miner = Pubkey::from_str(miner).map_err(HttpError::bad)?;
        let index_kind = job_index_kind(&encode_hex(miner.as_ref()));
        let mut jobs = Vec::new();
        for index_record in self
            .store
            .list::<JobIndexRecord>(&index_kind)
            .map_err(HttpError::internal)?
        {
            if self
                .store
                .get::<ResultRecord>(RESULTS, &index_record.job_hash_hex)
                .map_err(HttpError::internal)?
                .is_none()
            {
                let job: JobRecord = self.record(JOBS, &index_record.job_hash_hex)?;
                jobs.push(job);
            }
        }
        jobs.sort_by_key(|job| {
            (
                job.assigned_epoch,
                job.received_unix_seconds,
                job.job_hash_hex.clone(),
            )
        });
        Ok(serde_json::json!({ "jobs": jobs }))
    }

    fn get_result(&self, job_hash: &str) -> HttpResult {
        validate_hash_name(job_hash)?;
        let record: ResultRecord = self.record(RESULTS, job_hash)?;
        serde_json::to_value(record).map_err(HttpError::bad)
    }

    fn put_result(&self, submission: ResultSubmission) -> HttpResult {
        let result_bytes = decode_hex_exact(
            &submission.result_hex,
            ResultLeafV1::CANONICAL_SIZE,
            "result_hex",
        )
        .map_err(HttpError::bad)?;
        let result = ResultLeafV1::from_canonical_bytes(&result_bytes).map_err(HttpError::bad)?;
        let job_hash_hex = encode_hex(&result.job_hash);
        let job: JobRecord = self.record(JOBS, &job_hash_hex)?;
        let payload_bytes = decode_hex_exact(
            &job.payload_hex,
            JobPayloadV1::CANONICAL_SIZE,
            "stored payload",
        )
        .map_err(HttpError::bad)?;
        let payload = JobPayloadV1::from_canonical_bytes(&payload_bytes).map_err(HttpError::bad)?;
        if submission.output_tokens.len() != result.output_token_count as usize
            || result.output_token_count > payload.max_output_tokens
            || payload
                .input_token_count
                .checked_add(result.output_token_count)
                != Some(result.trace_length)
        {
            return Err(HttpError::new(409, "result counts do not match the job"));
        }
        let output_hash = output_hash_v1(&submission.output_tokens).map_err(HttpError::bad)?;
        if output_hash != result.output_hash {
            return Err(HttpError::new(
                409,
                "output tokens do not match output_hash",
            ));
        }
        let record = ResultRecord {
            job_hash_hex: job_hash_hex.clone(),
            result_hex: encode_hex(&result_bytes),
            output_tokens: submission.output_tokens,
            received_unix_seconds: unix_seconds().map_err(HttpError::bad)?,
        };
        if let Some(existing) = self
            .store
            .get::<ResultRecord>(RESULTS, &job_hash_hex)
            .map_err(HttpError::internal)?
        {
            if existing.result_hex != record.result_hex
                || existing.output_tokens != record.output_tokens
            {
                return Err(HttpError::new(409, "conflicting result already exists"));
            }
        } else {
            self.store
                .put_if_absent(RESULTS, &job_hash_hex, &record)
                .map_err(HttpError::internal)?;
        }
        let index_kind = result_index_kind(job.assigned_epoch, &job.assigned_miner_hex);
        let index_record = ResultIndexRecord {
            job_hash_hex: job_hash_hex.clone(),
        };
        self.store
            .put_if_absent(&index_kind, &job_hash_hex, &index_record)
            .map_err(HttpError::internal)?;
        Ok(serde_json::json!({ "job_hash_hex": job_hash_hex, "status": "accepted" }))
    }

    fn close_epoch(&self, epoch: u64, request: CloseEpochRequest) -> HttpResult {
        let miner = Pubkey::from_str(&request.miner_pubkey).map_err(HttpError::bad)?;
        let mut selected = Vec::<(String, ResultRecord, ResultLeafV1)>::new();
        let index_kind = result_index_kind(epoch, &encode_hex(miner.as_ref()));
        for index_record in self
            .store
            .list::<ResultIndexRecord>(&index_kind)
            .map_err(HttpError::internal)?
        {
            let job: JobRecord = self.record(JOBS, &index_record.job_hash_hex)?;
            if job.assigned_epoch != epoch || job.assigned_miner_hex != encode_hex(miner.as_ref()) {
                continue;
            }
            let result_record: ResultRecord = self.record(RESULTS, &index_record.job_hash_hex)?;
            let bytes = decode_hex_exact(
                &result_record.result_hex,
                ResultLeafV1::CANONICAL_SIZE,
                "stored result",
            )
            .map_err(HttpError::bad)?;
            let result = ResultLeafV1::from_canonical_bytes(&bytes).map_err(HttpError::bad)?;
            selected.push((result_record.job_hash_hex.clone(), result_record, result));
        }
        selected.sort_by(|a, b| a.0.cmp(&b.0));
        if selected.is_empty() {
            return Err(HttpError::new(409, "epoch has no completed jobs"));
        }
        let leaves = selected
            .iter()
            .map(|(_, _, result)| result.hash().map_err(HttpError::bad))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let tree =
            MerkleTree::new(HashAlgorithm::Sha256V1, D_BATCH, &leaves).map_err(HttpError::bad)?;
        let record = EpochRecord {
            epoch,
            miner_pubkey_hex: encode_hex(miner.as_ref()),
            result_root_hex: encode_hex(&tree.root()),
            result_count: u32::try_from(selected.len()).map_err(HttpError::bad)?,
            result_job_hashes: selected.into_iter().map(|entry| entry.0).collect(),
            onchain: None,
        };
        let epoch_key = epoch.to_string();
        if let Some(existing) = self
            .store
            .get::<EpochRecord>(EPOCHS, &epoch_key)
            .map_err(HttpError::internal)?
        {
            if existing.miner_pubkey_hex != record.miner_pubkey_hex
                || existing.result_root_hex != record.result_root_hex
                || existing.result_count != record.result_count
                || existing.result_job_hashes != record.result_job_hashes
            {
                return Err(HttpError::new(409, "conflicting epoch already exists"));
            }
            return serde_json::to_value(existing).map_err(HttpError::bad);
        }
        self.store
            .put_if_absent(EPOCHS, &epoch_key, &record)
            .map_err(HttpError::internal)?;
        serde_json::to_value(record).map_err(HttpError::bad)
    }

    fn sync_epoch(&self, epoch: u64, request: SyncEpochRequest) -> HttpResult {
        let epoch_key = epoch.to_string();
        let mut record: EpochRecord = self.record(EPOCHS, &epoch_key)?;
        let epoch_account = Pubkey::from_str(&request.epoch_account).map_err(HttpError::bad)?;
        let rpc =
            RpcClient::new_with_commitment(self.rpc_url.clone(), CommitmentConfig::confirmed());
        let account = rpc.get_account(&epoch_account).map_err(HttpError::bad)?;
        if account.owner != self.program || account.data.len() != EPOCH_ACCOUNT_BYTES {
            return Err(HttpError::new(409, "invalid verifier-owned epoch account"));
        }
        let data = &account.data;
        if &data[..4] != b"QXE2" || data[4] != 2 {
            return Err(HttpError::new(409, "epoch is not seeded on chain"));
        }
        let chain_epoch = read_u64(data, 8).map_err(HttpError::bad)?;
        let chain_miner = read_32(data, 16).map_err(HttpError::bad)?;
        let chain_root = read_32(data, 48).map_err(HttpError::bad)?;
        let chain_count = read_u32(data, 80).map_err(HttpError::bad)?;
        if chain_epoch != record.epoch
            || encode_hex(&chain_miner) != record.miner_pubkey_hex
            || encode_hex(&chain_root) != record.result_root_hex
            || chain_count != record.result_count
        {
            return Err(HttpError::new(
                409,
                "on-chain epoch does not match local manifest",
            ));
        }
        record.onchain = Some(OnchainEpoch {
            epoch_account: epoch_account.to_string(),
            close_slot: read_u64(data, 88).map_err(HttpError::bad)?,
            entropy_slot: read_u64(data, 96).map_err(HttpError::bad)?,
            audit_seed_hex: encode_hex(&read_32(data, 104).map_err(HttpError::bad)?),
            challenge_deadline_slot: read_u64(data, 136).map_err(HttpError::bad)?,
            response_window_slots: read_u64(data, 144).map_err(HttpError::bad)?,
        });
        self.store
            .set(EPOCHS, &epoch_key, &record)
            .map_err(HttpError::internal)?;
        serde_json::to_value(record).map_err(HttpError::bad)
    }

    fn get_epoch(&self, epoch: u64) -> HttpResult {
        let record: EpochRecord = self.record(EPOCHS, &epoch.to_string())?;
        serde_json::to_value(record).map_err(HttpError::bad)
    }

    fn audit_packages(&self, epoch: u64) -> HttpResult {
        let record: EpochRecord = self.record(EPOCHS, &epoch.to_string())?;
        let mut packages = Vec::new();
        for job_hash in &record.result_job_hashes {
            if let Some(package) = self.package_if_selected(&record, job_hash)? {
                packages.push(package);
            }
        }
        serde_json::to_value(packages).map_err(HttpError::bad)
    }

    fn challenge_package(&self, job_hash: &str) -> HttpResult {
        validate_hash_name(job_hash)?;
        let job: JobRecord = self.record(JOBS, job_hash)?;
        let epoch: EpochRecord = self.record(EPOCHS, &job.assigned_epoch.to_string())?;
        self.package_if_selected(&epoch, job_hash)?
            .ok_or_else(|| HttpError::new(403, "job was not selected for mandatory audit"))
            .and_then(|package| serde_json::to_value(package).map_err(HttpError::bad))
    }

    fn package_if_selected(
        &self,
        epoch: &EpochRecord,
        job_hash: &str,
    ) -> std::result::Result<Option<ChallengePackage>, HttpError> {
        let onchain = epoch
            .onchain
            .as_ref()
            .ok_or_else(|| HttpError::new(409, "epoch has not been synced from Solana"))?;
        let index = epoch
            .result_job_hashes
            .iter()
            .position(|candidate| candidate == job_hash)
            .ok_or_else(|| HttpError::new(404, "result is not in the epoch"))?;
        let job: JobRecord = self.record(JOBS, job_hash)?;
        let result: ResultRecord = self.record(RESULTS, job_hash)?;
        let payload_bytes = decode_hex_exact(
            &job.payload_hex,
            JobPayloadV1::CANONICAL_SIZE,
            "stored payload",
        )
        .map_err(HttpError::bad)?;
        let payload = JobPayloadV1::from_canonical_bytes(&payload_bytes).map_err(HttpError::bad)?;
        let seed = decode_32(&onchain.audit_seed_hex, "audit seed").map_err(HttpError::bad)?;
        let selected = audit_selected_v1(
            &seed,
            epoch.epoch,
            u32::try_from(index).map_err(HttpError::bad)?,
            &payload.hash().map_err(HttpError::bad)?,
            payload.audit_probability_ppm,
        )
        .map_err(HttpError::bad)?;
        if !selected {
            return Ok(None);
        }
        let tree = self.result_tree(epoch)?;
        let proof = tree.proof(index).map_err(HttpError::bad)?;
        Ok(Some(ChallengePackage {
            epoch: epoch.epoch,
            epoch_account: onchain.epoch_account.clone(),
            model_registry_hex: encode_hex(&payload.model_registry),
            audit_seed_hex: onchain.audit_seed_hex.clone(),
            payload_hex: job.payload_hex,
            input_tokens: job.input_tokens,
            requester_signature_hex: job.requester_signature_hex,
            result_hex: result.result_hex,
            output_tokens: result.output_tokens,
            leaf_index: u32::try_from(index).map_err(HttpError::bad)?,
            leaf_count: epoch.result_count,
            result_root_hex: epoch.result_root_hex.clone(),
            result_proof_siblings_hex: proof.siblings.iter().map(|item| encode_hex(item)).collect(),
        }))
    }

    fn result_tree(&self, epoch: &EpochRecord) -> std::result::Result<MerkleTree, HttpError> {
        let leaves = epoch
            .result_job_hashes
            .iter()
            .map(|job_hash| {
                let record: ResultRecord = self.record(RESULTS, job_hash)?;
                let bytes = decode_hex_exact(
                    &record.result_hex,
                    ResultLeafV1::CANONICAL_SIZE,
                    "stored result",
                )
                .map_err(HttpError::bad)?;
                ResultLeafV1::from_canonical_bytes(&bytes)
                    .and_then(|result| result.hash())
                    .map_err(HttpError::bad)
            })
            .collect::<std::result::Result<Vec<_>, _>>()?;
        MerkleTree::new(HashAlgorithm::Sha256V1, D_BATCH, &leaves).map_err(HttpError::bad)
    }

    fn record<T: DeserializeOwned>(&self, kind: &str, key: &str) -> Result<T, HttpError> {
        self.store
            .get(kind, key)
            .map_err(HttpError::internal)?
            .ok_or_else(|| HttpError::new(404, "record not found"))
    }
}

impl GatewayStore {
    fn label(&self) -> &'static str {
        match self {
            Self::Files(_) => "local-files",
            Self::Dodil(_) => "dodil-datak3",
        }
    }

    fn prepare(&self) -> Result<()> {
        match self {
            Self::Files(root) => {
                for name in [JOBS, RESULTS, EPOCHS] {
                    fs::create_dir_all(root.join(name))?;
                }
                Ok(())
            }
            Self::Dodil(store) => store.prepare(),
        }
    }

    fn get<T: DeserializeOwned>(&self, kind: &str, key: &str) -> Result<Option<T>> {
        let bytes = match self {
            Self::Files(root) => {
                let path = record_path(root, kind, key);
                if !path.exists() {
                    return Ok(None);
                }
                fs::read(&path).with_context(|| format!("read {}", path.display()))?
            }
            Self::Dodil(store) => match store.get_bytes(kind, key)? {
                Some(value) => value,
                None => return Ok(None),
            },
        };
        serde_json::from_slice(&bytes)
            .context("decode stored record")
            .map(Some)
    }

    fn list<T: DeserializeOwned>(&self, kind: &str) -> Result<Vec<T>> {
        match self {
            Self::Files(root) => json_entries(&root.join(kind))?
                .iter()
                .map(|path| read_json_file(path))
                .collect(),
            Self::Dodil(store) => store
                .list_bytes(kind)?
                .iter()
                .map(|bytes| serde_json::from_slice(bytes).context("decode stored record"))
                .collect(),
        }
    }

    fn put_if_absent(&self, kind: &str, key: &str, value: &impl Serialize) -> Result<()> {
        let bytes = serde_json::to_vec_pretty(value)?;
        match self {
            Self::Files(root) => {
                let path = record_path(root, kind, key);
                if path.exists() {
                    let existing = fs::read(&path)?;
                    if existing != bytes {
                        bail!("conflicting record already exists");
                    }
                    return Ok(());
                }
                write_record_bytes(&path, &bytes)
            }
            Self::Dodil(store) => store.put_if_absent(kind, key, &bytes),
        }
    }

    fn set(&self, kind: &str, key: &str, value: &impl Serialize) -> Result<()> {
        let bytes = serde_json::to_vec_pretty(value)?;
        match self {
            Self::Files(root) => write_record_bytes(&record_path(root, kind, key), &bytes),
            Self::Dodil(store) => store.set(kind, key, &bytes),
        }
    }
}

impl DodilStore {
    fn from_env(bucket: &str) -> Result<Self> {
        if bucket.is_empty()
            || !bucket
                .bytes()
                .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'-')
        {
            bail!("DODIL bucket must contain only lowercase letters, digits, and hyphens");
        }
        let token =
            std::env::var("DODIL_TOKEN").context("DODIL_TOKEN is required with --dodil-bucket")?;
        if !token.starts_with("dk_") || token.len() < 32 {
            bail!("DODIL_TOKEN must be a DODIL API key");
        }
        let endpoint = std::env::var("DODIL_OBJECT_ENDPOINT")
            .unwrap_or_else(|_| "https://object.uk-lon-1.dodil.io".to_owned());
        let bucket_url =
            reqwest::Url::parse(&format!("{}/{bucket}", endpoint.trim_end_matches('/')))?;
        Ok(Self {
            client: reqwest::blocking::Client::builder()
                .timeout(std::time::Duration::from_secs(30))
                .build()?,
            bucket_url,
            token,
        })
    }

    fn prepare(&self) -> Result<()> {
        self.list_keys(JOBS)?;
        Ok(())
    }

    fn get_bytes(&self, kind: &str, key: &str) -> Result<Option<Vec<u8>>> {
        let response = self
            .client
            .get(self.object_url(kind, key)?)
            .bearer_auth(&self.token)
            .send()
            .context("read DODIL DataK3 object")?;
        if response.status() == reqwest::StatusCode::NOT_FOUND {
            return Ok(None);
        }
        let response = checked_dodil_response(response)?;
        Ok(Some(response.bytes()?.to_vec()))
    }

    fn list_bytes(&self, kind: &str) -> Result<Vec<Vec<u8>>> {
        self.list_keys(kind)?
            .iter()
            .map(|key| {
                self.get_key(key)?
                    .ok_or_else(|| anyhow!("DODIL object disappeared while listing"))
            })
            .collect()
    }

    fn put_if_absent(&self, kind: &str, key: &str, bytes: &[u8]) -> Result<()> {
        let response = self
            .client
            .put(self.object_url(kind, key)?)
            .bearer_auth(&self.token)
            .header("Content-Type", "application/json")
            .header("If-None-Match", "*")
            .body(bytes.to_vec())
            .send()
            .context("write DODIL DataK3 object")?;
        if response.status() == reqwest::StatusCode::PRECONDITION_FAILED {
            let stored = self
                .get_bytes(kind, key)?
                .ok_or_else(|| anyhow!("DODIL object disappeared after write conflict"))?;
            if stored != bytes {
                bail!("conflicting record already exists");
            }
            return Ok(());
        }
        checked_dodil_response(response)?;
        Ok(())
    }

    fn set(&self, kind: &str, key: &str, bytes: &[u8]) -> Result<()> {
        self.put(kind, key, bytes)
    }

    fn put(&self, kind: &str, key: &str, bytes: &[u8]) -> Result<()> {
        let response = self
            .client
            .put(self.object_url(kind, key)?)
            .bearer_auth(&self.token)
            .header("Content-Type", "application/json")
            .body(bytes.to_vec())
            .send()
            .context("write DODIL DataK3 object")?;
        checked_dodil_response(response)?;
        Ok(())
    }

    fn list_keys(&self, kind: &str) -> Result<Vec<String>> {
        let prefix = format!("{kind}/");
        let mut continuation: Option<String> = None;
        let mut keys = Vec::new();
        loop {
            let mut request = self
                .client
                .get(self.bucket_url.clone())
                .bearer_auth(&self.token);
            request = request.query(&[("list-type", "2"), ("prefix", prefix.as_str())]);
            if let Some(token) = continuation.as_deref() {
                request = request.query(&[("continuation-token", token)]);
            }
            let body = checked_dodil_response(request.send().context("list DODIL objects")?)?
                .text()
                .context("read DODIL object list")?;
            let page: ObjectList =
                quick_xml::de::from_str(&body).context("decode DODIL object list")?;
            keys.extend(page.contents.into_iter().map(|entry| entry.key));
            if !page.is_truncated {
                break;
            }
            continuation = Some(
                page.next_continuation_token
                    .ok_or_else(|| anyhow!("truncated DODIL list has no continuation token"))?,
            );
        }
        keys.sort();
        Ok(keys)
    }

    fn get_key(&self, object_key: &str) -> Result<Option<Vec<u8>>> {
        let mut url = self.bucket_url.clone();
        url.path_segments_mut()
            .map_err(|_| anyhow!("DODIL object endpoint cannot be a base URL"))?
            .extend(object_key.split('/'));
        let response = self
            .client
            .get(url)
            .bearer_auth(&self.token)
            .send()
            .context("read DODIL DataK3 object")?;
        if response.status() == reqwest::StatusCode::NOT_FOUND {
            return Ok(None);
        }
        Ok(Some(checked_dodil_response(response)?.bytes()?.to_vec()))
    }

    fn object_url(&self, kind: &str, key: &str) -> Result<reqwest::Url> {
        let mut url = self.bucket_url.clone();
        let mut segments = url
            .path_segments_mut()
            .map_err(|_| anyhow!("DODIL object endpoint cannot be a base URL"))?;
        segments
            .extend(kind.split('/'))
            .push(&format!("{key}.json"));
        drop(segments);
        Ok(url)
    }
}

fn checked_dodil_response(
    response: reqwest::blocking::Response,
) -> Result<reqwest::blocking::Response> {
    let status = response.status();
    if status.is_success() {
        return Ok(response);
    }
    let body = response.text().unwrap_or_default();
    bail!("DODIL DataK3 returned {status}: {body}")
}

fn record_path(root: &Path, kind: &str, key: &str) -> PathBuf {
    root.join(kind).join(format!("{key}.json"))
}

fn result_index_kind(epoch: u64, miner_hex: &str) -> String {
    format!("result-index/{epoch}/{miner_hex}")
}

fn job_index_kind(miner_hex: &str) -> String {
    format!("job-index/{miner_hex}")
}

fn require_miner(app: &App, request: &Request) -> Result<(), HttpError> {
    let provided = request
        .headers()
        .iter()
        .find(|header| header.field.equiv("X-QX-Miner-Token"))
        .map(|header| header.value.as_str());
    if provided != Some(app.miner_token.as_str()) {
        return Err(HttpError::new(
            401,
            "missing or invalid X-QX-Miner-Token header",
        ));
    }
    Ok(())
}

fn read_json<T: DeserializeOwned>(request: &mut Request) -> Result<T> {
    if request
        .body_length()
        .is_some_and(|length| length as u64 > MAX_REQUEST_BYTES)
    {
        bail!("request body exceeds {MAX_REQUEST_BYTES} bytes");
    }
    let mut body = Vec::new();
    request
        .as_reader()
        .take(MAX_REQUEST_BYTES + 1)
        .read_to_end(&mut body)?;
    if body.len() as u64 > MAX_REQUEST_BYTES {
        bail!("request body exceeds {MAX_REQUEST_BYTES} bytes");
    }
    serde_json::from_slice(&body).context("invalid JSON request")
}

fn send_json(request: Request, status: u16, value: &impl Serialize) -> Result<()> {
    let body = serde_json::to_vec_pretty(value)?;
    let content_type = Header::from_bytes("Content-Type", "application/json")
        .map_err(|_| anyhow!("invalid content-type header"))?;
    request
        .respond(
            Response::from_data(body)
                .with_status_code(StatusCode(status))
                .with_header(content_type),
        )
        .context("send HTTP response")
}

fn validate_hash_name(value: &str) -> Result<(), HttpError> {
    decode_hex_exact(value, 32, "job hash").map_err(HttpError::bad)?;
    Ok(())
}

fn decode_32(value: &str, label: &str) -> Result<[u8; 32]> {
    decode_hex_exact(value, 32, label)?
        .try_into()
        .map_err(|_| anyhow!("{label} must decode to 32 bytes"))
}

fn decode_hex_exact(value: &str, expected: usize, label: &str) -> Result<Vec<u8>> {
    if value.len() != expected * 2 || !value.is_ascii() {
        bail!("{label} must be {} lowercase hex characters", expected * 2);
    }
    let bytes = value.as_bytes();
    let mut output = Vec::with_capacity(expected);
    for pair in bytes.chunks_exact(2) {
        output.push((hex_nibble(pair[0])? << 4) | hex_nibble(pair[1])?);
    }
    Ok(output)
}

fn hex_nibble(value: u8) -> Result<u8> {
    match value {
        b'0'..=b'9' => Ok(value - b'0'),
        b'a'..=b'f' => Ok(value - b'a' + 10),
        _ => bail!("hex input must use lowercase 0-9a-f"),
    }
}

fn encode_hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        output.push(HEX[usize::from(byte >> 4)] as char);
        output.push(HEX[usize::from(byte & 15)] as char);
    }
    output
}

fn read_32(bytes: &[u8], offset: usize) -> Result<[u8; 32]> {
    bytes
        .get(offset..offset + 32)
        .and_then(|value| value.try_into().ok())
        .ok_or_else(|| anyhow!("truncated epoch account"))
}

fn read_u32(bytes: &[u8], offset: usize) -> Result<u32> {
    bytes
        .get(offset..offset + 4)
        .and_then(|value| value.try_into().ok())
        .map(u32::from_le_bytes)
        .ok_or_else(|| anyhow!("truncated epoch account"))
}

fn read_u64(bytes: &[u8], offset: usize) -> Result<u64> {
    bytes
        .get(offset..offset + 8)
        .and_then(|value| value.try_into().ok())
        .map(u64::from_le_bytes)
        .ok_or_else(|| anyhow!("truncated epoch account"))
}

fn unix_seconds() -> Result<u64> {
    Ok(SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs())
}

fn json_entries(directory: &Path) -> Result<Vec<PathBuf>> {
    let entries = match fs::read_dir(directory) {
        Ok(entries) => entries,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(error) => return Err(error.into()),
    };
    let mut paths = entries
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| {
            path.extension()
                .is_some_and(|extension| extension == "json")
        })
        .collect::<Vec<_>>();
    paths.sort();
    Ok(paths)
}

fn read_json_file<T: DeserializeOwned>(path: &Path) -> Result<T> {
    let bytes = fs::read(path).with_context(|| format!("read {}", path.display()))?;
    serde_json::from_slice(&bytes).with_context(|| format!("decode {}", path.display()))
}

fn write_record(path: &Path, value: &impl Serialize) -> Result<()> {
    write_record_bytes(path, &serde_json::to_vec_pretty(value)?)
}

fn write_record_bytes(path: &Path, bytes: &[u8]) -> Result<()> {
    let parent = path
        .parent()
        .ok_or_else(|| anyhow!("record has no parent"))?;
    fs::create_dir_all(parent)?;
    let temporary = path.with_extension("json.tmp");
    fs::write(&temporary, bytes)?;
    fs::rename(&temporary, path)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::{Signer as _, SigningKey};

    #[test]
    fn missing_local_index_is_an_empty_listing() {
        let directory = std::env::temp_dir().join(format!(
            "qx-gateway-missing-index-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        assert!(!directory.exists());
        assert!(json_entries(&directory).unwrap().is_empty());
    }

    #[test]
    fn hex_codec_is_canonical() {
        let bytes = [0, 1, 15, 16, 254, 255];
        let encoded = encode_hex(&bytes);
        assert_eq!(encoded, "00010f10feff");
        assert_eq!(
            decode_hex_exact(&encoded, bytes.len(), "test").unwrap(),
            bytes
        );
        assert!(decode_hex_exact("AA", 1, "test").is_err());
    }

    #[test]
    fn signed_job_to_selected_challenge_package_round_trip() {
        let directory = std::env::temp_dir().join(format!(
            "qx-gateway-test-{}-{}",
            std::process::id(),
            unix_seconds().unwrap()
        ));
        let program = Pubkey::new_unique();
        let miner = Pubkey::new_unique();
        let model = Pubkey::new_unique();
        let signing = SigningKey::from_bytes(&[7; 32]);
        let tokens = vec![1, 2, 3];
        let model_id = [5; 32];
        let payload = JobPayloadV1 {
            protocol_version: 1,
            protocol_instance: program.to_bytes(),
            job_id: [1; 32],
            requester: signing.verifying_key().to_bytes(),
            assigned_miner: miner.to_bytes(),
            model_id,
            execution_spec: [6; 32],
            input_hash: input_hash_v1(&model_id, &tokens).unwrap(),
            input_token_count: 3,
            max_output_tokens: 2,
            payment: 0,
            slashable_exposure: 1,
            deadline_slot: 1_000,
            nonce: 1,
            audit_probability_ppm: 1_000_000,
            assigned_epoch: 9,
            model_registry: model.to_bytes(),
        };
        let job_hash = payload.hash().unwrap();
        let job_hash_hex = encode_hex(&job_hash);
        let app = App {
            store: GatewayStore::Files(directory.clone()),
            program,
            rpc_url: "http://invalid.test".to_owned(),
            miner_token: "x".repeat(32),
        };
        app.prepare().unwrap();
        app.put_job(JobSubmission {
            payload_hex: encode_hex(&payload.canonical_bytes().unwrap()),
            input_tokens: tokens,
            requester_signature_hex: encode_hex(&signing.sign(&job_hash).to_bytes()),
        })
        .unwrap();
        let pending = app.pending_jobs(&miner.to_string()).unwrap();
        assert_eq!(pending["jobs"].as_array().unwrap().len(), 1);
        let output_tokens = vec![4, 5];
        let result = ResultLeafV1 {
            job_hash,
            output_hash: output_hash_v1(&output_tokens).unwrap(),
            trace_root: [8; 32],
            output_token_count: 2,
            trace_length: 5,
        };
        app.put_result(ResultSubmission {
            result_hex: encode_hex(&result.canonical_bytes()),
            output_tokens: output_tokens.clone(),
        })
        .unwrap();
        let stored_result = app.get_result(&job_hash_hex).unwrap();
        assert_eq!(
            stored_result["output_tokens"],
            serde_json::json!(output_tokens)
        );
        let pending = app.pending_jobs(&miner.to_string()).unwrap();
        assert!(pending["jobs"].as_array().unwrap().is_empty());
        app.close_epoch(
            9,
            CloseEpochRequest {
                miner_pubkey: miner.to_string(),
            },
        )
        .unwrap();
        let mut epoch: EpochRecord = app.store.get(EPOCHS, "9").unwrap().unwrap();
        epoch.onchain = Some(OnchainEpoch {
            epoch_account: Pubkey::new_unique().to_string(),
            close_slot: 10,
            entropy_slot: 11,
            audit_seed_hex: encode_hex(&[9; 32]),
            challenge_deadline_slot: 100,
            response_window_slots: 10,
        });
        app.store.set(EPOCHS, "9", &epoch).unwrap();
        let package = app
            .package_if_selected(&epoch, &job_hash_hex)
            .unwrap()
            .unwrap();
        assert_eq!(package.leaf_index, 0);
        assert_eq!(package.leaf_count, 1);
        assert!(package.result_proof_siblings_hex.is_empty());
        fs::remove_dir_all(directory).unwrap();
    }
}
