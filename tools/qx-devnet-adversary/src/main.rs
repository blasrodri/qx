use std::{path::PathBuf, str::FromStr, thread, time::Duration};

use anyhow::{Context, Result, anyhow, bail};
use clap::Parser;
use qx_commit::{
    CheckpointV1, D_BATCH, D_JOB, D_OPERATOR_TREE, D_PROGRAM_TREE, D_TOKEN, HashAlgorithm,
    JobPayloadV1, MerkleProof, MerkleTree, ResultLeafV1, StateTreeKindV1, TokenResultV1,
    VmInstructionV1, VmOutputStateV1, VmSegmentProgramV1, VmStateV1, job_genesis_hash_v1,
    prove_state_chunk_v1, state_hash_v1, vm_program_tree_v1, vm_segment_schedule_v1,
};
use serde::Serialize;
use solana_client::rpc_client::RpcClient;
use solana_commitment_config::CommitmentConfig;
use solana_sdk::{
    instruction::{AccountMeta, Instruction},
    pubkey::Pubkey,
    signature::{Keypair, Signature, Signer, read_keypair_file},
    transaction::Transaction,
};
use solana_system_interface::instruction::create_account;

const MODEL_BYTES: usize = 156;
const MINER_BYTES: usize = 24;
const JOB_BYTES: usize = 323;
const EPOCH_BYTES: usize = 152;
const DISPUTE_BYTES: usize = 768;
const COMPUTE_BUDGET_PROGRAM: &str = "ComputeBudget111111111111111111111111111111";

#[derive(Debug, Parser)]
#[command(
    name = "qx-devnet-adversary",
    about = "Convict a deliberately false QX result through real devnet transactions"
)]
struct Args {
    #[arg(long, default_value = "https://api.devnet.solana.com")]
    rpc_url: String,
    #[arg(long)]
    program: String,
    #[arg(long)]
    payer_keypair: PathBuf,
    #[arg(long)]
    output: Option<PathBuf>,
    #[arg(long, default_value_t = 600_000)]
    compute_unit_limit: u32,
}

#[derive(Serialize)]
struct StepEvidence {
    step: String,
    signature: String,
    simulated_compute_units: u64,
    quoted_fee_lamports: u64,
}

#[derive(Serialize)]
struct Evidence {
    status: &'static str,
    cluster: &'static str,
    program: String,
    model_account: String,
    epoch_account: String,
    dispute_account: String,
    miner_account: String,
    challenger: String,
    epoch: u64,
    dishonest_output_token: u32,
    correct_output_token: u32,
    winner: &'static str,
    transaction_count: usize,
    setup_simulated_compute_units: u64,
    dispute_simulated_compute_units: u64,
    total_simulated_compute_units: u64,
    setup_quoted_transaction_fees_lamports: u64,
    dispute_quoted_transaction_fees_lamports: u64,
    quoted_transaction_fees_lamports: u64,
    account_rent_lamports: RentEvidence,
    steps: Vec<StepEvidence>,
}

#[derive(Serialize)]
struct RentEvidence {
    model: u64,
    miner: u64,
    epoch: u64,
    dispute: u64,
    job: u64,
    total: u64,
}

struct Scenario {
    program: Pubkey,
    payer: Keypair,
    model: Keypair,
    miner: Keypair,
    epoch_account: Keypair,
    dispute: Keypair,
    epoch_number: u64,
    model_id: [u8; 32],
    execution_spec: [u8; 32],
    schedule: MerkleTree,
    segments: Vec<VmSegmentProgramV1>,
    instruction: VmInstructionV1,
    job: JobPayloadV1,
    job_account: Keypair,
    result: ResultLeafV1,
    results: MerkleTree,
    job_genesis: [u8; 32],
    common_token: TokenResultV1,
    challenger_token: TokenResultV1,
    miner_token: TokenResultV1,
    prior_final_checkpoint: CheckpointV1,
    prior_checkpoint_tree: MerkleTree,
    final_checkpoint: CheckpointV1,
    checkpoint_tree: MerkleTree,
    challenger_job_tree: MerkleTree,
    miner_job_tree: MerkleTree,
    memory_opening: qx_commit::StateChunkProofV1,
    common_kv: [u8; 32],
    pre: VmStateV1,
    challenger_output: VmOutputStateV1,
    miner_output: VmOutputStateV1,
    challenger_operator: MerkleTree,
    miner_operator: MerkleTree,
}

fn main() -> Result<()> {
    let args = Args::parse();
    if args.rpc_url != "https://api.devnet.solana.com" {
        bail!("this adversarial demo is intentionally restricted to Solana devnet");
    }
    if !(200_000..=1_400_000).contains(&args.compute_unit_limit) {
        bail!("compute-unit-limit must be 200000..=1400000");
    }
    let program = Pubkey::from_str(&args.program).context("invalid verifier program public key")?;
    let payer = read_keypair_file(&args.payer_keypair)
        .map_err(|error| anyhow!("read payer {}: {error}", args.payer_keypair.display()))?;
    let rpc = RpcClient::new_with_commitment(args.rpc_url, CommitmentConfig::confirmed());
    let program_account = rpc
        .get_account(&program)
        .with_context(|| format!("load verifier program {program}"))?;
    if !program_account.executable {
        bail!("verifier program {program} is not executable");
    }
    if rpc.get_balance(&payer.pubkey())? == 0 {
        bail!("devnet payer {} has zero balance", payer.pubkey());
    }

    let current_slot = rpc.get_slot()?;
    let scenario = Scenario::new(program, payer, current_slot)?;
    let mut steps = Vec::new();
    scenario.create_model(&rpc, args.compute_unit_limit, &mut steps)?;
    scenario.register_job(&rpc, args.compute_unit_limit, &mut steps)?;
    scenario.submit_epoch(&rpc, args.compute_unit_limit, &mut steps)?;
    scenario.finalize_epoch(&rpc, args.compute_unit_limit, &mut steps)?;
    scenario.open_dispute(&rpc, args.compute_unit_limit, &mut steps)?;
    scenario.run_bisection(&rpc, args.compute_unit_limit, &mut steps)?;
    scenario.reveal_tokens(&rpc, args.compute_unit_limit, &mut steps)?;
    scenario.reveal_final_head(&rpc, args.compute_unit_limit, &mut steps)?;
    scenario.resolve_argmax(&rpc, args.compute_unit_limit, &mut steps)?;
    scenario.settle(&rpc, args.compute_unit_limit, &mut steps)?;

    let dispute = rpc.get_account(&scenario.dispute.pubkey())?;
    if dispute.owner != program
        || dispute.data.len() != DISPUTE_BYTES
        || dispute.data.get(..4) != Some(b"QXD2")
        || dispute.data[4] != 8
        || dispute.data[5] != 2
    {
        bail!("on-chain dispute did not convict the miner");
    }
    let setup_simulated_compute_units = steps
        .iter()
        .take(4)
        .map(|step| step.simulated_compute_units)
        .sum();
    let dispute_simulated_compute_units = steps
        .iter()
        .skip(4)
        .map(|step| step.simulated_compute_units)
        .sum();
    let total_simulated_compute_units =
        setup_simulated_compute_units + dispute_simulated_compute_units;
    let setup_quoted_transaction_fees_lamports = steps
        .iter()
        .take(4)
        .map(|step| step.quoted_fee_lamports)
        .sum();
    let dispute_quoted_transaction_fees_lamports = steps
        .iter()
        .skip(4)
        .map(|step| step.quoted_fee_lamports)
        .sum();
    let quoted_transaction_fees_lamports =
        setup_quoted_transaction_fees_lamports + dispute_quoted_transaction_fees_lamports;
    let model_rent = rpc.get_minimum_balance_for_rent_exemption(MODEL_BYTES)?;
    let miner_rent = rpc.get_minimum_balance_for_rent_exemption(MINER_BYTES)?;
    let epoch_rent = rpc.get_minimum_balance_for_rent_exemption(EPOCH_BYTES)?;
    let dispute_rent = rpc.get_minimum_balance_for_rent_exemption(DISPUTE_BYTES)?;
    let job_rent = rpc.get_minimum_balance_for_rent_exemption(JOB_BYTES)?;
    let account_rent_lamports = RentEvidence {
        model: model_rent,
        miner: miner_rent,
        epoch: epoch_rent,
        dispute: dispute_rent,
        job: job_rent,
        total: model_rent + miner_rent + epoch_rent + dispute_rent + job_rent,
    };
    let evidence = Evidence {
        status: "passed",
        cluster: "devnet",
        program: program.to_string(),
        model_account: scenario.model.pubkey().to_string(),
        epoch_account: scenario.epoch_account.pubkey().to_string(),
        dispute_account: scenario.dispute.pubkey().to_string(),
        miner_account: scenario.miner.pubkey().to_string(),
        challenger: scenario.payer.pubkey().to_string(),
        epoch: scenario.epoch_number,
        dishonest_output_token: scenario.miner_token.predicted_token,
        correct_output_token: scenario.challenger_token.predicted_token,
        winner: "challenger",
        transaction_count: steps.len(),
        setup_simulated_compute_units,
        dispute_simulated_compute_units,
        total_simulated_compute_units,
        setup_quoted_transaction_fees_lamports,
        dispute_quoted_transaction_fees_lamports,
        quoted_transaction_fees_lamports,
        account_rent_lamports,
        steps,
    };
    let json = serde_json::to_string_pretty(&evidence)?;
    if let Some(path) = args.output {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(&path, format!("{json}\n"))?;
        println!("evidence={}", path.display());
    }
    println!("{json}");
    Ok(())
}

impl Scenario {
    fn new(program: Pubkey, payer: Keypair, current_slot: u64) -> Result<Self> {
        let model = Keypair::new();
        let miner = Keypair::new();
        let epoch_account = Keypair::new();
        let dispute = Keypair::new();
        let job_account = Keypair::new();
        let epoch_number = current_slot;
        let model_id = [0x21; 32];
        let execution_spec = [0x22; 32];
        let instruction = VmInstructionV1 {
            opcode: 8,
            source_a: 0,
            source_b: 8,
            source_c: 0,
            destination: 16,
        };
        let instruction_tree = vm_program_tree_v1(&[instruction])?;
        let segments = (0..4)
            .map(|segment_index| VmSegmentProgramV1 {
                segment_index,
                instruction_root: instruction_tree.root(),
                instruction_count: 1,
            })
            .collect::<Vec<_>>();
        let schedule = vm_segment_schedule_v1(&segments)?;
        let job = JobPayloadV1 {
            protocol_version: 1,
            protocol_instance: program.to_bytes(),
            job_id: [0x23; 32],
            requester: payer.pubkey().to_bytes(),
            assigned_miner: miner.pubkey().to_bytes(),
            model_id,
            execution_spec,
            input_hash: [0x24; 32],
            input_token_count: 1,
            max_output_tokens: 2,
            payment: 1_000,
            slashable_exposure: 2_000,
            deadline_slot: current_slot.saturating_add(10_000),
            nonce: current_slot,
            audit_probability_ppm: 1_000_000,
            assigned_epoch: epoch_number,
            model_registry: model.pubkey().to_bytes(),
        };

        let mut memory = vec![0_u8; 256];
        memory[..4].copy_from_slice(&7.0_f32.to_le_bytes());
        memory[4..8].copy_from_slice(&42_u32.to_le_bytes());
        memory[8..12].copy_from_slice(&7.0_f32.to_le_bytes());
        memory[12..16].copy_from_slice(&9_u32.to_le_bytes());
        memory[16..20].copy_from_slice(&u32::MAX.to_le_bytes());
        let memory_root = state_hash_v1(&memory)?;
        let memory_opening = prove_state_chunk_v1(StateTreeKindV1::Residual, &memory, 0)?;
        let common_kv = [0x72; 32];
        let genesis_checkpoint = CheckpointV1 {
            job_id: job.job_id,
            token_index: 1,
            layer_index: u16::MAX,
            residual_hash: [0x30; 32],
            global_kv_root: [0x31; 32],
        };
        let middle_checkpoint = CheckpointV1 {
            layer_index: 1,
            residual_hash: [0x32; 32],
            global_kv_root: [0x33; 32],
            ..genesis_checkpoint
        };
        let final_checkpoint = CheckpointV1 {
            layer_index: 3,
            residual_hash: memory_root,
            global_kv_root: common_kv,
            ..genesis_checkpoint
        };
        let checkpoint_tree = MerkleTree::new(
            HashAlgorithm::Blake3V1,
            D_TOKEN,
            &[
                genesis_checkpoint.hash()?,
                middle_checkpoint.hash()?,
                final_checkpoint.hash()?,
            ],
        )?;
        let prior_final_checkpoint = CheckpointV1 {
            job_id: job.job_id,
            token_index: 0,
            layer_index: 3,
            residual_hash: [0x28; 32],
            global_kv_root: genesis_checkpoint.global_kv_root,
        };
        let prior_checkpoint_tree = MerkleTree::new(
            HashAlgorithm::Blake3V1,
            D_TOKEN,
            &[
                CheckpointV1 {
                    layer_index: u16::MAX,
                    ..prior_final_checkpoint
                }
                .hash()?,
                CheckpointV1 {
                    layer_index: 1,
                    ..prior_final_checkpoint
                }
                .hash()?,
                prior_final_checkpoint.hash()?,
            ],
        )?;
        let common_token = TokenResultV1 {
            job_id: job.job_id,
            token_index: 0,
            checkpoint_root: prior_checkpoint_tree.root(),
            predicted_token: 41,
            checkpoint_count: 3,
        };
        let challenger_token = TokenResultV1 {
            job_id: job.job_id,
            token_index: 1,
            checkpoint_root: checkpoint_tree.root(),
            predicted_token: 9,
            checkpoint_count: 3,
        };
        let miner_token = TokenResultV1 {
            predicted_token: 42,
            ..challenger_token
        };
        let job_genesis = job_genesis_hash_v1(&job.job_id)?;
        let common_token_hash = common_token.hash()?;
        let challenger_job_tree = MerkleTree::new(
            HashAlgorithm::Blake3V1,
            D_JOB,
            &[job_genesis, common_token_hash, challenger_token.hash()?],
        )?;
        let miner_job_tree = MerkleTree::new(
            HashAlgorithm::Blake3V1,
            D_JOB,
            &[job_genesis, common_token_hash, miner_token.hash()?],
        )?;
        let result = ResultLeafV1 {
            job_hash: job.hash()?,
            output_hash: [0x25; 32],
            trace_root: miner_job_tree.root(),
            output_token_count: 2,
            trace_length: 3,
        };
        let results = MerkleTree::new(HashAlgorithm::Sha256V1, D_BATCH, &[result.hash()?])?;
        let pre = VmStateV1 {
            job_hash: job.hash()?,
            token_index: 1,
            segment_index: 3,
            program_counter: 0,
            residual_root: memory_root,
            global_kv_root: common_kv,
        };
        let mut challenger_memory = memory.clone();
        challenger_memory[16..20].copy_from_slice(&7.0_f32.to_bits().to_le_bytes());
        challenger_memory[20..24].copy_from_slice(&9_u32.to_le_bytes());
        let challenger_output = VmOutputStateV1 {
            job_hash: pre.job_hash,
            token_index: pre.token_index,
            segment_index: pre.segment_index,
            program_counter: 1,
            residual_root: state_hash_v1(&challenger_memory)?,
            global_kv_root: common_kv,
            predicted_token: 9,
        };
        let mut miner_memory = memory.clone();
        miner_memory[16..20].copy_from_slice(&7.0_f32.to_bits().to_le_bytes());
        miner_memory[20..24].copy_from_slice(&42_u32.to_le_bytes());
        let miner_output = VmOutputStateV1 {
            residual_root: state_hash_v1(&miner_memory)?,
            predicted_token: 42,
            ..challenger_output
        };
        let challenger_operator = MerkleTree::new(
            HashAlgorithm::Blake3V1,
            D_OPERATOR_TREE,
            &[pre.hash()?, challenger_output.hash()?],
        )?;
        let miner_operator = MerkleTree::new(
            HashAlgorithm::Blake3V1,
            D_OPERATOR_TREE,
            &[pre.hash()?, miner_output.hash()?],
        )?;
        Ok(Self {
            program,
            payer,
            model,
            miner,
            epoch_account,
            dispute,
            epoch_number,
            model_id,
            execution_spec,
            schedule,
            segments,
            instruction,
            job,
            job_account,
            result,
            results,
            job_genesis,
            common_token,
            challenger_token,
            miner_token,
            prior_final_checkpoint,
            prior_checkpoint_tree,
            final_checkpoint,
            checkpoint_tree,
            challenger_job_tree,
            miner_job_tree,
            memory_opening,
            common_kv,
            pre,
            challenger_output,
            miner_output,
            challenger_operator,
            miner_operator,
        })
    }

    fn ix(&self, opcode: u8, accounts: Vec<AccountMeta>, body: &[u8]) -> Instruction {
        let mut data = Vec::with_capacity(1 + body.len());
        data.push(opcode);
        data.extend_from_slice(body);
        Instruction::new_with_bytes(self.program, &data, accounts)
    }

    fn create_model(&self, rpc: &RpcClient, cu: u32, steps: &mut Vec<StepEvidence>) -> Result<()> {
        let mut body = Vec::new();
        body.extend_from_slice(&self.model_id);
        body.extend_from_slice(&[0x26; 32]);
        body.extend_from_slice(&1_u32.to_le_bytes());
        body.extend_from_slice(&self.execution_spec);
        body.extend_from_slice(&self.schedule.root());
        body.extend_from_slice(&(self.segments.len() as u32).to_le_bytes());
        body.extend_from_slice(&3_u32.to_le_bytes());
        body.extend_from_slice(&4_u16.to_le_bytes());
        body.extend_from_slice(&2_u16.to_le_bytes());
        body.extend_from_slice(&4_096_u32.to_le_bytes());
        body.extend_from_slice(&49_155_u32.to_le_bytes());
        let instructions = vec![
            create_account(
                &self.payer.pubkey(),
                &self.model.pubkey(),
                rpc.get_minimum_balance_for_rent_exemption(MODEL_BYTES)?,
                MODEL_BYTES as u64,
                &self.program,
            ),
            self.ix(6, vec![AccountMeta::new(self.model.pubkey(), true)], &body),
        ];
        send_step(
            rpc,
            &self.payer,
            "register_model",
            instructions,
            &[&self.model],
            cu,
            steps,
        )
    }

    fn register_job(&self, rpc: &RpcClient, cu: u32, steps: &mut Vec<StepEvidence>) -> Result<()> {
        let job_rent = rpc.get_minimum_balance_for_rent_exemption(JOB_BYTES)?;
        let miner_rent = rpc.get_minimum_balance_for_rent_exemption(MINER_BYTES)?;
        let instructions = vec![
            create_account(
                &self.payer.pubkey(),
                &self.miner.pubkey(),
                miner_rent + self.job.slashable_exposure,
                MINER_BYTES as u64,
                &self.program,
            ),
            create_account(
                &self.payer.pubkey(),
                &self.job_account.pubkey(),
                job_rent + self.job.payment,
                JOB_BYTES as u64,
                &self.program,
            ),
            self.ix(
                8,
                vec![
                    AccountMeta::new(self.job_account.pubkey(), true),
                    AccountMeta::new_readonly(self.payer.pubkey(), true),
                    AccountMeta::new_readonly(self.model.pubkey(), false),
                    AccountMeta::new(self.miner.pubkey(), false),
                ],
                &self.job.canonical_bytes()?,
            ),
        ];
        send_step(
            rpc,
            &self.payer,
            "register_and_fund_job",
            instructions,
            &[&self.miner, &self.job_account],
            cu,
            steps,
        )
    }

    fn submit_epoch(&self, rpc: &RpcClient, cu: u32, steps: &mut Vec<StepEvidence>) -> Result<()> {
        let mut body = Vec::new();
        body.extend_from_slice(&self.epoch_number.to_le_bytes());
        body.extend_from_slice(&self.results.root());
        body.extend_from_slice(&1_u32.to_le_bytes());
        let instructions = vec![
            create_account(
                &self.payer.pubkey(),
                &self.epoch_account.pubkey(),
                rpc.get_minimum_balance_for_rent_exemption(EPOCH_BYTES)?,
                EPOCH_BYTES as u64,
                &self.program,
            ),
            self.ix(
                9,
                vec![
                    AccountMeta::new(self.epoch_account.pubkey(), true),
                    AccountMeta::new(self.miner.pubkey(), true),
                ],
                &body,
            ),
        ];
        send_step(
            rpc,
            &self.payer,
            "submit_dishonest_epoch",
            instructions,
            &[&self.miner, &self.epoch_account],
            cu,
            steps,
        )
    }

    fn finalize_epoch(
        &self,
        rpc: &RpcClient,
        cu: u32,
        steps: &mut Vec<StepEvidence>,
    ) -> Result<()> {
        let instruction = self.ix(
            10,
            vec![AccountMeta::new(self.epoch_account.pubkey(), false)],
            &[],
        );
        let mut last_error = None;
        for _ in 0..45 {
            match send_step(
                rpc,
                &self.payer,
                "finalize_entropy",
                vec![instruction.clone()],
                &[],
                cu,
                steps,
            ) {
                Ok(()) => return Ok(()),
                Err(error) => last_error = Some(error),
            }
            thread::sleep(Duration::from_secs(1));
        }
        Err(last_error.unwrap_or_else(|| anyhow!("epoch finalization timed out")))
    }

    fn open_dispute(&self, rpc: &RpcClient, cu: u32, steps: &mut Vec<StepEvidence>) -> Result<()> {
        let job_hash = self.job.hash()?;
        let signature = self.payer.sign_message(&job_hash);
        let signature_bytes: [u8; 64] = signature.as_ref().try_into().expect("signature size");
        let verify = solana_ed25519_program::new_ed25519_instruction_with_signature(
            &job_hash,
            &signature_bytes,
            self.payer.pubkey().as_array(),
        );
        let proof = self.results.proof(0)?;
        let mut body = self.job.canonical_bytes()?;
        body.extend_from_slice(&self.result.canonical_bytes());
        append_proof(&mut body, &proof);
        body.extend_from_slice(&self.challenger_job_tree.root());
        let open = self.ix(
            11,
            vec![
                AccountMeta::new_readonly(self.epoch_account.pubkey(), false),
                AccountMeta::new_readonly(self.model.pubkey(), false),
                AccountMeta::new(self.job_account.pubkey(), false),
                AccountMeta::new(self.dispute.pubkey(), true),
                AccountMeta::new_readonly(self.payer.pubkey(), true),
                AccountMeta::new_readonly(solana_sdk_ids::sysvar::instructions::ID, false),
            ],
            &body,
        );
        let instructions = vec![
            create_account(
                &self.payer.pubkey(),
                &self.dispute.pubkey(),
                rpc.get_minimum_balance_for_rent_exemption(DISPUTE_BYTES)?
                    + self.job.slashable_exposure,
                DISPUTE_BYTES as u64,
                &self.program,
            ),
            verify,
            open,
        ];
        send_step(
            rpc,
            &self.payer,
            "open_challenge",
            instructions,
            &[&self.dispute],
            cu,
            steps,
        )
    }

    fn settle(&self, rpc: &RpcClient, cu: u32, steps: &mut Vec<StepEvidence>) -> Result<()> {
        let instruction = self.ix(
            61,
            vec![
                AccountMeta::new(self.job_account.pubkey(), false),
                AccountMeta::new(self.miner.pubkey(), false),
                AccountMeta::new(self.dispute.pubkey(), false),
                AccountMeta::new(self.payer.pubkey(), false),
                AccountMeta::new(self.payer.pubkey(), false),
            ],
            &[],
        );
        send_step(
            rpc,
            &self.payer,
            "settle_challenger_win",
            vec![instruction],
            &[],
            cu,
            steps,
        )?;
        let job = rpc.get_account(&self.job_account.pubkey())?;
        let miner = rpc.get_account(&self.miner.pubkey())?;
        if job.data.get(314) != Some(&3) || miner.data.get(16..24) != Some(&0_u64.to_le_bytes()) {
            bail!("economic settlement did not release the job and collateral lock");
        }
        Ok(())
    }

    fn run_bisection(&self, rpc: &RpcClient, cu: u32, steps: &mut Vec<StepEvidence>) -> Result<()> {
        for (step, opcode, signer, tree, end) in [
            (
                "challenger_job_bounds",
                12,
                &self.payer,
                &self.challenger_job_tree,
                self.challenger_token.hash()?,
            ),
            (
                "miner_job_bounds",
                13,
                &self.miner,
                &self.miner_job_tree,
                self.miner_token.hash()?,
            ),
        ] {
            let body = bounds_body(tree, self.job_genesis, end)?;
            self.send_party(rpc, step, opcode, signer, body, cu, steps)?;
        }
        let common = self.common_token.hash()?;
        for (step, opcode, signer, tree) in [
            (
                "challenger_job_midpoint",
                14,
                &self.payer,
                &self.challenger_job_tree,
            ),
            ("miner_job_midpoint", 15, &self.miner, &self.miner_job_tree),
        ] {
            let mut body = common.to_vec();
            let proof = tree.proof(1)?;
            body.push(proof.siblings.len() as u8);
            append_siblings(&mut body, &proof);
            self.send_party(rpc, step, opcode, signer, body, cu, steps)?;
        }
        Ok(())
    }

    fn reveal_tokens(&self, rpc: &RpcClient, cu: u32, steps: &mut Vec<StepEvidence>) -> Result<()> {
        let mut challenger = self.challenger_token.canonical_bytes().to_vec();
        challenger.extend_from_slice(&self.common_token.predicted_token.to_le_bytes());
        challenger.extend_from_slice(&self.common_token.canonical_bytes());
        challenger.extend_from_slice(&self.prior_final_checkpoint.canonical_bytes());
        let proof = self.prior_checkpoint_tree.proof(2)?;
        challenger.push(proof.siblings.len() as u8);
        append_siblings(&mut challenger, &proof);
        let ix = self.ix(
            25,
            vec![
                AccountMeta::new_readonly(self.model.pubkey(), false),
                AccountMeta::new(self.dispute.pubkey(), false),
                AccountMeta::new_readonly(self.payer.pubkey(), true),
            ],
            &challenger,
        );
        send_step(
            rpc,
            &self.payer,
            "challenger_token",
            vec![ix],
            &[],
            cu,
            steps,
        )?;

        let ix = self.ix(
            26,
            vec![
                AccountMeta::new(self.dispute.pubkey(), false),
                AccountMeta::new_readonly(self.miner.pubkey(), true),
            ],
            &self.miner_token.canonical_bytes(),
        );
        send_step(
            rpc,
            &self.payer,
            "miner_token",
            vec![ix],
            &[&self.miner],
            cu,
            steps,
        )
    }

    fn reveal_final_head(
        &self,
        rpc: &RpcClient,
        cu: u32,
        steps: &mut Vec<StepEvidence>,
    ) -> Result<()> {
        let mut challenger = self.final_checkpoint.canonical_bytes().to_vec();
        let checkpoint_proof = self.checkpoint_tree.proof(2)?;
        challenger.push(checkpoint_proof.siblings.len() as u8);
        append_siblings(&mut challenger, &checkpoint_proof);
        self.append_segment(&mut challenger, 3)?;
        challenger.extend_from_slice(&self.challenger_output.residual_root);
        challenger.extend_from_slice(&self.common_kv);
        challenger.extend_from_slice(&operator_bounds_body(
            &self.challenger_operator,
            self.pre.hash()?,
            self.challenger_output.hash()?,
        )?);
        let ix = self.ix(
            36,
            vec![
                AccountMeta::new_readonly(self.model.pubkey(), false),
                AccountMeta::new(self.dispute.pubkey(), false),
                AccountMeta::new_readonly(self.payer.pubkey(), true),
            ],
            &challenger,
        );
        send_step(
            rpc,
            &self.payer,
            "challenger_final_head",
            vec![ix],
            &[],
            cu,
            steps,
        )?;

        let mut miner = Vec::new();
        miner.extend_from_slice(&self.miner_output.residual_root);
        miner.extend_from_slice(&self.common_kv);
        miner.extend_from_slice(&operator_bounds_body(
            &self.miner_operator,
            self.pre.hash()?,
            self.miner_output.hash()?,
        )?);
        let ix = self.ix(
            37,
            vec![
                AccountMeta::new_readonly(self.model.pubkey(), false),
                AccountMeta::new(self.dispute.pubkey(), false),
                AccountMeta::new_readonly(self.miner.pubkey(), true),
            ],
            &miner,
        );
        send_step(
            rpc,
            &self.payer,
            "miner_final_head",
            vec![ix],
            &[&self.miner],
            cu,
            steps,
        )
    }

    fn resolve_argmax(
        &self,
        rpc: &RpcClient,
        cu: u32,
        steps: &mut Vec<StepEvidence>,
    ) -> Result<()> {
        let instruction_tree = MerkleTree::new(
            HashAlgorithm::Blake3V1,
            D_PROGRAM_TREE,
            &[self.instruction.hash(0)?],
        )?;
        let instruction_proof = instruction_tree.proof(0)?;
        let mut body = Vec::new();
        body.extend_from_slice(&self.pre.segment_index.to_le_bytes());
        body.extend_from_slice(&self.pre.program_counter.to_le_bytes());
        body.extend_from_slice(&self.pre.residual_root);
        body.extend_from_slice(&self.pre.global_kv_root);
        body.extend_from_slice(&self.instruction.canonical_bytes());
        body.extend_from_slice(&1_u32.to_le_bytes());
        body.push(instruction_proof.siblings.len() as u8);
        append_siblings(&mut body, &instruction_proof);
        body.extend_from_slice(&self.memory_opening.byte_length.to_le_bytes());
        body.extend_from_slice(&self.memory_opening.chunk_index.to_le_bytes());
        body.extend_from_slice(&(self.memory_opening.chunk.len() as u16).to_le_bytes());
        body.extend_from_slice(&self.memory_opening.chunk);
        body.push(self.memory_opening.siblings.len() as u8);
        for sibling in &self.memory_opening.siblings {
            body.extend_from_slice(sibling);
        }
        let ix = self.ix(
            35,
            vec![
                AccountMeta::new_readonly(self.model.pubkey(), false),
                AccountMeta::new(self.dispute.pubkey(), false),
            ],
            &body,
        );
        send_step(rpc, &self.payer, "resolve_argmax", vec![ix], &[], cu, steps)
    }

    fn append_segment(&self, body: &mut Vec<u8>, index: usize) -> Result<()> {
        let proof = self.schedule.proof(index)?;
        body.extend_from_slice(&self.segments[index].canonical_bytes());
        body.push(proof.siblings.len() as u8);
        append_siblings(body, &proof);
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn send_party(
        &self,
        rpc: &RpcClient,
        step: &str,
        opcode: u8,
        signer: &Keypair,
        body: Vec<u8>,
        cu: u32,
        steps: &mut Vec<StepEvidence>,
    ) -> Result<()> {
        let ix = self.ix(
            opcode,
            vec![
                AccountMeta::new(self.dispute.pubkey(), false),
                AccountMeta::new_readonly(signer.pubkey(), true),
            ],
            &body,
        );
        if signer.pubkey() == self.payer.pubkey() {
            send_step(rpc, &self.payer, step, vec![ix], &[], cu, steps)
        } else {
            send_step(rpc, &self.payer, step, vec![ix], &[signer], cu, steps)
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn send_step(
    rpc: &RpcClient,
    payer: &Keypair,
    step: &str,
    mut instructions: Vec<Instruction>,
    additional_signers: &[&Keypair],
    compute_unit_limit: u32,
    evidence: &mut Vec<StepEvidence>,
) -> Result<()> {
    let mut budget_data = vec![2];
    budget_data.extend_from_slice(&compute_unit_limit.to_le_bytes());
    instructions.insert(
        0,
        Instruction::new_with_bytes(
            Pubkey::from_str(COMPUTE_BUDGET_PROGRAM).expect("valid compute budget program"),
            &budget_data,
            vec![],
        ),
    );
    let blockhash = rpc.get_latest_blockhash()?;
    let mut signers: Vec<&dyn Signer> = vec![payer];
    signers.extend(
        additional_signers
            .iter()
            .map(|signer| *signer as &dyn Signer),
    );
    let transaction = Transaction::new_signed_with_payer(
        &instructions,
        Some(&payer.pubkey()),
        &signers,
        blockhash,
    );
    let quoted_fee_lamports = rpc.get_fee_for_message(&transaction.message)?;
    let simulation = rpc.simulate_transaction(&transaction)?;
    if let Some(error) = simulation.value.err {
        bail!(
            "{step} simulation failed: {error:?}; logs={:?}",
            simulation.value.logs
        );
    }
    let units = simulation.value.units_consumed.unwrap_or_default();
    let signature: Signature = rpc
        .send_and_confirm_transaction(&transaction)
        .with_context(|| format!("send {step}"))?;
    println!(
        "event=devnet_adversary_step step={step} signature={signature} simulated_compute_units={units} quoted_fee_lamports={quoted_fee_lamports}"
    );
    evidence.push(StepEvidence {
        step: step.to_owned(),
        signature: signature.to_string(),
        simulated_compute_units: units,
        quoted_fee_lamports,
    });
    Ok(())
}

fn bounds_body(tree: &MerkleTree, start: [u8; 32], end: [u8; 32]) -> Result<Vec<u8>> {
    let start_proof = tree.proof(0)?;
    let end_proof = tree.proof(2)?;
    let mut body = Vec::new();
    body.extend_from_slice(&start);
    body.extend_from_slice(&end);
    body.push(start_proof.siblings.len() as u8);
    append_siblings(&mut body, &start_proof);
    append_siblings(&mut body, &end_proof);
    Ok(body)
}

fn operator_bounds_body(tree: &MerkleTree, start: [u8; 32], end: [u8; 32]) -> Result<Vec<u8>> {
    let start_proof = tree.proof(0)?;
    let end_proof = tree.proof(1)?;
    let mut body = Vec::new();
    body.extend_from_slice(&tree.root());
    body.extend_from_slice(&start);
    body.extend_from_slice(&end);
    body.push(start_proof.siblings.len() as u8);
    append_siblings(&mut body, &start_proof);
    append_siblings(&mut body, &end_proof);
    Ok(body)
}

fn append_proof(body: &mut Vec<u8>, proof: &MerkleProof) {
    body.extend_from_slice(&proof.leaf_index.to_le_bytes());
    body.extend_from_slice(&proof.leaf_count.to_le_bytes());
    body.push(proof.siblings.len() as u8);
    append_siblings(body, proof);
}

fn append_siblings(body: &mut Vec<u8>, proof: &MerkleProof) {
    for sibling in &proof.siblings {
        body.extend_from_slice(sibling);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adversarial_fixture_binds_a_wrong_final_token() {
        let scenario = Scenario::new(Pubkey::new_unique(), Keypair::new(), 100).unwrap();
        assert_eq!(scenario.result.trace_root, scenario.miner_job_tree.root());
        assert_ne!(
            scenario.challenger_job_tree.root(),
            scenario.miner_job_tree.root()
        );
        assert_eq!(scenario.result.trace_length, 3);
        assert_eq!(scenario.job.input_token_count, 1);
        assert_eq!(scenario.result.output_token_count, 2);
        assert_eq!(
            scenario.challenger_token.checkpoint_root,
            scenario.miner_token.checkpoint_root
        );
        assert_ne!(
            scenario.challenger_token.predicted_token,
            scenario.miner_token.predicted_token
        );
    }
}
