use std::path::Path;

use qx_format::TensorKind;
use qx_quant::QuantType;

use crate::{
    CommitError, D_MACRO_STATE, D_MACRO_WORKSPACE, D_MACRO_WORKSPACE_SLOT, D_OPERATOR_TREE, Digest,
    HashAlgorithm, MerkleTree, SemanticTensorV1, VmInstructionV1, VmOutputStateV1,
    VmSegmentProgramV1, VmStateV1, artifact_semantic_tensors_v1, hash_domain, vm_program_tree_v1,
    vm_segment_schedule_v1,
};

/// Each Granite layer has seven shared tensors followed by 32 gate/up and 32
/// down tensors in the canonical semantic weight order.
pub const GRANITE_TENSORS_PER_LAYER_V1: u32 = 71;

/// Coarse instructions used for the first, cheap dispute descent. A coarse
/// instruction is never evaluated atomically on chain: after it is isolated,
/// its deterministic scalar trace is bisected and resolved by the existing
/// scalar/Q4/raw/KV terminals.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum GraniteMacroOpcodeV1 {
    Reserved = 64,
    RmsNorm = 65,
    Dense = 66,
    Rope = 67,
    KvAppend = 68,
    Attention = 69,
    Residual = 70,
    TopK = 71,
    Moe = 72,
    FinalLogits = 73,
    ArgMax = 74,
    Embedding = 75,
}

impl TryFrom<u8> for GraniteMacroOpcodeV1 {
    type Error = CommitError;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            64 => Ok(Self::Reserved),
            65 => Ok(Self::RmsNorm),
            66 => Ok(Self::Dense),
            67 => Ok(Self::Rope),
            68 => Ok(Self::KvAppend),
            69 => Ok(Self::Attention),
            70 => Ok(Self::Residual),
            71 => Ok(Self::TopK),
            72 => Ok(Self::Moe),
            73 => Ok(Self::FinalLogits),
            74 => Ok(Self::ArgMax),
            75 => Ok(Self::Embedding),
            _ => Err(CommitError::InvalidExecution(
                "unknown Granite macro opcode".to_owned(),
            )),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u32)]
pub enum GraniteDenseOutputV1 {
    Query = 1,
    Key = 2,
    Value = 3,
    Attention = 4,
    Router = 5,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GraniteMacroOutputV1 {
    pub opcode: GraniteMacroOpcodeV1,
    pub output_root: Digest,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum GraniteWorkspaceSlotV1 {
    Residual = 0,
    Normalized = 1,
    Query = 2,
    Key = 3,
    Value = 4,
    RotatedQueryKey = 5,
    KvLeaf = 6,
    Attention = 7,
    AttentionProjection = 8,
    PostAttention = 9,
    MoeInput = 10,
    RouterLogits = 11,
    Routes = 12,
    Moe = 13,
    Logits = 14,
    Winner = 15,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GraniteMacroWorkspaceV1 {
    slots: [Digest; 16],
}

impl GraniteMacroWorkspaceV1 {
    pub fn new(residual_root: Digest) -> Self {
        let mut slots = [[0; 32]; 16];
        slots[GraniteWorkspaceSlotV1::Residual as usize] = residual_root;
        Self { slots }
    }

    pub fn set(&mut self, slot: GraniteWorkspaceSlotV1, root: Digest) {
        self.slots[slot as usize] = root;
    }

    pub fn get(&self, slot: GraniteWorkspaceSlotV1) -> Digest {
        self.slots[slot as usize]
    }

    pub fn slot_leaf_hash(
        slot: GraniteWorkspaceSlotV1,
        value_root: &Digest,
    ) -> Result<Digest, CommitError> {
        hash_domain(
            HashAlgorithm::Blake3V1,
            D_MACRO_WORKSPACE_SLOT,
            &[&(slot as u32).to_le_bytes(), value_root],
        )
    }

    /// Opening used only after a dispute has isolated one macro. Normal
    /// inference retains neither these proofs nor the underlying activations.
    pub fn proof(&self, slot: GraniteWorkspaceSlotV1) -> Result<crate::MerkleProof, CommitError> {
        self.tree()?.proof(slot as usize)
    }

    fn tree(&self) -> Result<MerkleTree, CommitError> {
        let leaves = self
            .slots
            .iter()
            .enumerate()
            .map(|(index, root)| {
                let slot = GraniteWorkspaceSlotV1::try_from(index as u8)?;
                Self::slot_leaf_hash(slot, root)
            })
            .collect::<Result<Vec<_>, _>>()?;
        MerkleTree::new(HashAlgorithm::Blake3V1, D_MACRO_WORKSPACE, &leaves)
    }

    pub fn root(&self) -> Result<Digest, CommitError> {
        Ok(self.tree()?.root())
    }
}

impl TryFrom<u8> for GraniteWorkspaceSlotV1 {
    type Error = CommitError;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::Residual),
            1 => Ok(Self::Normalized),
            2 => Ok(Self::Query),
            3 => Ok(Self::Key),
            4 => Ok(Self::Value),
            5 => Ok(Self::RotatedQueryKey),
            6 => Ok(Self::KvLeaf),
            7 => Ok(Self::Attention),
            8 => Ok(Self::AttentionProjection),
            9 => Ok(Self::PostAttention),
            10 => Ok(Self::MoeInput),
            11 => Ok(Self::RouterLogits),
            12 => Ok(Self::Routes),
            13 => Ok(Self::Moe),
            14 => Ok(Self::Logits),
            15 => Ok(Self::Winner),
            _ => Err(CommitError::InvalidIndex {
                index: usize::from(value),
                length: 16,
            }),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GraniteMacroStateV1 {
    pub job_hash: Digest,
    pub token_index: u32,
    pub segment_index: u32,
    pub program_counter: u32,
    pub workspace_root: Digest,
    pub global_kv_root: Digest,
}

impl GraniteMacroStateV1 {
    pub fn hash(&self) -> Result<Digest, CommitError> {
        hash_domain(
            HashAlgorithm::Blake3V1,
            D_MACRO_STATE,
            &[
                &self.job_hash,
                &self.token_index.to_le_bytes(),
                &self.segment_index.to_le_bytes(),
                &self.program_counter.to_le_bytes(),
                &self.workspace_root,
                &self.global_kv_root,
            ],
        )
    }
}

#[derive(Clone, Debug)]
pub struct GraniteMacroTraceV1 {
    pub state_hashes: Vec<Digest>,
    pub tree: MerkleTree,
    pub final_residual_root: Digest,
    pub final_global_kv_root: Digest,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GraniteLayerTensorIdsV1 {
    pub input_norm: u32,
    pub query: u32,
    pub key: u32,
    pub value: u32,
    pub attention_output: u32,
    pub post_attention_norm: u32,
    pub router: u32,
    pub gate_up_first: u32,
    pub down_first: u32,
}

pub fn granite_layer_tensor_ids_v1(layer: u16) -> Result<GraniteLayerTensorIdsV1, CommitError> {
    let shape = qx_model::ModelShape::granite_v0();
    if usize::from(layer) >= shape.layers {
        return Err(CommitError::InvalidIndex {
            index: usize::from(layer),
            length: shape.layers,
        });
    }
    let base = u32::from(layer)
        .checked_mul(GRANITE_TENSORS_PER_LAYER_V1)
        .and_then(|value| value.checked_add(1))
        .ok_or(CommitError::ArithmeticOverflow)?;
    Ok(GraniteLayerTensorIdsV1 {
        input_norm: base,
        query: base + 1,
        key: base + 2,
        value: base + 3,
        attention_output: base + 4,
        post_attention_norm: base + 5,
        router: base + 6,
        gate_up_first: base + 7,
        down_first: base + 39,
    })
}

pub fn granite_embedding_tensor_id_v1() -> u32 {
    0
}

pub fn granite_final_norm_tensor_id_v1() -> Result<u32, CommitError> {
    u32::try_from(qx_model::LAYERS)
        .map_err(|_| CommitError::ArithmeticOverflow)?
        .checked_mul(GRANITE_TENSORS_PER_LAYER_V1)
        .and_then(|value| value.checked_add(1))
        .ok_or(CommitError::ArithmeticOverflow)
}

/// Check that the model artifact consumed by the runtime has exactly the
/// semantic tensors referenced by the deterministic macro lowering.
pub fn validate_granite_macro_artifact_v1(path: impl AsRef<Path>) -> Result<(), CommitError> {
    let tensors = artifact_semantic_tensors_v1(path)?;
    let shape = qx_model::ModelShape::granite_v0();
    let expected_count = usize::try_from(granite_final_norm_tensor_id_v1()?)
        .map_err(|_| CommitError::ArithmeticOverflow)?
        .checked_add(1)
        .ok_or(CommitError::ArithmeticOverflow)?;
    if tensors.len() != expected_count {
        return Err(CommitError::InvalidArtifact(format!(
            "Granite macro program expects {expected_count} semantic tensors, found {}",
            tensors.len()
        )));
    }
    let expected = |id: u32,
                    layer: u16,
                    expert: u8,
                    kind: TensorKind,
                    rows: usize,
                    cols: usize,
                    raw: bool|
     -> Result<(), CommitError> {
        let tensor = tensors.get(id as usize).ok_or(CommitError::InvalidIndex {
            index: id as usize,
            length: tensors.len(),
        })?;
        let q4 = tensor.quant_type == QuantType::Q4 && tensor.group_size == 32;
        if *tensor
            != (SemanticTensorV1 {
                tensor_id: id,
                layer,
                expert,
                kind,
                rows: u32::try_from(rows).map_err(|_| CommitError::ArithmeticOverflow)?,
                cols: u32::try_from(cols).map_err(|_| CommitError::ArithmeticOverflow)?,
                group_size: if raw { 0 } else { 32 },
                quant_type: tensor.quant_type,
            })
            || (!raw && !q4)
        {
            return Err(CommitError::InvalidArtifact(format!(
                "semantic tensor {id} does not match the Granite macro program: {tensor:?}"
            )));
        }
        Ok(())
    };

    expected(
        granite_embedding_tensor_id_v1(),
        u16::MAX,
        u8::MAX,
        TensorKind::Embedding,
        shape.vocabulary,
        shape.hidden_size,
        false,
    )?;
    for layer in 0..shape.layers {
        let layer_u16 = u16::try_from(layer).map_err(|_| CommitError::ArithmeticOverflow)?;
        let ids = granite_layer_tensor_ids_v1(layer_u16)?;
        for (id, kind, rows, cols, raw) in [
            (
                ids.input_norm,
                TensorKind::InputNorm,
                shape.hidden_size,
                1,
                true,
            ),
            (
                ids.query,
                TensorKind::QProj,
                shape.hidden_size,
                shape.hidden_size,
                false,
            ),
            (
                ids.key,
                TensorKind::KProj,
                shape.key_value_heads * shape.head_dim,
                shape.hidden_size,
                false,
            ),
            (
                ids.value,
                TensorKind::VProj,
                shape.key_value_heads * shape.head_dim,
                shape.hidden_size,
                false,
            ),
            (
                ids.attention_output,
                TensorKind::OProj,
                shape.hidden_size,
                shape.hidden_size,
                false,
            ),
            (
                ids.post_attention_norm,
                TensorKind::PostAttentionNorm,
                shape.hidden_size,
                1,
                true,
            ),
            (
                ids.router,
                TensorKind::Router,
                shape.experts,
                shape.hidden_size,
                false,
            ),
        ] {
            expected(id, layer_u16, u8::MAX, kind, rows, cols, raw)?;
        }
        for expert in 0..shape.experts {
            let expert_u8 = u8::try_from(expert).map_err(|_| CommitError::ArithmeticOverflow)?;
            expected(
                ids.gate_up_first + expert as u32,
                layer_u16,
                expert_u8,
                TensorKind::GateUp,
                shape.hidden_size,
                shape.hidden_size,
                false,
            )?;
            expected(
                ids.down_first + expert as u32,
                layer_u16,
                expert_u8,
                TensorKind::Down,
                shape.hidden_size,
                shape.expert_intermediate,
                false,
            )?;
        }
    }
    expected(
        granite_final_norm_tensor_id_v1()?,
        u16::MAX,
        u8::MAX,
        TensorKind::FinalNorm,
        shape.hidden_size,
        1,
        true,
    )
}

fn macro_instruction(
    opcode: GraniteMacroOpcodeV1,
    source_a: u32,
    source_b: u32,
    source_c: u32,
    destination: u32,
) -> VmInstructionV1 {
    VmInstructionV1 {
        opcode: opcode as u8,
        source_a,
        source_b,
        source_c,
        destination,
    }
}

/// Exact macro schedule for one Granite decoder layer. The fields bind every
/// static tensor reference, model dimension, constant, and output role. The
/// selected experts remain runtime data; `Moe` derives their tensor IDs from
/// authenticated Top-K output and the two committed tensor-ID bases.
pub fn granite_layer_macro_program_v1(layer: u16) -> Result<Vec<VmInstructionV1>, CommitError> {
    let shape = qx_model::ModelShape::granite_v0();
    let ids = granite_layer_tensor_ids_v1(layer)?;
    let layer = u32::from(layer);
    let hidden = u32::try_from(shape.hidden_size).map_err(|_| CommitError::ArithmeticOverflow)?;
    let kv_width = u32::try_from(shape.key_value_heads * shape.head_dim)
        .map_err(|_| CommitError::ArithmeticOverflow)?;
    let heads =
        u32::try_from(shape.attention_heads).map_err(|_| CommitError::ArithmeticOverflow)?;
    let kv_heads =
        u32::try_from(shape.key_value_heads).map_err(|_| CommitError::ArithmeticOverflow)?;
    let head_dim = u32::try_from(shape.head_dim).map_err(|_| CommitError::ArithmeticOverflow)?;
    let experts = u32::try_from(shape.experts).map_err(|_| CommitError::ArithmeticOverflow)?;
    let top_k = u32::try_from(shape.active_experts).map_err(|_| CommitError::ArithmeticOverflow)?;
    let moe_shape = hidden
        | u32::try_from(shape.expert_intermediate)
            .map_err(|_| CommitError::ArithmeticOverflow)?
            .checked_shl(16)
            .ok_or(CommitError::ArithmeticOverflow)?;

    Ok(vec![
        macro_instruction(
            GraniteMacroOpcodeV1::RmsNorm,
            layer,
            ids.input_norm,
            hidden,
            shape.rms_epsilon.to_bits(),
        ),
        macro_instruction(
            GraniteMacroOpcodeV1::Dense,
            layer,
            ids.query,
            hidden,
            GraniteDenseOutputV1::Query as u32,
        ),
        macro_instruction(
            GraniteMacroOpcodeV1::Dense,
            layer,
            ids.key,
            kv_width,
            GraniteDenseOutputV1::Key as u32,
        ),
        macro_instruction(
            GraniteMacroOpcodeV1::Dense,
            layer,
            ids.value,
            kv_width,
            GraniteDenseOutputV1::Value as u32,
        ),
        macro_instruction(GraniteMacroOpcodeV1::Rope, layer, heads, kv_heads, head_dim),
        macro_instruction(
            GraniteMacroOpcodeV1::KvAppend,
            layer,
            kv_width,
            u32::try_from(shape.max_context).map_err(|_| CommitError::ArithmeticOverflow)?,
            0,
        ),
        macro_instruction(
            GraniteMacroOpcodeV1::Attention,
            layer,
            heads,
            kv_heads,
            qx_model::ATTENTION_MULTIPLIER.to_bits(),
        ),
        macro_instruction(
            GraniteMacroOpcodeV1::Dense,
            layer,
            ids.attention_output,
            hidden,
            GraniteDenseOutputV1::Attention as u32,
        ),
        macro_instruction(
            GraniteMacroOpcodeV1::Residual,
            layer,
            qx_model::RESIDUAL_MULTIPLIER.to_bits(),
            0,
            hidden,
        ),
        macro_instruction(
            GraniteMacroOpcodeV1::RmsNorm,
            layer,
            ids.post_attention_norm,
            hidden,
            shape.rms_epsilon.to_bits(),
        ),
        macro_instruction(
            GraniteMacroOpcodeV1::Dense,
            layer,
            ids.router,
            experts,
            GraniteDenseOutputV1::Router as u32,
        ),
        macro_instruction(
            GraniteMacroOpcodeV1::TopK,
            layer,
            experts,
            top_k,
            u32::from(shape.normalize_router_weights),
        ),
        macro_instruction(
            GraniteMacroOpcodeV1::Moe,
            layer,
            ids.gate_up_first,
            ids.down_first,
            moe_shape,
        ),
        macro_instruction(
            GraniteMacroOpcodeV1::Residual,
            layer,
            qx_model::RESIDUAL_MULTIPLIER.to_bits(),
            1,
            hidden,
        ),
    ])
}

pub fn granite_final_head_macro_program_v1() -> Result<Vec<VmInstructionV1>, CommitError> {
    let shape = qx_model::ModelShape::granite_v0();
    let hidden = u32::try_from(shape.hidden_size).map_err(|_| CommitError::ArithmeticOverflow)?;
    let vocab = u32::try_from(shape.vocabulary).map_err(|_| CommitError::ArithmeticOverflow)?;
    Ok(vec![
        macro_instruction(
            GraniteMacroOpcodeV1::RmsNorm,
            u32::MAX,
            granite_final_norm_tensor_id_v1()?,
            hidden,
            shape.rms_epsilon.to_bits(),
        ),
        macro_instruction(
            GraniteMacroOpcodeV1::FinalLogits,
            granite_embedding_tensor_id_v1(),
            vocab,
            hidden,
            qx_model::LOGITS_SCALING.to_bits(),
        ),
        macro_instruction(GraniteMacroOpcodeV1::ArgMax, vocab, 1, 0, 0),
    ])
}

pub fn granite_embedding_macro_program_v1() -> Vec<VmInstructionV1> {
    vec![macro_instruction(
        GraniteMacroOpcodeV1::Embedding,
        granite_embedding_tensor_id_v1(),
        qx_model::HIDDEN as u32,
        qx_model::EMBEDDING_MULTIPLIER.to_bits(),
        0,
    )]
}

#[derive(Clone, Debug)]
pub struct GraniteMacroPlanV1 {
    pub checkpoint_interval: u8,
    pub programs: Vec<Vec<VmInstructionV1>>,
    pub segments: Vec<VmSegmentProgramV1>,
    pub schedule: MerkleTree,
}

/// Lower the complete post-embedding Granite execution into a compact macro
/// schedule. Segment zero is the protocol-reserved leaf, checkpoint segments
/// cover `checkpoint_interval` layers, and the last segment is the final head.
pub fn granite_macro_plan_v1(checkpoint_interval: u8) -> Result<GraniteMacroPlanV1, CommitError> {
    if !matches!(checkpoint_interval, 1 | 2 | 4) {
        return Err(CommitError::InvalidShape(
            "Granite checkpoint interval must be 1, 2, or 4".to_owned(),
        ));
    }
    let layers = qx_model::LAYERS;
    let interval = usize::from(checkpoint_interval);
    let mut programs = Vec::with_capacity(layers / interval + 2);
    programs.push(vec![macro_instruction(
        GraniteMacroOpcodeV1::Reserved,
        0,
        0,
        0,
        0,
    )]);
    for start in (0..layers).step_by(interval) {
        let mut program = Vec::with_capacity(14 * interval);
        if start == 0 {
            program.extend(granite_embedding_macro_program_v1());
        }
        for layer in start..start + interval {
            program.extend(granite_layer_macro_program_v1(
                u16::try_from(layer).map_err(|_| CommitError::ArithmeticOverflow)?,
            )?);
        }
        programs.push(program);
    }
    programs.push(granite_final_head_macro_program_v1()?);

    let segments = programs
        .iter()
        .enumerate()
        .map(|(index, program)| {
            let tree = vm_program_tree_v1(program)?;
            Ok(VmSegmentProgramV1 {
                segment_index: u32::try_from(index).map_err(|_| CommitError::ArithmeticOverflow)?,
                instruction_root: tree.root(),
                instruction_count: u32::try_from(program.len())
                    .map_err(|_| CommitError::ArithmeticOverflow)?,
            })
        })
        .collect::<Result<Vec<_>, CommitError>>()?;
    let schedule = vm_segment_schedule_v1(&segments)?;
    Ok(GraniteMacroPlanV1 {
        checkpoint_interval,
        programs,
        segments,
        schedule,
    })
}

fn macro_slot(instruction: VmInstructionV1) -> Result<GraniteWorkspaceSlotV1, CommitError> {
    Ok(match instruction.opcode {
        value if value == GraniteMacroOpcodeV1::RmsNorm as u8 => {
            if instruction.source_a == u32::MAX {
                GraniteWorkspaceSlotV1::Normalized
            } else if instruction.source_b % GRANITE_TENSORS_PER_LAYER_V1 == 6 {
                GraniteWorkspaceSlotV1::MoeInput
            } else {
                GraniteWorkspaceSlotV1::Normalized
            }
        }
        value if value == GraniteMacroOpcodeV1::Dense as u8 => match instruction.destination {
            value if value == GraniteDenseOutputV1::Query as u32 => GraniteWorkspaceSlotV1::Query,
            value if value == GraniteDenseOutputV1::Key as u32 => GraniteWorkspaceSlotV1::Key,
            value if value == GraniteDenseOutputV1::Value as u32 => GraniteWorkspaceSlotV1::Value,
            value if value == GraniteDenseOutputV1::Attention as u32 => {
                GraniteWorkspaceSlotV1::AttentionProjection
            }
            value if value == GraniteDenseOutputV1::Router as u32 => {
                GraniteWorkspaceSlotV1::RouterLogits
            }
            _ => {
                return Err(CommitError::InvalidExecution(
                    "unknown dense output".to_owned(),
                ));
            }
        },
        value if value == GraniteMacroOpcodeV1::Rope as u8 => {
            GraniteWorkspaceSlotV1::RotatedQueryKey
        }
        value if value == GraniteMacroOpcodeV1::KvAppend as u8 => GraniteWorkspaceSlotV1::KvLeaf,
        value if value == GraniteMacroOpcodeV1::Attention as u8 => {
            GraniteWorkspaceSlotV1::Attention
        }
        value if value == GraniteMacroOpcodeV1::Residual as u8 => {
            if instruction.source_c == 0 {
                GraniteWorkspaceSlotV1::PostAttention
            } else {
                GraniteWorkspaceSlotV1::Residual
            }
        }
        value if value == GraniteMacroOpcodeV1::TopK as u8 => GraniteWorkspaceSlotV1::Routes,
        value if value == GraniteMacroOpcodeV1::Moe as u8 => GraniteWorkspaceSlotV1::Moe,
        value if value == GraniteMacroOpcodeV1::FinalLogits as u8 => GraniteWorkspaceSlotV1::Logits,
        value if value == GraniteMacroOpcodeV1::ArgMax as u8 => GraniteWorkspaceSlotV1::Winner,
        value if value == GraniteMacroOpcodeV1::Embedding as u8 => GraniteWorkspaceSlotV1::Residual,
        _ => {
            return Err(CommitError::InvalidExecution(
                "unknown Granite macro opcode".to_owned(),
            ));
        }
    })
}

/// Fold challenge-only macro outputs into a rolling commitment to every live
/// tensor root. `kv_roots` contains the authenticated global root after each
/// `KvAppend`, in program order. First and last leaves retain the existing VM
/// state hashes so receipt checkpoint formats do not change.
pub fn granite_macro_trace_v1(
    pre_state: VmStateV1,
    program: &[VmInstructionV1],
    outputs: &[GraniteMacroOutputV1],
    kv_roots: &[Digest],
    predicted_token: Option<u32>,
) -> Result<GraniteMacroTraceV1, CommitError> {
    if pre_state.program_counter != 0 || program.len() != outputs.len() || program.is_empty() {
        return Err(CommitError::InvalidExecution(
            "macro trace shape does not match its segment program".to_owned(),
        ));
    }
    let expected_kv = program
        .iter()
        .filter(|instruction| instruction.opcode == GraniteMacroOpcodeV1::KvAppend as u8)
        .count();
    if expected_kv != kv_roots.len() {
        return Err(CommitError::InvalidExecution(
            "macro trace K/V frontier count is inconsistent".to_owned(),
        ));
    }
    let mut state_hashes = Vec::with_capacity(program.len() + 1);
    state_hashes.push(pre_state.hash()?);
    let mut workspace = GraniteMacroWorkspaceV1::new(pre_state.residual_root);
    let mut global_kv_root = pre_state.global_kv_root;
    let mut kv_index = 0;
    for (index, (instruction, output)) in program.iter().zip(outputs).enumerate() {
        if instruction.opcode != output.opcode as u8 {
            return Err(CommitError::InvalidExecution(
                "macro output opcode does not match its registered instruction".to_owned(),
            ));
        }
        workspace.set(macro_slot(*instruction)?, output.output_root);
        if instruction.opcode == GraniteMacroOpcodeV1::KvAppend as u8 {
            global_kv_root = kv_roots[kv_index];
            kv_index += 1;
        }
        let next_pc = u32::try_from(index + 1).map_err(|_| CommitError::ArithmeticOverflow)?;
        let last = index + 1 == program.len();
        let hash = if last {
            if let Some(predicted_token) = predicted_token {
                VmOutputStateV1 {
                    job_hash: pre_state.job_hash,
                    token_index: pre_state.token_index,
                    segment_index: pre_state.segment_index,
                    program_counter: next_pc,
                    residual_root: workspace.root()?,
                    global_kv_root,
                    predicted_token,
                }
                .hash()?
            } else {
                VmStateV1 {
                    job_hash: pre_state.job_hash,
                    token_index: pre_state.token_index,
                    segment_index: pre_state.segment_index,
                    program_counter: next_pc,
                    residual_root: workspace.get(GraniteWorkspaceSlotV1::Residual),
                    global_kv_root,
                }
                .hash()?
            }
        } else {
            GraniteMacroStateV1 {
                job_hash: pre_state.job_hash,
                token_index: pre_state.token_index,
                segment_index: pre_state.segment_index,
                program_counter: next_pc,
                workspace_root: workspace.root()?,
                global_kv_root,
            }
            .hash()?
        };
        state_hashes.push(hash);
    }
    let final_residual_root = if predicted_token.is_some() {
        workspace.root()?
    } else {
        workspace.get(GraniteWorkspaceSlotV1::Residual)
    };
    let tree = MerkleTree::new(HashAlgorithm::Blake3V1, D_OPERATOR_TREE, &state_hashes)?;
    Ok(GraniteMacroTraceV1 {
        state_hashes,
        tree,
        final_residual_root,
        final_global_kv_root: global_kv_root,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_tensor_ids_cover_the_registered_artifact_shape() {
        assert_eq!(granite_embedding_tensor_id_v1(), 0);
        let first = granite_layer_tensor_ids_v1(0).unwrap();
        assert_eq!(first.input_norm, 1);
        assert_eq!(first.gate_up_first, 8);
        assert_eq!(first.down_first, 40);
        let last = granite_layer_tensor_ids_v1(23).unwrap();
        assert_eq!(last.down_first + 31, 1_704);
        assert_eq!(granite_final_norm_tensor_id_v1().unwrap(), 1_705);
        assert!(granite_layer_tensor_ids_v1(24).is_err());
    }

    #[test]
    fn interval_two_plan_is_small_and_binds_each_real_layer() {
        let plan = granite_macro_plan_v1(2).unwrap();
        assert_eq!(plan.programs.len(), 14);
        assert_eq!(plan.programs[0].len(), 1);
        assert_eq!(plan.programs[1].len(), 29);
        assert_eq!(plan.programs[12].len(), 28);
        assert_eq!(plan.programs[13].len(), 3);
        assert_eq!(plan.segments.len(), plan.programs.len());
        assert_ne!(
            plan.segments[1].instruction_root,
            plan.segments[2].instruction_root
        );
        assert_ne!(plan.schedule.root(), [0; 32]);
    }

    #[test]
    fn every_supported_interval_has_dense_ordered_segments() {
        for interval in [1, 2, 4] {
            let plan = granite_macro_plan_v1(interval).unwrap();
            assert_eq!(
                plan.programs.len(),
                qx_model::LAYERS / usize::from(interval) + 2
            );
            for (index, segment) in plan.segments.iter().enumerate() {
                assert_eq!(segment.segment_index, index as u32);
                assert_eq!(
                    segment.instruction_count as usize,
                    plan.programs[index].len()
                );
            }
        }
        assert!(granite_macro_plan_v1(3).is_err());
    }

    #[test]
    fn installed_production_artifact_matches_the_macro_tensor_map() {
        let artifact = Path::new(env!("CARGO_MANIFEST_DIR")).join(
            "../../external/models/granite-m5-pro-q4-rowinline-k4-vector-unroll4-down-short-context-active.qx",
        );
        if artifact.exists() {
            validate_granite_macro_artifact_v1(artifact).unwrap();
        }
    }

    #[test]
    fn rolling_workspace_binds_every_live_macro_output() {
        let program = granite_layer_macro_program_v1(0).unwrap();
        let outputs = program
            .iter()
            .enumerate()
            .map(|(index, instruction)| GraniteMacroOutputV1 {
                opcode: GraniteMacroOpcodeV1::try_from(instruction.opcode).unwrap(),
                output_root: [u8::try_from(index + 1).unwrap(); 32],
            })
            .collect::<Vec<_>>();
        let pre_state = VmStateV1 {
            job_hash: [1; 32],
            token_index: 2,
            segment_index: 3,
            program_counter: 0,
            residual_root: [4; 32],
            global_kv_root: [5; 32],
        };
        let trace =
            granite_macro_trace_v1(pre_state, &program, &outputs, &[[6; 32]], None).unwrap();
        assert_eq!(trace.state_hashes.len(), program.len() + 1);
        assert_eq!(trace.state_hashes[0], pre_state.hash().unwrap());
        assert_eq!(
            trace.final_residual_root,
            outputs.last().unwrap().output_root
        );
        assert_eq!(trace.final_global_kv_root, [6; 32]);
        let expected_end = VmStateV1 {
            program_counter: program.len() as u32,
            residual_root: trace.final_residual_root,
            global_kv_root: [6; 32],
            ..pre_state
        }
        .hash()
        .unwrap();
        assert_eq!(*trace.state_hashes.last().unwrap(), expected_end);

        let mut corrupt = outputs.clone();
        corrupt[1].output_root[0] ^= 1;
        let corrupt_trace =
            granite_macro_trace_v1(pre_state, &program, &corrupt, &[[6; 32]], None).unwrap();
        assert_ne!(trace.tree.root(), corrupt_trace.tree.root());
        assert_eq!(trace.state_hashes.last(), corrupt_trace.state_hashes.last());
    }
}
