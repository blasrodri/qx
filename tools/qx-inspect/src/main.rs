use anyhow::{Context, Result, bail};
use clap::Parser;
use qx_model::{
    ATTENTION_HEADS, EXPERT_INTERMEDIATE, EXPERTS, Granite3A400M, HIDDEN, KV_HEADS, LAYERS, TOP_K,
};
use qx_quant::{QuantType, QuantizedTensor};
use qx_safetensors::Checkpoint;
use std::{fs, path::PathBuf};

#[derive(Debug, Parser)]
#[command(
    name = "qx-inspect",
    about = "Inspect the fixed Granite 3.0 1B-A400M QX V0 input"
)]
struct Args {
    /// Hugging Face model directory containing config.json and safetensors.
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    layer: Option<usize>,
    #[arg(long)]
    expert: Option<usize>,
    #[arg(long, value_parser = parse_quant_type)]
    quantize: Option<QuantType>,
}

fn parse_quant_type(value: &str) -> Result<QuantType, String> {
    QuantType::parse(value).ok_or_else(|| "expected q2, q3, or q4".to_owned())
}

fn main() -> Result<()> {
    let args = Args::parse();
    if args.expert.is_some() != args.layer.is_some() {
        bail!("--layer and --expert must be supplied together");
    }
    if let Some(layer) = args.layer
        && layer >= LAYERS
    {
        bail!("layer must be in 0..{LAYERS}");
    }
    if let Some(expert) = args.expert
        && expert >= EXPERTS
    {
        bail!("expert must be in 0..{EXPERTS}");
    }

    let config_path = args.model.join("config.json");
    let config = serde_json::from_slice::<qx_model::GraniteConfig>(
        &fs::read(&config_path).with_context(|| format!("read {}", config_path.display()))?,
    )
    .with_context(|| format!("parse {}", config_path.display()))?;
    config.validate_v0().map_err(anyhow::Error::msg)?;

    println!("QX Model Inspector\n");
    println!("model:        {}", Granite3A400M::MODEL_ID);
    println!("architecture: GraniteMoE");
    println!("layers:       {LAYERS}");
    println!("hidden:       {HIDDEN}");
    println!("attention:    {ATTENTION_HEADS} heads / {KV_HEADS} KV heads");
    println!("experts:      {EXPERTS}");
    println!("top-k:        {TOP_K}");
    println!("intermediate: {EXPERT_INTERMEDIATE}");

    let checkpoint = Checkpoint::open(&args.model)?;
    if let (Some(layer), Some(expert)) = (args.layer, args.expert) {
        inspect_expert(&checkpoint, layer, expert, args.quantize)?;
    } else {
        print_layer_overview(&checkpoint, 0)?;
        let count = checkpoint.tensor_names().count();
        println!("\ntensors indexed: {count}");
    }
    Ok(())
}

fn print_layer_overview(checkpoint: &Checkpoint, layer: usize) -> Result<()> {
    println!("\nLayer {layer}:");
    for suffix in [
        "router.layer.weight",
        "input_linear.weight",
        "output_linear.weight",
    ] {
        let needle = format!("model.layers.{layer}.block_sparse_moe.{suffix}");
        if let Ok(info) = checkpoint.info(&needle) {
            println!(
                "{suffix}\n    shape: {:?}\n    dtype: {}\n    bytes: {}",
                info.shape, info.dtype, info.byte_len
            );
        } else {
            println!("{suffix}\n    not found");
        }
    }
    println!("\nestimated expert bytes (one input + output expert, FP16 source):");
    for quant_type in [QuantType::Q4, QuantType::Q3, QuantType::Q2] {
        let input = estimate_quantized_bytes(Granite3A400M::expert_input_elements(), quant_type);
        let output = estimate_quantized_bytes(Granite3A400M::expert_output_elements(), quant_type);
        println!("    {quant_type}: {} bytes", input + output);
    }
    Ok(())
}

fn inspect_expert(
    checkpoint: &Checkpoint,
    layer: usize,
    expert: usize,
    requested: Option<QuantType>,
) -> Result<()> {
    println!("\nExpert {expert} / Layer {layer}");
    for (label, name) in [
        (
            "input projection",
            format!("model.layers.{layer}.block_sparse_moe.input_linear.weight"),
        ),
        (
            "output projection",
            format!("model.layers.{layer}.block_sparse_moe.output_linear.weight"),
        ),
    ] {
        let info = checkpoint
            .info(&name)
            .with_context(|| format!("find {label} ({name})"))?;
        println!(
            "\n{label}:\n    shape: {:?}\n    dtype: {}\n    full tensor bytes: {}",
            info.shape, info.dtype, info.byte_len
        );
        let (rows, cols) = expert_matrix_shape(&info.shape, label)?;
        let (shape, values) = checkpoint.tensor_f32(&name)?;
        let values = first_expert(&shape, &values, expert, rows * cols)?;
        println!("    expert shape: [{rows}, {cols}]");
        println!("    original bytes (FP16 estimate): {}", values.len() * 2);
        for quant_type in requested
            .into_iter()
            .chain([QuantType::Q4, QuantType::Q3, QuantType::Q2])
        {
            let q = QuantizedTensor::quantize(&values, rows, cols, quant_type, 32);
            println!(
                "    {quant_type}: {} bytes, compression: {:.2}x, MSE: {:.6}, relative L2: {:.6}",
                q.bytes(),
                values.len() as f32 * 2.0 / q.bytes() as f32,
                q.mse(&values),
                q.relative_l2(&values)
            );
        }
    }
    Ok(())
}

fn expert_matrix_shape(shape: &[usize], label: &str) -> Result<(usize, usize)> {
    if shape.len() != 3 || shape[0] != EXPERTS {
        bail!("{label} must have shape [32, rows, cols], got {shape:?}");
    }
    Ok((shape[1], shape[2]))
}

fn first_expert(
    shape: &[usize],
    values: &[f32],
    expert: usize,
    elements: usize,
) -> Result<Vec<f32>> {
    if shape.first().copied() != Some(EXPERTS) || values.len() != shape.iter().product::<usize>() {
        bail!("unexpected expert tensor shape/data");
    }
    let start = expert * elements;
    Ok(values[start..start + elements].to_vec())
}

fn estimate_quantized_bytes(elements: usize, quant_type: QuantType) -> usize {
    let groups = elements.div_ceil(32);
    (elements * quant_type.bits()).div_ceil(8) + groups * 2
}
