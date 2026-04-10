use crate::models::with_tracing::{linear_b, linear_no_bias, Linear};
use candle::{DType, Device, DeviceLocation, IndexOp, Module, Result, Tensor, D};
use candle_nn::{conv1d_no_bias, embedding, ops, Conv1d, Conv1dConfig, Embedding, VarBuilder};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

fn elapsed_millis(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1_000.0
}

fn profile_sync_enabled(device: &Device) -> bool {
    matches!(
        device.location(),
        DeviceLocation::Metal { .. } | DeviceLocation::Cuda { .. } | DeviceLocation::Hip { .. }
    ) && matches!(
        std::env::var("CANDLE_QWEN35_PROFILE_SYNC").as_deref(),
        Ok("1" | "true" | "TRUE" | "yes" | "YES")
    )
}

fn profile_start(device: &Device) -> Result<Instant> {
    if profile_sync_enabled(device) {
        device.synchronize()?;
    }
    Ok(Instant::now())
}

fn profile_elapsed(start: Instant, device: &Device) -> Result<f64> {
    if profile_sync_enabled(device) {
        device.synchronize()?;
    }
    Ok(elapsed_millis(start))
}

fn default_attention_bias() -> bool {
    false
}

fn default_attention_dropout() -> f64 {
    0.0
}

fn default_head_dim() -> usize {
    256
}

fn default_linear_conv_kernel_dim() -> usize {
    4
}

fn default_linear_key_head_dim() -> usize {
    128
}

fn default_linear_value_head_dim() -> usize {
    128
}

fn default_linear_num_key_heads() -> usize {
    16
}

fn default_linear_num_value_heads() -> usize {
    32
}

fn default_partial_rotary_factor() -> f64 {
    0.25
}

fn default_rope_theta() -> f64 {
    10_000.0
}

fn default_rope_type() -> String {
    "default".to_string()
}

#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
pub struct RopeParameters {
    #[serde(default = "default_rope_type")]
    pub rope_type: String,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
    #[serde(default = "default_partial_rotary_factor")]
    pub partial_rotary_factor: f64,
}

impl Default for RopeParameters {
    fn default() -> Self {
        Self {
            rope_type: default_rope_type(),
            rope_theta: default_rope_theta(),
            partial_rotary_factor: default_partial_rotary_factor(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
pub struct TextConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub hidden_act: candle_nn::Activation,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f64,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default = "default_attention_bias")]
    pub attention_bias: bool,
    #[serde(default = "default_attention_dropout")]
    pub attention_dropout: f64,
    #[serde(default = "default_head_dim")]
    pub head_dim: usize,
    #[serde(default = "default_linear_conv_kernel_dim")]
    pub linear_conv_kernel_dim: usize,
    #[serde(default = "default_linear_key_head_dim")]
    pub linear_key_head_dim: usize,
    #[serde(default = "default_linear_value_head_dim")]
    pub linear_value_head_dim: usize,
    #[serde(default = "default_linear_num_key_heads")]
    pub linear_num_key_heads: usize,
    #[serde(default = "default_linear_num_value_heads")]
    pub linear_num_value_heads: usize,
    #[serde(default)]
    pub layer_types: Vec<String>,
    #[serde(default)]
    pub rope_parameters: Option<RopeParameters>,
}

impl TextConfig {
    pub fn normalized(mut self) -> Self {
        if self.layer_types.is_empty() {
            self.layer_types = (0..self.num_hidden_layers)
                .map(|idx| {
                    if (idx + 1) % 4 == 0 {
                        "full_attention".to_string()
                    } else {
                        "linear_attention".to_string()
                    }
                })
                .collect();
        }
        self
    }

    fn rope_theta(&self) -> f64 {
        self.rope_parameters
            .as_ref()
            .map(|params| params.rope_theta)
            .unwrap_or_else(default_rope_theta)
    }

    fn partial_rotary_factor(&self) -> f64 {
        self.rope_parameters
            .as_ref()
            .map(|params| params.partial_rotary_factor)
            .unwrap_or_else(default_partial_rotary_factor)
    }
}

#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
pub struct Config {
    pub text_config: TextConfig,
}

impl Config {
    pub fn normalized(mut self) -> Self {
        self.text_config = self.text_config.normalized();
        self
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct RuntimeProfile {
    pub qkv_projection_millis: f64,
    pub kv_append_write_millis: f64,
    pub layout_prepare_millis: f64,
    pub attention_score_millis: f64,
    pub attention_softmax_millis: f64,
    pub attention_mix_millis: f64,
    pub output_projection_millis: f64,
    pub full_attention_mask_prepare_millis: f64,
    pub full_attention_input_layout_millis: f64,
    pub full_attention_kv_materialize_millis: f64,
    pub full_attention_output_collect_millis: f64,
    pub full_attention_output_reshape_millis: f64,
    pub full_attention_gate_millis: f64,
    pub full_attention_kernel_execute_millis: f64,
    pub scheduler_planning_millis: f64,
    pub transfer_millis: f64,
    pub linear_attention_millis: f64,
    pub full_attention_millis: f64,
    pub mlp_millis: f64,
    pub linear_conv_millis: f64,
    pub linear_chunk_prepare_millis: f64,
    pub linear_chunk_solve_millis: f64,
    pub linear_chunk_scan_millis: f64,
    pub linear_chunk_index_millis: f64,
    pub linear_chunk_local_attn_millis: f64,
    pub linear_chunk_recurrent_read_millis: f64,
    pub linear_chunk_state_update_millis: f64,
    pub linear_recurrent_loop_millis: f64,
    pub linear_full_kernel_pack_millis: f64,
    pub linear_full_kernel_execute_millis: f64,
    pub linear_full_kernel_unpack_millis: f64,
}

impl RuntimeProfile {
    pub fn add_assign(&mut self, other: &Self) {
        self.qkv_projection_millis += other.qkv_projection_millis;
        self.kv_append_write_millis += other.kv_append_write_millis;
        self.layout_prepare_millis += other.layout_prepare_millis;
        self.attention_score_millis += other.attention_score_millis;
        self.attention_softmax_millis += other.attention_softmax_millis;
        self.attention_mix_millis += other.attention_mix_millis;
        self.output_projection_millis += other.output_projection_millis;
        self.full_attention_mask_prepare_millis += other.full_attention_mask_prepare_millis;
        self.full_attention_input_layout_millis += other.full_attention_input_layout_millis;
        self.full_attention_kv_materialize_millis += other.full_attention_kv_materialize_millis;
        self.full_attention_output_collect_millis += other.full_attention_output_collect_millis;
        self.full_attention_output_reshape_millis += other.full_attention_output_reshape_millis;
        self.full_attention_gate_millis += other.full_attention_gate_millis;
        self.full_attention_kernel_execute_millis += other.full_attention_kernel_execute_millis;
        self.scheduler_planning_millis += other.scheduler_planning_millis;
        self.transfer_millis += other.transfer_millis;
        self.linear_attention_millis += other.linear_attention_millis;
        self.full_attention_millis += other.full_attention_millis;
        self.mlp_millis += other.mlp_millis;
        self.linear_conv_millis += other.linear_conv_millis;
        self.linear_chunk_prepare_millis += other.linear_chunk_prepare_millis;
        self.linear_chunk_solve_millis += other.linear_chunk_solve_millis;
        self.linear_chunk_scan_millis += other.linear_chunk_scan_millis;
        self.linear_chunk_index_millis += other.linear_chunk_index_millis;
        self.linear_chunk_local_attn_millis += other.linear_chunk_local_attn_millis;
        self.linear_chunk_recurrent_read_millis += other.linear_chunk_recurrent_read_millis;
        self.linear_chunk_state_update_millis += other.linear_chunk_state_update_millis;
        self.linear_recurrent_loop_millis += other.linear_recurrent_loop_millis;
        self.linear_full_kernel_pack_millis += other.linear_full_kernel_pack_millis;
        self.linear_full_kernel_execute_millis += other.linear_full_kernel_execute_millis;
        self.linear_full_kernel_unpack_millis += other.linear_full_kernel_unpack_millis;
    }

    pub fn scaled(&self, factor: f64) -> Self {
        Self {
            qkv_projection_millis: self.qkv_projection_millis * factor,
            kv_append_write_millis: self.kv_append_write_millis * factor,
            layout_prepare_millis: self.layout_prepare_millis * factor,
            attention_score_millis: self.attention_score_millis * factor,
            attention_softmax_millis: self.attention_softmax_millis * factor,
            attention_mix_millis: self.attention_mix_millis * factor,
            output_projection_millis: self.output_projection_millis * factor,
            full_attention_mask_prepare_millis: self.full_attention_mask_prepare_millis * factor,
            full_attention_input_layout_millis: self.full_attention_input_layout_millis * factor,
            full_attention_kv_materialize_millis: self.full_attention_kv_materialize_millis
                * factor,
            full_attention_output_collect_millis: self.full_attention_output_collect_millis
                * factor,
            full_attention_output_reshape_millis: self.full_attention_output_reshape_millis
                * factor,
            full_attention_gate_millis: self.full_attention_gate_millis * factor,
            full_attention_kernel_execute_millis: self.full_attention_kernel_execute_millis
                * factor,
            scheduler_planning_millis: self.scheduler_planning_millis * factor,
            transfer_millis: self.transfer_millis * factor,
            linear_attention_millis: self.linear_attention_millis * factor,
            full_attention_millis: self.full_attention_millis * factor,
            mlp_millis: self.mlp_millis * factor,
            linear_conv_millis: self.linear_conv_millis * factor,
            linear_chunk_prepare_millis: self.linear_chunk_prepare_millis * factor,
            linear_chunk_solve_millis: self.linear_chunk_solve_millis * factor,
            linear_chunk_scan_millis: self.linear_chunk_scan_millis * factor,
            linear_chunk_index_millis: self.linear_chunk_index_millis * factor,
            linear_chunk_local_attn_millis: self.linear_chunk_local_attn_millis * factor,
            linear_chunk_recurrent_read_millis: self.linear_chunk_recurrent_read_millis * factor,
            linear_chunk_state_update_millis: self.linear_chunk_state_update_millis * factor,
            linear_recurrent_loop_millis: self.linear_recurrent_loop_millis * factor,
            linear_full_kernel_pack_millis: self.linear_full_kernel_pack_millis * factor,
            linear_full_kernel_execute_millis: self.linear_full_kernel_execute_millis * factor,
            linear_full_kernel_unpack_millis: self.linear_full_kernel_unpack_millis * factor,
        }
    }
}

#[derive(Debug, Clone)]
pub struct LinearAttentionBenchResult {
    pub layer_id: usize,
    pub sequence_length: usize,
    pub repeats: usize,
    pub mean_total_millis: f64,
    pub best_total_millis: f64,
    pub iteration_total_millis: Vec<f64>,
    pub mean_profile: RuntimeProfile,
    pub best_profile: RuntimeProfile,
}

#[derive(Debug, Clone)]
pub struct LinearAttentionTrace {
    pub layer_id: usize,
    pub sequence_length: usize,
    pub layer_output: Tensor,
    pub recurrent_state: Tensor,
    pub profile: RuntimeProfile,
}

#[derive(Debug, Clone)]
struct RotaryEmbedding {
    cos: Tensor,
    sin: Tensor,
    rotary_dim: usize,
}

impl RotaryEmbedding {
    fn new(cfg: &TextConfig, device: &Device, dtype: DType) -> Result<Self> {
        let rotary_dim = ((cfg.head_dim as f64) * cfg.partial_rotary_factor()).round() as usize;
        let rotary_dim = rotary_dim.max(2).min(cfg.head_dim);
        let inv_freq: Vec<_> = (0..rotary_dim)
            .step_by(2)
            .map(|i| 1f32 / cfg.rope_theta().powf(i as f64 / rotary_dim as f64) as f32)
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq =
            Tensor::from_vec(inv_freq, (1, inv_freq_len), device)?.to_dtype(DType::F32)?;
        let positions = Tensor::arange(0u32, cfg.max_position_embeddings as u32, device)?
            .to_dtype(DType::F32)?
            .reshape((cfg.max_position_embeddings, 1))?;
        let freqs = positions.matmul(&inv_freq)?;
        Ok(Self {
            cos: freqs.cos()?.to_dtype(dtype)?,
            sin: freqs.sin()?.to_dtype(dtype)?,
            rotary_dim,
        })
    }

    fn apply(&self, q: &Tensor, k: &Tensor, seqlen_offset: usize) -> Result<(Tensor, Tensor)> {
        let (_, _, seq_len, head_dim) = q.dims4()?;
        if self.rotary_dim >= head_dim {
            let cos = self.cos.narrow(0, seqlen_offset, seq_len)?;
            let sin = self.sin.narrow(0, seqlen_offset, seq_len)?;
            let q_embed = candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)?;
            let k_embed = candle_nn::rotary_emb::rope(&k.contiguous()?, &cos, &sin)?;
            return Ok((q_embed, k_embed));
        }

        let q_rot = q.narrow(D::Minus1, 0, self.rotary_dim)?;
        let q_pass = q.narrow(D::Minus1, self.rotary_dim, head_dim - self.rotary_dim)?;
        let k_rot = k.narrow(D::Minus1, 0, self.rotary_dim)?;
        let k_pass = k.narrow(D::Minus1, self.rotary_dim, head_dim - self.rotary_dim)?;
        let cos = self.cos.narrow(0, seqlen_offset, seq_len)?;
        let sin = self.sin.narrow(0, seqlen_offset, seq_len)?;
        let q_rot = candle_nn::rotary_emb::rope(&q_rot.contiguous()?, &cos, &sin)?;
        let k_rot = candle_nn::rotary_emb::rope(&k_rot.contiguous()?, &cos, &sin)?;
        Ok((
            Tensor::cat(&[&q_rot, &q_pass], D::Minus1)?,
            Tensor::cat(&[&k_rot, &k_pass], D::Minus1)?,
        ))
    }
}

#[derive(Debug, Clone)]
struct Qwen35RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl Qwen35RmsNorm {
    fn new(dim: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            weight: vb.get(dim, "weight")?,
            eps,
        })
    }
}

impl Module for Qwen35RmsNorm {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs_dtype = xs.dtype();
        let xs = xs.to_dtype(DType::F32)?;
        let variance = (xs.sqr()?.sum_keepdim(D::Minus1)? / xs.dim(D::Minus1)? as f64)?;
        let xs = xs.broadcast_div(&(variance + self.eps)?.sqrt()?)?;
        let weight = (self.weight.to_dtype(DType::F32)? + 1.0)?;
        xs.broadcast_mul(&weight)?.to_dtype(xs_dtype)
    }
}

#[derive(Debug, Clone)]
struct Qwen35RmsNormGated {
    weight: Tensor,
    eps: f64,
}

impl Qwen35RmsNormGated {
    fn new(dim: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            weight: vb.get(dim, "weight")?,
            eps,
        })
    }

    fn forward(&self, hidden_states: &Tensor, gate: &Tensor) -> Result<Tensor> {
        let out_dtype = hidden_states.dtype();
        let hidden_states = hidden_states.to_dtype(DType::F32)?;
        let variance =
            (hidden_states.sqr()?.sum_keepdim(D::Minus1)? / hidden_states.dim(D::Minus1)? as f64)?;
        let hidden_states = hidden_states.broadcast_div(&(variance + self.eps)?.sqrt()?)?;
        let hidden_states = hidden_states.broadcast_mul(&self.weight.to_dtype(DType::F32)?)?;
        let gated = hidden_states.broadcast_mul(&ops::silu(&gate.to_dtype(DType::F32)?)?)?;
        gated.to_dtype(out_dtype)
    }
}

#[derive(Debug, Clone)]
struct Mlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    act_fn: candle_nn::Activation,
}

impl Mlp {
    fn new(cfg: &TextConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            gate_proj: linear_no_bias(cfg.hidden_size, cfg.intermediate_size, vb.pp("gate_proj"))?,
            up_proj: linear_no_bias(cfg.hidden_size, cfg.intermediate_size, vb.pp("up_proj"))?,
            down_proj: linear_no_bias(cfg.intermediate_size, cfg.hidden_size, vb.pp("down_proj"))?,
            act_fn: cfg.hidden_act,
        })
    }
}

impl Module for Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let lhs = xs.apply(&self.gate_proj)?.apply(&self.act_fn)?;
        let rhs = xs.apply(&self.up_proj)?;
        (lhs * rhs)?.apply(&self.down_proj)
    }
}

fn repeat_heads(xs: &Tensor, n_rep: usize) -> Result<Tensor> {
    let (b_sz, seq_len, heads, head_dim) = xs.dims4()?;
    if n_rep == 1 {
        return Ok(xs.clone());
    }
    xs.reshape((b_sz, seq_len, heads, 1, head_dim))?
        .expand((b_sz, seq_len, heads, n_rep, head_dim))?
        .reshape((b_sz, seq_len, heads * n_rep, head_dim))
}

fn l2norm(xs: &Tensor, eps: f64) -> Result<Tensor> {
    let norm = xs.sqr()?.sum_keepdim(D::Minus1)?;
    xs.broadcast_div(&(norm + eps)?.sqrt()?)
}

fn softplus(xs: &Tensor) -> Result<Tensor> {
    ((xs.exp()? + 1.0)?).log()
}

fn linear_attention_compute_dtype(device: &Device, input_dtype: DType) -> DType {
    match (device.location(), input_dtype) {
        (DeviceLocation::Metal { .. }, DType::F16 | DType::BF16) => input_dtype,
        _ => DType::F32,
    }
}

fn recommended_metal_linear_chunk_size(sequence_length: usize) -> usize {
    match sequence_length {
        0..=1024 => 16,
        _ => 24,
    }
}

fn debug_linear_chunk_choice(sequence_length: usize, chunk_size: usize) {
    static LOGGED: AtomicBool = AtomicBool::new(false);
    if std::env::var("CANDLE_QWEN35_DEBUG_CHUNK").is_ok() && !LOGGED.swap(true, Ordering::Relaxed) {
        eprintln!(
            "qwen3.5 linear chunk choice: sequence_length={} chunk_size={}",
            sequence_length, chunk_size
        );
    }
}

fn linear_attention_chunk_size(device: &Device, sequence_length: usize) -> usize {
    if let Ok(raw_value) = std::env::var("CANDLE_QWEN35_LINEAR_CHUNK_SIZE") {
        if let Ok(parsed) = raw_value.trim().parse::<usize>() {
            if parsed > 0 {
                debug_linear_chunk_choice(sequence_length, parsed);
                return parsed;
            }
        }
    }
    let chunk_size = match device.location() {
        DeviceLocation::Metal { .. } => recommended_metal_linear_chunk_size(sequence_length),
        _ => 64,
    };
    debug_linear_chunk_choice(sequence_length, chunk_size);
    chunk_size
}

fn use_delta_state_kernel(
    device: &Device,
    scan_mode: DeltaNetScanMode,
    sequence_length: usize,
) -> bool {
    matches!(device.location(), DeviceLocation::Metal { .. })
        && matches!(scan_mode, DeltaNetScanMode::PrebatchedLocal)
        && sequence_length >= 4096
        && matches!(
            std::env::var("CANDLE_QWEN35_DELTA_STATE_KERNEL").as_deref(),
            Ok("1") | Ok("true") | Ok("TRUE") | Ok("yes") | Ok("YES")
        )
}

fn use_delta_state_scan_kernel(
    device: &Device,
    scan_mode: DeltaNetScanMode,
    sequence_length: usize,
) -> bool {
    if !(matches!(scan_mode, DeltaNetScanMode::PrebatchedLocal) && sequence_length >= 4096) {
        return false;
    }

    match device.location() {
        DeviceLocation::Metal { .. } | DeviceLocation::Cuda { .. } | DeviceLocation::Hip { .. } => {
            matches!(
                std::env::var("CANDLE_QWEN35_DELTA_STATE_SCAN_KERNEL").as_deref(),
                Ok("1") | Ok("true") | Ok("TRUE") | Ok("yes") | Ok("YES")
            )
        }
        _ => false,
    }
}

fn use_delta_chunk_fused_kernel(
    device: &Device,
    scan_mode: DeltaNetScanMode,
    sequence_length: usize,
) -> bool {
    if !(matches!(scan_mode, DeltaNetScanMode::PrebatchedLocal) && sequence_length >= 4096) {
        return false;
    }

    match device.location() {
        DeviceLocation::Metal { .. } | DeviceLocation::Cuda { .. } | DeviceLocation::Hip { .. } => {
            matches!(
                std::env::var("CANDLE_QWEN35_DELTA_CHUNK_FUSED_KERNEL").as_deref(),
                Ok("1") | Ok("true") | Ok("TRUE") | Ok("yes") | Ok("YES")
            )
        }
        _ => false,
    }
}

fn use_delta_full_scan_kernel(
    device: &Device,
    scan_mode: DeltaNetScanMode,
    sequence_length: usize,
) -> bool {
    if !(matches!(scan_mode, DeltaNetScanMode::PrebatchedLocal) && sequence_length >= 4096) {
        return false;
    }

    match device.location() {
        DeviceLocation::Metal { .. } | DeviceLocation::Cuda { .. } | DeviceLocation::Hip { .. } => {
            matches!(
                std::env::var("CANDLE_QWEN35_DELTA_FULL_KERNEL").as_deref(),
                Ok("1") | Ok("true") | Ok("TRUE") | Ok("yes") | Ok("YES")
            )
        }
        _ => false,
    }
}

fn use_delta_recurrent_prefill_kernel(device: &Device, sequence_length: usize) -> bool {
    sequence_length >= 4096
        && match device.location() {
            DeviceLocation::Metal { .. }
            | DeviceLocation::Cuda { .. }
            | DeviceLocation::Hip { .. } => matches!(
                std::env::var("CANDLE_QWEN35_DELTA_RECURRENT_PREFILL_KERNEL").as_deref(),
                Ok("1") | Ok("true") | Ok("TRUE") | Ok("yes") | Ok("YES")
            ),
            _ => false,
        }
}

fn use_delta_chunk_step_kernel(
    device: &Device,
    scan_mode: DeltaNetScanMode,
    sequence_length: usize,
    chunk_size: usize,
) -> bool {
    if !(matches!(scan_mode, DeltaNetScanMode::PrebatchedLocal)
        && sequence_length >= 2048
        && chunk_size <= 24)
    {
        return false;
    }

    match device.location() {
        DeviceLocation::Metal { .. } => {
            match std::env::var("CANDLE_QWEN35_DELTA_CHUNK_STEP_KERNEL") {
                Ok(value)
                    if matches!(
                        value.as_str(),
                        "0" | "false" | "FALSE" | "no" | "NO" | "off" | "OFF"
                    ) =>
                {
                    false
                }
                Ok(_) => true,
                Err(_) => true,
            }
        }
        DeviceLocation::Cuda { .. } | DeviceLocation::Hip { .. } => matches!(
            std::env::var("CANDLE_QWEN35_DELTA_CHUNK_STEP_KERNEL").as_deref(),
            Ok("1") | Ok("true") | Ok("TRUE") | Ok("yes") | Ok("YES")
        ),
        _ => false,
    }
}

fn delta_chunk_step_2d_enabled() -> bool {
    match std::env::var("CANDLE_QWEN35_DELTA_CHUNK_STEP_2D_KERNEL") {
        Ok(value)
            if matches!(
                value.as_str(),
                "0" | "false" | "FALSE" | "no" | "NO" | "off" | "OFF"
            ) =>
        {
            false
        }
        Ok(_) => true,
        Err(_) => true,
    }
}

fn delta_chunk_step_windowed_2d_enabled() -> bool {
    matches!(
        std::env::var("CANDLE_QWEN35_DELTA_CHUNK_WINDOWED_2D_KERNEL").as_deref(),
        Ok("1" | "true" | "TRUE" | "yes" | "YES")
    )
}

fn use_delta_chunk_windowed_kernel(
    device: &Device,
    scan_mode: DeltaNetScanMode,
    sequence_length: usize,
    chunk_size: usize,
) -> bool {
    if !(matches!(scan_mode, DeltaNetScanMode::PrebatchedLocal)
        && sequence_length >= 2048
        && chunk_size <= 24)
    {
        return false;
    }

    match device.location() {
        DeviceLocation::Metal { .. } => {
            match std::env::var("CANDLE_QWEN35_DELTA_CHUNK_WINDOWED_KERNEL") {
                Ok(value)
                    if matches!(
                        value.as_str(),
                        "0" | "false" | "FALSE" | "no" | "NO" | "off" | "OFF"
                    ) =>
                {
                    false
                }
                Ok(_) => true,
                Err(_) => true,
            }
        }
        DeviceLocation::Cuda { .. } | DeviceLocation::Hip { .. } => matches!(
            std::env::var("CANDLE_QWEN35_DELTA_CHUNK_WINDOWED_KERNEL").as_deref(),
            Ok("1") | Ok("true") | Ok("TRUE") | Ok("yes") | Ok("YES")
        ),
        _ => false,
    }
}

fn use_linear_prefill_packed_kernel(device: &Device, sequence_length: usize) -> bool {
    if sequence_length < 2048 {
        return false;
    }

    match device.location() {
        DeviceLocation::Metal { .. } => {
            match std::env::var("CANDLE_QWEN35_LINEAR_PACKED_PREFILL") {
                Ok(value)
                    if matches!(
                        value.as_str(),
                        "0" | "false" | "FALSE" | "no" | "NO" | "off" | "OFF"
                    ) =>
                {
                    false
                }
                Ok(_) => true,
                Err(_) => true,
            }
        }
        DeviceLocation::Cuda { .. } => matches!(
            std::env::var("CANDLE_QWEN35_LINEAR_PACKED_PREFILL").as_deref(),
            Ok("1") | Ok("true") | Ok("TRUE") | Ok("yes") | Ok("YES")
        ),
        DeviceLocation::Hip { .. } => matches!(
            std::env::var("CANDLE_QWEN35_LINEAR_PACKED_PREFILL").as_deref(),
            Ok("1") | Ok("true") | Ok("TRUE") | Ok("yes") | Ok("YES")
        ),
        _ => false,
    }
}

fn use_full_attention_prefill_megakernel(
    device: &Device,
    q_len: usize,
    kv_len: usize,
    seqlen_offset: usize,
) -> bool {
    if q_len < 2048 || kv_len != q_len + seqlen_offset {
        return false;
    }

    match device.location() {
        DeviceLocation::Metal { .. } => {
            match std::env::var("CANDLE_QWEN35_FULL_PREFILL_MEGAKERNEL") {
                Ok(value)
                    if matches!(
                        value.as_str(),
                        "0" | "false" | "FALSE" | "no" | "NO" | "off" | "OFF"
                    ) =>
                {
                    false
                }
                Ok(_) => true,
                Err(_) => true,
            }
        }
        DeviceLocation::Cuda { .. } => matches!(
            std::env::var("CANDLE_QWEN35_FULL_PREFILL_MEGAKERNEL").as_deref(),
            Ok("1") | Ok("true") | Ok("TRUE") | Ok("yes") | Ok("YES")
        ),
        DeviceLocation::Hip { .. } => matches!(
            std::env::var("CANDLE_QWEN35_FULL_PREFILL_MEGAKERNEL").as_deref(),
            Ok("1") | Ok("true") | Ok("TRUE") | Ok("yes") | Ok("YES")
        ),
        _ => false,
    }
}

fn use_delta_chunk_scan_kernel(
    device: &Device,
    scan_mode: DeltaNetScanMode,
    sequence_length: usize,
    chunk_size: usize,
) -> bool {
    if !(matches!(scan_mode, DeltaNetScanMode::PrebatchedLocal)
        && sequence_length >= 4096
        && chunk_size <= 16)
    {
        return false;
    }

    match device.location() {
        DeviceLocation::Metal { .. } | DeviceLocation::Cuda { .. } | DeviceLocation::Hip { .. } => {
            matches!(
                std::env::var("CANDLE_QWEN35_DELTA_CHUNK_SCAN_KERNEL").as_deref(),
                Ok("1") | Ok("true") | Ok("TRUE") | Ok("yes") | Ok("YES")
            )
        }
        _ => false,
    }
}

#[derive(Debug, Clone, Copy)]
struct LinearPrefillConvPack {
    batch_size: usize,
    conv_dim: usize,
    total_len: usize,
    seq_len: usize,
    kernel_size: usize,
}

impl candle::CustomOp2 for LinearPrefillConvPack {
    fn name(&self) -> &'static str {
        "linear-prefill-conv-pack"
    }

    fn cpu_fwd(
        &self,
        _s1: &candle::CpuStorage,
        _l1: &candle::Layout,
        _s2: &candle::CpuStorage,
        _l2: &candle::Layout,
    ) -> Result<(candle::CpuStorage, candle::Shape)> {
        candle::bail!("linear-prefill-conv-pack has no cpu implementation")
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        mixed_qkv: &candle::CudaStorage,
        mixed_qkv_layout: &candle::Layout,
        weights: &candle::CudaStorage,
        weights_layout: &candle::Layout,
    ) -> Result<(candle::CudaStorage, candle::Shape)> {
        use candle::backend::BackendStorage;
        use candle::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
        use candle::cuda_backend::WrapErr;

        if !(mixed_qkv_layout.is_contiguous() && weights_layout.is_contiguous()) {
            candle::bail!("linear-prefill-conv-pack requires contiguous inputs")
        }

        let (batch_size, conv_dim, total_len) = mixed_qkv_layout.shape().dims3()?;
        let (weights_conv_dim, kernel_size) = weights_layout.shape().dims2()?;
        if batch_size != self.batch_size
            || conv_dim != self.conv_dim
            || total_len != self.total_len
            || weights_conv_dim != self.conv_dim
            || kernel_size != self.kernel_size
        {
            candle::bail!(
                "linear-prefill-conv-pack shape mismatch: mixed_qkv={:?} weights={:?} expected=({}, {}, {}, {})",
                mixed_qkv_layout.shape().dims(),
                weights_layout.shape().dims(),
                self.batch_size,
                self.conv_dim,
                self.total_len,
                self.kernel_size
            )
        }
        if total_len < self.seq_len + self.kernel_size.saturating_sub(1) {
            candle::bail!(
                "linear-prefill-conv-pack total_len {} too small for seq_len {} kernel {}",
                total_len,
                self.seq_len,
                self.kernel_size
            )
        }

        let device = mixed_qkv.device().clone();
        let output_shape = candle::Shape::from((self.batch_size, self.seq_len, self.conv_dim));
        let elem_count = output_shape.elem_count();
        let cfg = LaunchConfig::for_num_elems(elem_count as u32);

        macro_rules! launch {
            ($ty:ty, $kernel:expr) => {{
                let mixed_qkv = mixed_qkv.as_cuda_slice::<$ty>()?;
                let mixed_qkv = match mixed_qkv_layout.contiguous_offsets() {
                    Some((o1, o2)) => mixed_qkv.slice(o1..o2),
                    None => candle::bail!("linear-prefill-conv-pack requires contiguous inputs"),
                };
                let weights = weights.as_cuda_slice::<$ty>()?;
                let weights = match weights_layout.contiguous_offsets() {
                    Some((o1, o2)) => weights.slice(o1..o2),
                    None => candle::bail!("linear-prefill-conv-pack requires contiguous inputs"),
                };
                let output = unsafe { device.alloc::<$ty>(elem_count) }?;
                let func = device
                    .get_or_load_func($kernel, &candle::cuda_backend::kernels::QWEN35_DELTA)?;
                let mut builder = func.builder();
                candle::builder_arg!(
                    builder,
                    self.batch_size,
                    self.conv_dim,
                    self.total_len,
                    self.seq_len,
                    self.kernel_size
                );
                builder.arg(&mixed_qkv);
                builder.arg(&weights);
                builder.arg(&output);
                unsafe { builder.launch(cfg) }.w()?;
                let storage = candle::CudaStorage::wrap_cuda_slice(output, device.clone());
                Ok((storage, output_shape.clone()))
            }};
        }

        match mixed_qkv.dtype() {
            DType::F16 => launch!(half::f16, "linear_prefill_conv_pack_f16"),
            DType::F32 => launch!(f32, "linear_prefill_conv_pack_f32"),
            DType::BF16 => launch!(half::bf16, "linear_prefill_conv_pack_bf16"),
            other => candle::bail!("linear-prefill-conv-pack unsupported dtype {other:?}"),
        }
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        mixed_qkv: &candle::MetalStorage,
        mixed_qkv_layout: &candle::Layout,
        weights: &candle::MetalStorage,
        weights_layout: &candle::Layout,
    ) -> Result<(candle::MetalStorage, candle::Shape)> {
        use candle::backend::BackendStorage;
        use candle::MetalError;

        if !(mixed_qkv_layout.is_contiguous() && weights_layout.is_contiguous()) {
            candle::bail!("linear-prefill-conv-pack requires contiguous inputs")
        }

        let (batch_size, conv_dim, total_len) = mixed_qkv_layout.shape().dims3()?;
        let (weights_conv_dim, kernel_size) = weights_layout.shape().dims2()?;
        if batch_size != self.batch_size
            || conv_dim != self.conv_dim
            || total_len != self.total_len
            || weights_conv_dim != self.conv_dim
            || kernel_size != self.kernel_size
        {
            candle::bail!(
                "linear-prefill-conv-pack shape mismatch: mixed_qkv={:?} weights={:?} expected=({}, {}, {}, {})",
                mixed_qkv_layout.shape().dims(),
                weights_layout.shape().dims(),
                self.batch_size,
                self.conv_dim,
                self.total_len,
                self.kernel_size
            )
        }
        if total_len < self.seq_len + self.kernel_size.saturating_sub(1) {
            candle::bail!(
                "linear-prefill-conv-pack total_len {} too small for seq_len {} kernel {}",
                total_len,
                self.seq_len,
                self.kernel_size
            )
        }

        let device = mixed_qkv.device();
        let storage_dtype = mixed_qkv.dtype();
        let dtype = match storage_dtype {
            DType::F16 => candle_metal_kernels::DType::F16,
            DType::F32 => candle_metal_kernels::DType::F32,
            DType::BF16 => candle_metal_kernels::DType::BF16,
            other => candle::bail!("linear-prefill-conv-pack unsupported dtype {other:?}"),
        };
        let output_shape = candle::Shape::from((self.batch_size, self.seq_len, self.conv_dim));
        let elem_count = output_shape.elem_count();
        let output = device.new_buffer(elem_count, storage_dtype, "linear-prefill-conv-pack")?;
        let encoder = device.command_encoder()?;
        encoder.set_label("linear-prefill-conv-pack");
        let mixed_qkv = candle_metal_kernels::BufferOffset {
            buffer: mixed_qkv.buffer(),
            offset_in_bytes: mixed_qkv_layout.start_offset() * mixed_qkv.dtype().size_in_bytes(),
        };
        let weights = candle_metal_kernels::BufferOffset {
            buffer: weights.buffer(),
            offset_in_bytes: weights_layout.start_offset() * weights.dtype().size_in_bytes(),
        };
        candle_metal_kernels::call_linear_prefill_conv_pack(
            device.metal_device(),
            &encoder,
            device.kernels(),
            dtype,
            self.batch_size,
            self.conv_dim,
            self.total_len,
            self.seq_len,
            self.kernel_size,
            mixed_qkv,
            weights,
            &output,
        )
        .map_err(MetalError::from)?;
        let storage = candle::MetalStorage::new(output, device.clone(), elem_count, storage_dtype);
        Ok((storage, output_shape))
    }

    #[cfg(feature = "hip")]
    fn hip_fwd(
        &self,
        mixed_qkv: &candle::HipStorage,
        mixed_qkv_layout: &candle::Layout,
        weights: &candle::HipStorage,
        weights_layout: &candle::Layout,
    ) -> Result<(candle::HipStorage, candle::Shape)> {
        use candle::backend::BackendStorage;
        use std::ffi::c_void;

        if !(mixed_qkv_layout.is_contiguous() && weights_layout.is_contiguous()) {
            candle::bail!("linear-prefill-conv-pack requires contiguous inputs")
        }

        let (batch_size, conv_dim, total_len) = mixed_qkv_layout.shape().dims3()?;
        let (weights_conv_dim, kernel_size) = weights_layout.shape().dims2()?;
        if batch_size != self.batch_size
            || conv_dim != self.conv_dim
            || total_len != self.total_len
            || weights_conv_dim != self.conv_dim
            || kernel_size != self.kernel_size
        {
            candle::bail!(
                "linear-prefill-conv-pack shape mismatch: mixed_qkv={:?} weights={:?} expected=({}, {}, {}, {})",
                mixed_qkv_layout.shape().dims(),
                weights_layout.shape().dims(),
                self.batch_size,
                self.conv_dim,
                self.total_len,
                self.kernel_size
            )
        }
        if total_len < self.seq_len + self.kernel_size.saturating_sub(1) {
            candle::bail!(
                "linear-prefill-conv-pack total_len {} too small for seq_len {} kernel {}",
                total_len,
                self.seq_len,
                self.kernel_size
            )
        }

        let device = mixed_qkv.device().clone();
        let storage_dtype = mixed_qkv.dtype();
        let dtype_code = candle::hip::qwen35_dtype_code(storage_dtype)?;
        let output_shape = candle::Shape::from((self.batch_size, self.seq_len, self.conv_dim));
        let elem_count = output_shape.elem_count();

        macro_rules! launch {
            ($ty:ty, $zero:expr) => {{
                let mixed_qkv = mixed_qkv.cpu_storage().as_slice::<$ty>()?;
                let mixed_qkv = match mixed_qkv_layout.contiguous_offsets() {
                    Some((o1, o2)) => &mixed_qkv[o1..o2],
                    None => candle::bail!("linear-prefill-conv-pack requires contiguous inputs"),
                };
                let weights = weights.cpu_storage().as_slice::<$ty>()?;
                let weights = match weights_layout.contiguous_offsets() {
                    Some((o1, o2)) => &weights[o1..o2],
                    None => candle::bail!("linear-prefill-conv-pack requires contiguous inputs"),
                };
                let mut output = vec![$zero; elem_count];
                let status = unsafe {
                    candle::hip::ffi::qwen35_hip_linear_prefill_conv_pack(
                        dtype_code,
                        self.batch_size,
                        self.conv_dim,
                        self.total_len,
                        self.seq_len,
                        self.kernel_size,
                        mixed_qkv.as_ptr() as *const c_void,
                        weights.as_ptr() as *const c_void,
                        output.as_mut_ptr() as *mut c_void,
                    )
                };
                if status != 0 {
                    return Err(candle::hip::qwen35_error(self.name(), status));
                }
                let storage = <$ty as candle::WithDType>::to_cpu_storage_owned(output);
                Ok((
                    candle::HipStorage::wrap_cpu_storage(storage, device.clone()),
                    output_shape.clone(),
                ))
            }};
        }

        match storage_dtype {
            DType::F16 => launch!(half::f16, half::f16::from_bits(0)),
            DType::F32 => launch!(f32, 0.0f32),
            DType::BF16 => launch!(half::bf16, half::bf16::from_bits(0)),
            other => candle::bail!("linear-prefill-conv-pack unsupported dtype {other:?}"),
        }
    }
}

fn linear_prefill_conv_pack(
    mixed_qkv: &Tensor,
    weights: &Tensor,
    seq_len: usize,
    kernel_size: usize,
) -> Result<Tensor> {
    let (batch_size, conv_dim, total_len) = mixed_qkv.dims3()?;
    mixed_qkv.apply_op2_no_bwd(
        weights,
        &LinearPrefillConvPack {
            batch_size,
            conv_dim,
            total_len,
            seq_len,
            kernel_size,
        },
    )
}

#[derive(Debug, Clone, Copy)]
struct FullAttentionPrefillMegakernel {
    batch_size: usize,
    q_heads: usize,
    kv_heads: usize,
    q_len: usize,
    kv_len: usize,
    head_dim: usize,
    num_kv_groups: usize,
    scale: f32,
    seqlen_offset: usize,
}

impl candle::CustomOp3 for FullAttentionPrefillMegakernel {
    fn name(&self) -> &'static str {
        "full-attention-prefill-megakernel"
    }

    fn cpu_fwd(
        &self,
        _s1: &candle::CpuStorage,
        _l1: &candle::Layout,
        _s2: &candle::CpuStorage,
        _l2: &candle::Layout,
        _s3: &candle::CpuStorage,
        _l3: &candle::Layout,
    ) -> Result<(candle::CpuStorage, candle::Shape)> {
        candle::bail!("full-attention-prefill-megakernel has no cpu implementation")
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        query: &candle::CudaStorage,
        query_layout: &candle::Layout,
        key: &candle::CudaStorage,
        key_layout: &candle::Layout,
        value: &candle::CudaStorage,
        value_layout: &candle::Layout,
    ) -> Result<(candle::CudaStorage, candle::Shape)> {
        use candle::backend::BackendStorage;
        use candle::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
        use candle::cuda_backend::WrapErr;

        if !(query_layout.is_contiguous()
            && key_layout.is_contiguous()
            && value_layout.is_contiguous())
        {
            candle::bail!("full-attention-prefill-megakernel requires contiguous inputs")
        }

        let (batch_size, q_heads, q_len, head_dim) = query_layout.shape().dims4()?;
        let (key_batch, kv_heads, kv_len, key_head_dim) = key_layout.shape().dims4()?;
        let (value_batch, value_kv_heads, value_kv_len, value_head_dim) =
            value_layout.shape().dims4()?;
        if batch_size != self.batch_size
            || key_batch != self.batch_size
            || value_batch != self.batch_size
            || q_heads != self.q_heads
            || kv_heads != self.kv_heads
            || value_kv_heads != self.kv_heads
            || q_len != self.q_len
            || kv_len != self.kv_len
            || value_kv_len != self.kv_len
            || head_dim != self.head_dim
            || key_head_dim != self.head_dim
            || value_head_dim != self.head_dim
        {
            candle::bail!(
                "full-attention-prefill-megakernel shape mismatch: query={:?} key={:?} value={:?}",
                query_layout.shape().dims(),
                key_layout.shape().dims(),
                value_layout.shape().dims()
            )
        }

        let device = query.device().clone();
        let out_shape =
            candle::Shape::from((self.batch_size, self.q_heads, self.q_len, self.head_dim));
        let elem_count = out_shape.elem_count();
        let total_rows = self.batch_size * self.q_heads * self.q_len;
        let cfg = LaunchConfig::for_num_elems(total_rows as u32);

        macro_rules! launch {
            ($ty:ty, $kernel:expr) => {{
                let query = query.as_cuda_slice::<$ty>()?;
                let query = match query_layout.contiguous_offsets() {
                    Some((o1, o2)) => query.slice(o1..o2),
                    None => {
                        candle::bail!(
                            "full-attention-prefill-megakernel requires contiguous inputs"
                        )
                    }
                };
                let key = key.as_cuda_slice::<$ty>()?;
                let key = match key_layout.contiguous_offsets() {
                    Some((o1, o2)) => key.slice(o1..o2),
                    None => {
                        candle::bail!(
                            "full-attention-prefill-megakernel requires contiguous inputs"
                        )
                    }
                };
                let value = value.as_cuda_slice::<$ty>()?;
                let value = match value_layout.contiguous_offsets() {
                    Some((o1, o2)) => value.slice(o1..o2),
                    None => {
                        candle::bail!(
                            "full-attention-prefill-megakernel requires contiguous inputs"
                        )
                    }
                };
                let output = unsafe { device.alloc::<$ty>(elem_count) }?;
                let func = device
                    .get_or_load_func($kernel, &candle::cuda_backend::kernels::QWEN35_DELTA)?;
                let mut builder = func.builder();
                candle::builder_arg!(
                    builder,
                    self.batch_size,
                    self.q_heads,
                    self.kv_heads,
                    self.q_len,
                    self.kv_len,
                    self.head_dim,
                    self.num_kv_groups,
                    self.scale,
                    self.seqlen_offset
                );
                builder.arg(&query);
                builder.arg(&key);
                builder.arg(&value);
                builder.arg(&output);
                unsafe { builder.launch(cfg) }.w()?;
                let storage = candle::CudaStorage::wrap_cuda_slice(output, device.clone());
                Ok((storage, out_shape.clone()))
            }};
        }

        match query.dtype() {
            DType::F16 => launch!(half::f16, "full_attention_prefill_f16"),
            DType::F32 => launch!(f32, "full_attention_prefill_f32"),
            DType::BF16 => launch!(half::bf16, "full_attention_prefill_bf16"),
            other => candle::bail!("full-attention-prefill-megakernel unsupported dtype {other:?}"),
        }
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        query: &candle::MetalStorage,
        query_layout: &candle::Layout,
        key: &candle::MetalStorage,
        key_layout: &candle::Layout,
        value: &candle::MetalStorage,
        value_layout: &candle::Layout,
    ) -> Result<(candle::MetalStorage, candle::Shape)> {
        use candle::backend::BackendStorage;
        use candle::MetalError;

        if !(query_layout.is_contiguous()
            && key_layout.is_contiguous()
            && value_layout.is_contiguous())
        {
            candle::bail!("full-attention-prefill-megakernel requires contiguous inputs")
        }

        let (batch_size, q_heads, q_len, head_dim) = query_layout.shape().dims4()?;
        let (key_batch, kv_heads, kv_len, key_head_dim) = key_layout.shape().dims4()?;
        let (value_batch, value_kv_heads, value_kv_len, value_head_dim) =
            value_layout.shape().dims4()?;
        if batch_size != self.batch_size
            || key_batch != self.batch_size
            || value_batch != self.batch_size
            || q_heads != self.q_heads
            || kv_heads != self.kv_heads
            || value_kv_heads != self.kv_heads
            || q_len != self.q_len
            || kv_len != self.kv_len
            || value_kv_len != self.kv_len
            || head_dim != self.head_dim
            || key_head_dim != self.head_dim
            || value_head_dim != self.head_dim
        {
            candle::bail!(
                "full-attention-prefill-megakernel shape mismatch: query={:?} key={:?} value={:?}",
                query_layout.shape().dims(),
                key_layout.shape().dims(),
                value_layout.shape().dims()
            )
        }

        let device = query.device();
        let dtype = match query.dtype() {
            DType::F16 => candle_metal_kernels::DType::F16,
            DType::F32 => candle_metal_kernels::DType::F32,
            DType::BF16 => candle_metal_kernels::DType::BF16,
            other => candle::bail!("full-attention-prefill-megakernel unsupported dtype {other:?}"),
        };
        let out_shape =
            candle::Shape::from((self.batch_size, self.q_heads, self.q_len, self.head_dim));
        let elem_count = out_shape.elem_count();
        let output = device.new_buffer(
            elem_count,
            query.dtype(),
            "full-attention-prefill-megakernel",
        )?;
        let encoder = device.command_encoder()?;
        encoder.set_label("full-attention-prefill-megakernel");
        candle_metal_kernels::call_full_attention_prefill(
            device.metal_device(),
            &encoder,
            device.kernels(),
            dtype,
            self.batch_size,
            self.q_heads,
            self.kv_heads,
            self.q_len,
            self.kv_len,
            self.head_dim,
            self.num_kv_groups,
            self.scale,
            self.seqlen_offset,
            candle_metal_kernels::BufferOffset {
                buffer: query.buffer(),
                offset_in_bytes: query_layout.start_offset() * query.dtype().size_in_bytes(),
            },
            candle_metal_kernels::BufferOffset {
                buffer: key.buffer(),
                offset_in_bytes: key_layout.start_offset() * key.dtype().size_in_bytes(),
            },
            candle_metal_kernels::BufferOffset {
                buffer: value.buffer(),
                offset_in_bytes: value_layout.start_offset() * value.dtype().size_in_bytes(),
            },
            &output,
        )
        .map_err(MetalError::from)?;
        Ok((
            candle::MetalStorage::new(output, device.clone(), elem_count, query.dtype()),
            out_shape,
        ))
    }

    #[cfg(feature = "hip")]
    fn hip_fwd(
        &self,
        query: &candle::HipStorage,
        query_layout: &candle::Layout,
        key: &candle::HipStorage,
        key_layout: &candle::Layout,
        value: &candle::HipStorage,
        value_layout: &candle::Layout,
    ) -> Result<(candle::HipStorage, candle::Shape)> {
        use candle::backend::BackendStorage;
        use std::ffi::c_void;

        if !(query_layout.is_contiguous()
            && key_layout.is_contiguous()
            && value_layout.is_contiguous())
        {
            candle::bail!("full-attention-prefill-megakernel requires contiguous inputs")
        }

        let (batch_size, q_heads, q_len, head_dim) = query_layout.shape().dims4()?;
        let (key_batch, kv_heads, kv_len, key_head_dim) = key_layout.shape().dims4()?;
        let (value_batch, value_kv_heads, value_kv_len, value_head_dim) =
            value_layout.shape().dims4()?;
        if batch_size != self.batch_size
            || key_batch != self.batch_size
            || value_batch != self.batch_size
            || q_heads != self.q_heads
            || kv_heads != self.kv_heads
            || value_kv_heads != self.kv_heads
            || q_len != self.q_len
            || kv_len != self.kv_len
            || value_kv_len != self.kv_len
            || head_dim != self.head_dim
            || key_head_dim != self.head_dim
            || value_head_dim != self.head_dim
        {
            candle::bail!(
                "full-attention-prefill-megakernel shape mismatch: query={:?} key={:?} value={:?}",
                query_layout.shape().dims(),
                key_layout.shape().dims(),
                value_layout.shape().dims()
            )
        }

        let device = query.device().clone();
        let storage_dtype = query.dtype();
        let dtype_code = candle::hip::qwen35_dtype_code(storage_dtype)?;
        let out_shape =
            candle::Shape::from((self.batch_size, self.q_heads, self.q_len, self.head_dim));
        let elem_count = out_shape.elem_count();

        macro_rules! launch {
            ($ty:ty, $zero:expr) => {{
                let query = query.cpu_storage().as_slice::<$ty>()?;
                let query = match query_layout.contiguous_offsets() {
                    Some((o1, o2)) => &query[o1..o2],
                    None => {
                        candle::bail!(
                            "full-attention-prefill-megakernel requires contiguous inputs"
                        )
                    }
                };
                let key = key.cpu_storage().as_slice::<$ty>()?;
                let key = match key_layout.contiguous_offsets() {
                    Some((o1, o2)) => &key[o1..o2],
                    None => {
                        candle::bail!(
                            "full-attention-prefill-megakernel requires contiguous inputs"
                        )
                    }
                };
                let value = value.cpu_storage().as_slice::<$ty>()?;
                let value = match value_layout.contiguous_offsets() {
                    Some((o1, o2)) => &value[o1..o2],
                    None => {
                        candle::bail!(
                            "full-attention-prefill-megakernel requires contiguous inputs"
                        )
                    }
                };
                let mut output = vec![$zero; elem_count];
                let status = unsafe {
                    candle::hip::ffi::qwen35_hip_full_attention_prefill(
                        dtype_code,
                        self.batch_size,
                        self.q_heads,
                        self.kv_heads,
                        self.q_len,
                        self.kv_len,
                        self.head_dim,
                        self.num_kv_groups,
                        self.scale,
                        self.seqlen_offset,
                        query.as_ptr() as *const c_void,
                        key.as_ptr() as *const c_void,
                        value.as_ptr() as *const c_void,
                        output.as_mut_ptr() as *mut c_void,
                    )
                };
                if status != 0 {
                    return Err(candle::hip::qwen35_error(self.name(), status));
                }
                let storage = <$ty as candle::WithDType>::to_cpu_storage_owned(output);
                Ok((
                    candle::HipStorage::wrap_cpu_storage(storage, device.clone()),
                    out_shape.clone(),
                ))
            }};
        }

        match storage_dtype {
            DType::F16 => launch!(half::f16, half::f16::from_bits(0)),
            DType::F32 => launch!(f32, 0.0f32),
            DType::BF16 => launch!(half::bf16, half::bf16::from_bits(0)),
            other => candle::bail!("full-attention-prefill-megakernel unsupported dtype {other:?}"),
        }
    }
}

fn full_attention_prefill_megakernel(
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    num_kv_groups: usize,
    scale: f32,
    seqlen_offset: usize,
) -> Result<Tensor> {
    let (batch_size, q_heads, q_len, head_dim) = query.dims4()?;
    let (_, kv_heads, kv_len, value_head_dim) = value.dims4()?;
    if value_head_dim != head_dim {
        candle::bail!(
            "full-attention-prefill-megakernel requires matching head dims, got q={} v={}",
            head_dim,
            value_head_dim
        )
    }
    query.apply_op3_no_bwd(
        key,
        value,
        &FullAttentionPrefillMegakernel {
            batch_size,
            q_heads,
            kv_heads,
            q_len,
            kv_len,
            head_dim,
            num_kv_groups,
            scale,
            seqlen_offset,
        },
    )
}

#[derive(Debug, Clone, Copy)]
struct DeltaStateUpdate;

impl candle::CustomOp3 for DeltaStateUpdate {
    fn name(&self) -> &'static str {
        "delta-state-update"
    }

    fn cpu_fwd(
        &self,
        _s1: &candle::CpuStorage,
        _l1: &candle::Layout,
        _s2: &candle::CpuStorage,
        _l2: &candle::Layout,
        _s3: &candle::CpuStorage,
        _l3: &candle::Layout,
    ) -> Result<(candle::CpuStorage, candle::Shape)> {
        candle::bail!("delta-state-update has no cpu implementation")
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        prev_state: &candle::MetalStorage,
        prev_layout: &candle::Layout,
        weighted_key: &candle::MetalStorage,
        weighted_key_layout: &candle::Layout,
        value: &candle::MetalStorage,
        value_layout: &candle::Layout,
    ) -> Result<(candle::MetalStorage, candle::Shape)> {
        use candle::backend::BackendStorage;
        use candle::MetalError;

        if !(prev_layout.is_contiguous()
            && weighted_key_layout.is_contiguous()
            && value_layout.is_contiguous())
        {
            candle::bail!("delta-state-update requires contiguous inputs")
        }

        let (batch_heads, k_head_dim, v_head_dim) = prev_layout.shape().dims3()?;
        let (wk_batch_heads, chunk_size, wk_k_head_dim) = weighted_key_layout.shape().dims3()?;
        let (value_batch_heads, value_chunk_size, value_v_head_dim) =
            value_layout.shape().dims3()?;
        if wk_batch_heads != batch_heads
            || value_batch_heads != batch_heads
            || wk_k_head_dim != k_head_dim
            || value_v_head_dim != v_head_dim
            || value_chunk_size != chunk_size
        {
            candle::bail!(
                "delta-state-update shape mismatch: prev={:?} weighted_key={:?} value={:?}",
                prev_layout.shape().dims(),
                weighted_key_layout.shape().dims(),
                value_layout.shape().dims()
            )
        }

        let device = prev_state.device();
        let dtype = match prev_state.dtype() {
            DType::F16 => candle_metal_kernels::DType::F16,
            DType::F32 => candle_metal_kernels::DType::F32,
            DType::BF16 => candle_metal_kernels::DType::BF16,
            other => candle::bail!("delta-state-update unsupported dtype {other:?}"),
        };
        let elem_count = prev_layout.shape().elem_count();
        let output = device.new_buffer(elem_count, prev_state.dtype(), "delta-state-update")?;
        let encoder = device.command_encoder()?;
        encoder.set_label("delta-state-update");
        let prev = candle_metal_kernels::BufferOffset {
            buffer: prev_state.buffer(),
            offset_in_bytes: prev_layout.start_offset() * prev_state.dtype().size_in_bytes(),
        };
        let wk = candle_metal_kernels::BufferOffset {
            buffer: weighted_key.buffer(),
            offset_in_bytes: weighted_key_layout.start_offset()
                * weighted_key.dtype().size_in_bytes(),
        };
        let v = candle_metal_kernels::BufferOffset {
            buffer: value.buffer(),
            offset_in_bytes: value_layout.start_offset() * value.dtype().size_in_bytes(),
        };
        candle_metal_kernels::call_delta_state_update(
            device.metal_device(),
            &encoder,
            device.kernels(),
            dtype,
            batch_heads,
            chunk_size,
            k_head_dim,
            v_head_dim,
            prev,
            wk,
            v,
            &output,
        )
        .map_err(MetalError::from)?;
        let storage =
            candle::MetalStorage::new(output, device.clone(), elem_count, prev_state.dtype());
        Ok((storage, prev_layout.shape().clone()))
    }
}

fn delta_state_update(
    prev_state_scaled: &Tensor,
    weighted_key: &Tensor,
    value: &Tensor,
    use_kernel: bool,
) -> Result<Tensor> {
    if use_kernel {
        prev_state_scaled.apply_op3_no_bwd(weighted_key, value, &DeltaStateUpdate)
    } else {
        weighted_key
            .transpose(2, 1)?
            .matmul(value)?
            .broadcast_add(prev_state_scaled)
    }
}

#[derive(Debug, Clone, Copy)]
struct DeltaStateScan;

impl candle::CustomOp3 for DeltaStateScan {
    fn name(&self) -> &'static str {
        "delta-state-scan"
    }

    fn cpu_fwd(
        &self,
        _s1: &candle::CpuStorage,
        _l1: &candle::Layout,
        _s2: &candle::CpuStorage,
        _l2: &candle::Layout,
        _s3: &candle::CpuStorage,
        _l3: &candle::Layout,
    ) -> Result<(candle::CpuStorage, candle::Shape)> {
        candle::bail!("delta-state-scan has no cpu implementation")
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        initial_state: &candle::CudaStorage,
        initial_layout: &candle::Layout,
        packed_scan: &candle::CudaStorage,
        packed_layout: &candle::Layout,
        value: &candle::CudaStorage,
        value_layout: &candle::Layout,
    ) -> Result<(candle::CudaStorage, candle::Shape)> {
        use candle::backend::BackendStorage;
        use candle::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
        use candle::cuda_backend::WrapErr;

        if !(initial_layout.is_contiguous()
            && packed_layout.is_contiguous()
            && value_layout.is_contiguous())
        {
            candle::bail!("delta-state-scan requires contiguous inputs")
        }

        let (batch_heads, k_head_dim, v_head_dim) = initial_layout.shape().dims3()?;
        let (packed_bh, num_chunks, chunk_size, packed_width) = packed_layout.shape().dims4()?;
        let (value_bh, value_num_chunks, value_chunk_size, value_v_head_dim) =
            value_layout.shape().dims4()?;
        if packed_bh != batch_heads
            || value_bh != batch_heads
            || value_num_chunks != num_chunks
            || value_chunk_size != chunk_size
            || value_v_head_dim != v_head_dim
            || packed_width != 2 * k_head_dim + 1
        {
            candle::bail!(
                "delta-state-scan shape mismatch: initial={:?} packed={:?} value={:?}",
                initial_layout.shape().dims(),
                packed_layout.shape().dims(),
                value_layout.shape().dims()
            )
        }

        let device = initial_state.device().clone();
        let out_shape =
            candle::Shape::from_dims(&[batch_heads, num_chunks + 1, k_head_dim, v_head_dim]);
        let elem_count = out_shape.elem_count();
        let total_threads = batch_heads * v_head_dim;
        let cfg = LaunchConfig::for_num_elems(total_threads as u32);

        macro_rules! launch {
            ($ty:ty, $kernel:expr) => {{
                let initial_state = initial_state.as_cuda_slice::<$ty>()?;
                let initial_state = match initial_layout.contiguous_offsets() {
                    Some((o1, o2)) => initial_state.slice(o1..o2),
                    None => candle::bail!("delta-state-scan requires contiguous inputs"),
                };
                let packed_scan = packed_scan.as_cuda_slice::<$ty>()?;
                let packed_scan = match packed_layout.contiguous_offsets() {
                    Some((o1, o2)) => packed_scan.slice(o1..o2),
                    None => candle::bail!("delta-state-scan requires contiguous inputs"),
                };
                let value = value.as_cuda_slice::<$ty>()?;
                let value = match value_layout.contiguous_offsets() {
                    Some((o1, o2)) => value.slice(o1..o2),
                    None => candle::bail!("delta-state-scan requires contiguous inputs"),
                };
                let output = unsafe { device.alloc::<$ty>(elem_count) }?;
                let func = device
                    .get_or_load_func($kernel, &candle::cuda_backend::kernels::QWEN35_DELTA)?;
                let mut builder = func.builder();
                candle::builder_arg!(
                    builder,
                    batch_heads,
                    num_chunks,
                    chunk_size,
                    k_head_dim,
                    v_head_dim
                );
                builder.arg(&initial_state);
                builder.arg(&packed_scan);
                builder.arg(&value);
                builder.arg(&output);
                unsafe { builder.launch(cfg) }.w()?;
                let storage = candle::CudaStorage::wrap_cuda_slice(output, device.clone());
                Ok((storage, out_shape.clone()))
            }};
        }

        match initial_state.dtype() {
            DType::F16 => launch!(half::f16, "delta_state_scan_f16"),
            DType::F32 => launch!(f32, "delta_state_scan_f32"),
            DType::BF16 => launch!(half::bf16, "delta_state_scan_bf16"),
            other => candle::bail!("delta-state-scan unsupported dtype {other:?}"),
        }
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        initial_state: &candle::MetalStorage,
        initial_layout: &candle::Layout,
        packed_scan: &candle::MetalStorage,
        packed_layout: &candle::Layout,
        value: &candle::MetalStorage,
        value_layout: &candle::Layout,
    ) -> Result<(candle::MetalStorage, candle::Shape)> {
        use candle::backend::BackendStorage;
        use candle::MetalError;

        if !(initial_layout.is_contiguous()
            && packed_layout.is_contiguous()
            && value_layout.is_contiguous())
        {
            candle::bail!("delta-state-scan requires contiguous inputs")
        }

        let (batch_heads, k_head_dim, v_head_dim) = initial_layout.shape().dims3()?;
        let (packed_bh, num_chunks, chunk_size, packed_width) = packed_layout.shape().dims4()?;
        let (value_bh, value_num_chunks, value_chunk_size, value_v_head_dim) =
            value_layout.shape().dims4()?;
        if packed_bh != batch_heads
            || value_bh != batch_heads
            || value_num_chunks != num_chunks
            || value_chunk_size != chunk_size
            || value_v_head_dim != v_head_dim
            || packed_width != 2 * k_head_dim + 1
        {
            candle::bail!(
                "delta-state-scan shape mismatch: initial={:?} packed={:?} value={:?}",
                initial_layout.shape().dims(),
                packed_layout.shape().dims(),
                value_layout.shape().dims()
            )
        }

        let device = initial_state.device();
        let dtype = match initial_state.dtype() {
            DType::F16 => candle_metal_kernels::DType::F16,
            DType::F32 => candle_metal_kernels::DType::F32,
            DType::BF16 => candle_metal_kernels::DType::BF16,
            other => candle::bail!("delta-state-scan unsupported dtype {other:?}"),
        };
        let out_shape =
            candle::Shape::from_dims(&[batch_heads, num_chunks + 1, k_head_dim, v_head_dim]);
        let elem_count = out_shape.elem_count();
        let output = device.new_buffer(elem_count, initial_state.dtype(), "delta-state-scan")?;
        let encoder = device.command_encoder()?;
        encoder.set_label("delta-state-scan");
        let initial = candle_metal_kernels::BufferOffset {
            buffer: initial_state.buffer(),
            offset_in_bytes: initial_layout.start_offset() * initial_state.dtype().size_in_bytes(),
        };
        let packed = candle_metal_kernels::BufferOffset {
            buffer: packed_scan.buffer(),
            offset_in_bytes: packed_layout.start_offset() * packed_scan.dtype().size_in_bytes(),
        };
        let v = candle_metal_kernels::BufferOffset {
            buffer: value.buffer(),
            offset_in_bytes: value_layout.start_offset() * value.dtype().size_in_bytes(),
        };
        candle_metal_kernels::call_delta_state_scan(
            device.metal_device(),
            &encoder,
            device.kernels(),
            dtype,
            batch_heads,
            num_chunks,
            chunk_size,
            k_head_dim,
            v_head_dim,
            initial,
            packed,
            v,
            &output,
        )
        .map_err(MetalError::from)?;
        let storage =
            candle::MetalStorage::new(output, device.clone(), elem_count, initial_state.dtype());
        Ok((storage, out_shape))
    }

    #[cfg(feature = "hip")]
    fn hip_fwd(
        &self,
        initial_state: &candle::HipStorage,
        initial_layout: &candle::Layout,
        packed_scan: &candle::HipStorage,
        packed_layout: &candle::Layout,
        value: &candle::HipStorage,
        value_layout: &candle::Layout,
    ) -> Result<(candle::HipStorage, candle::Shape)> {
        use candle::backend::BackendStorage;
        use std::ffi::c_void;

        if !(initial_layout.is_contiguous()
            && packed_layout.is_contiguous()
            && value_layout.is_contiguous())
        {
            candle::bail!("delta-state-scan requires contiguous inputs")
        }

        let (batch_heads, k_head_dim, v_head_dim) = initial_layout.shape().dims3()?;
        let (packed_bh, num_chunks, chunk_size, packed_width) = packed_layout.shape().dims4()?;
        let (value_bh, value_num_chunks, value_chunk_size, value_v_head_dim) =
            value_layout.shape().dims4()?;
        if packed_bh != batch_heads
            || value_bh != batch_heads
            || value_num_chunks != num_chunks
            || value_chunk_size != chunk_size
            || value_v_head_dim != v_head_dim
            || packed_width != 2 * k_head_dim + 1
        {
            candle::bail!(
                "delta-state-scan shape mismatch: initial={:?} packed={:?} value={:?}",
                initial_layout.shape().dims(),
                packed_layout.shape().dims(),
                value_layout.shape().dims()
            )
        }

        let device = initial_state.device().clone();
        let storage_dtype = initial_state.dtype();
        let dtype_code = candle::hip::qwen35_dtype_code(storage_dtype)?;
        let out_shape =
            candle::Shape::from_dims(&[batch_heads, num_chunks + 1, k_head_dim, v_head_dim]);
        let elem_count = out_shape.elem_count();

        macro_rules! launch {
            ($ty:ty, $zero:expr) => {{
                let initial_state = initial_state.cpu_storage().as_slice::<$ty>()?;
                let initial_state = match initial_layout.contiguous_offsets() {
                    Some((o1, o2)) => &initial_state[o1..o2],
                    None => candle::bail!("delta-state-scan requires contiguous inputs"),
                };
                let packed_scan = packed_scan.cpu_storage().as_slice::<$ty>()?;
                let packed_scan = match packed_layout.contiguous_offsets() {
                    Some((o1, o2)) => &packed_scan[o1..o2],
                    None => candle::bail!("delta-state-scan requires contiguous inputs"),
                };
                let value = value.cpu_storage().as_slice::<$ty>()?;
                let value = match value_layout.contiguous_offsets() {
                    Some((o1, o2)) => &value[o1..o2],
                    None => candle::bail!("delta-state-scan requires contiguous inputs"),
                };
                let mut output = vec![$zero; elem_count];
                let status = unsafe {
                    candle::hip::ffi::qwen35_hip_delta_state_scan(
                        dtype_code,
                        batch_heads,
                        num_chunks,
                        chunk_size,
                        k_head_dim,
                        v_head_dim,
                        initial_state.as_ptr() as *const c_void,
                        packed_scan.as_ptr() as *const c_void,
                        value.as_ptr() as *const c_void,
                        output.as_mut_ptr() as *mut c_void,
                    )
                };
                if status != 0 {
                    return Err(candle::hip::qwen35_error(self.name(), status));
                }
                let storage = <$ty as candle::WithDType>::to_cpu_storage_owned(output);
                Ok((
                    candle::HipStorage::wrap_cpu_storage(storage, device.clone()),
                    out_shape.clone(),
                ))
            }};
        }

        match storage_dtype {
            DType::F16 => launch!(half::f16, half::f16::from_bits(0)),
            DType::F32 => launch!(f32, 0.0f32),
            DType::BF16 => launch!(half::bf16, half::bf16::from_bits(0)),
            other => candle::bail!("delta-state-scan unsupported dtype {other:?}"),
        }
    }
}

fn delta_state_scan(
    initial_state: &Tensor,
    packed_scan: &Tensor,
    value: &Tensor,
) -> Result<Tensor> {
    initial_state.apply_op3_no_bwd(packed_scan, value, &DeltaStateScan)
}

#[derive(Debug, Clone, Copy)]
struct DeltaChunkFused;

impl candle::CustomOp3 for DeltaChunkFused {
    fn name(&self) -> &'static str {
        "delta-chunk-fused"
    }

    fn cpu_fwd(
        &self,
        _s1: &candle::CpuStorage,
        _l1: &candle::Layout,
        _s2: &candle::CpuStorage,
        _l2: &candle::Layout,
        _s3: &candle::CpuStorage,
        _l3: &candle::Layout,
    ) -> Result<(candle::CpuStorage, candle::Shape)> {
        candle::bail!("delta-chunk-fused has no cpu implementation")
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        prev_state: &candle::CudaStorage,
        prev_layout: &candle::Layout,
        packed_chunk: &candle::CudaStorage,
        packed_layout: &candle::Layout,
        value: &candle::CudaStorage,
        value_layout: &candle::Layout,
    ) -> Result<(candle::CudaStorage, candle::Shape)> {
        use candle::backend::BackendStorage;
        use candle::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
        use candle::cuda_backend::WrapErr;

        if !(prev_layout.is_contiguous()
            && packed_layout.is_contiguous()
            && value_layout.is_contiguous())
        {
            candle::bail!("delta-chunk-fused requires contiguous inputs")
        }

        let (batch_heads, k_head_dim, v_head_dim) = prev_layout.shape().dims3()?;
        let (packed_bh, chunk_size, packed_width) = packed_layout.shape().dims3()?;
        let (value_bh, value_chunk_size, value_v_head_dim) = value_layout.shape().dims3()?;
        if packed_bh != batch_heads
            || value_bh != batch_heads
            || value_chunk_size != chunk_size
            || value_v_head_dim != v_head_dim
            || packed_width != 3 * k_head_dim + 1
        {
            candle::bail!(
                "delta-chunk-fused shape mismatch: prev={:?} packed={:?} value={:?}",
                prev_layout.shape().dims(),
                packed_layout.shape().dims(),
                value_layout.shape().dims()
            )
        }

        let device = prev_state.device().clone();
        let out_shape =
            candle::Shape::from_dims(&[batch_heads, 2 * chunk_size + k_head_dim, v_head_dim]);
        let elem_count = out_shape.elem_count();
        let total_threads = batch_heads * v_head_dim;
        let cfg = LaunchConfig::for_num_elems(total_threads as u32);

        macro_rules! launch {
            ($ty:ty, $kernel:expr) => {{
                let prev_state = prev_state.as_cuda_slice::<$ty>()?;
                let prev_state = match prev_layout.contiguous_offsets() {
                    Some((o1, o2)) => prev_state.slice(o1..o2),
                    None => candle::bail!("delta-chunk-fused requires contiguous inputs"),
                };
                let packed_chunk = packed_chunk.as_cuda_slice::<$ty>()?;
                let packed_chunk = match packed_layout.contiguous_offsets() {
                    Some((o1, o2)) => packed_chunk.slice(o1..o2),
                    None => candle::bail!("delta-chunk-fused requires contiguous inputs"),
                };
                let value = value.as_cuda_slice::<$ty>()?;
                let value = match value_layout.contiguous_offsets() {
                    Some((o1, o2)) => value.slice(o1..o2),
                    None => candle::bail!("delta-chunk-fused requires contiguous inputs"),
                };
                let output = unsafe { device.alloc::<$ty>(elem_count) }?;
                let func = device
                    .get_or_load_func($kernel, &candle::cuda_backend::kernels::QWEN35_DELTA)?;
                let mut builder = func.builder();
                candle::builder_arg!(builder, batch_heads, chunk_size, k_head_dim, v_head_dim);
                builder.arg(&prev_state);
                builder.arg(&packed_chunk);
                builder.arg(&value);
                builder.arg(&output);
                unsafe { builder.launch(cfg) }.w()?;
                let storage = candle::CudaStorage::wrap_cuda_slice(output, device.clone());
                Ok((storage, out_shape.clone()))
            }};
        }

        match prev_state.dtype() {
            DType::F16 => launch!(half::f16, "delta_chunk_fused_f16"),
            DType::F32 => launch!(f32, "delta_chunk_fused_f32"),
            DType::BF16 => launch!(half::bf16, "delta_chunk_fused_bf16"),
            other => candle::bail!("delta-chunk-fused unsupported dtype {other:?}"),
        }
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        prev_state: &candle::MetalStorage,
        prev_layout: &candle::Layout,
        packed_chunk: &candle::MetalStorage,
        packed_layout: &candle::Layout,
        value: &candle::MetalStorage,
        value_layout: &candle::Layout,
    ) -> Result<(candle::MetalStorage, candle::Shape)> {
        use candle::backend::BackendStorage;
        use candle::MetalError;

        if !(prev_layout.is_contiguous()
            && packed_layout.is_contiguous()
            && value_layout.is_contiguous())
        {
            candle::bail!("delta-chunk-fused requires contiguous inputs")
        }

        let (batch_heads, k_head_dim, v_head_dim) = prev_layout.shape().dims3()?;
        let (packed_bh, chunk_size, packed_width) = packed_layout.shape().dims3()?;
        let (value_bh, value_chunk_size, value_v_head_dim) = value_layout.shape().dims3()?;
        if packed_bh != batch_heads
            || value_bh != batch_heads
            || value_chunk_size != chunk_size
            || value_v_head_dim != v_head_dim
            || packed_width != 3 * k_head_dim + 1
        {
            candle::bail!(
                "delta-chunk-fused shape mismatch: prev={:?} packed={:?} value={:?}",
                prev_layout.shape().dims(),
                packed_layout.shape().dims(),
                value_layout.shape().dims()
            )
        }

        let device = prev_state.device();
        let dtype = match prev_state.dtype() {
            DType::F16 => candle_metal_kernels::DType::F16,
            DType::F32 => candle_metal_kernels::DType::F32,
            DType::BF16 => candle_metal_kernels::DType::BF16,
            other => candle::bail!("delta-chunk-fused unsupported dtype {other:?}"),
        };
        let out_shape =
            candle::Shape::from_dims(&[batch_heads, 2 * chunk_size + k_head_dim, v_head_dim]);
        let elem_count = out_shape.elem_count();
        let output = device.new_buffer(elem_count, prev_state.dtype(), "delta-chunk-fused")?;
        let encoder = device.command_encoder()?;
        encoder.set_label("delta-chunk-fused");
        let prev = candle_metal_kernels::BufferOffset {
            buffer: prev_state.buffer(),
            offset_in_bytes: prev_layout.start_offset() * prev_state.dtype().size_in_bytes(),
        };
        let packed = candle_metal_kernels::BufferOffset {
            buffer: packed_chunk.buffer(),
            offset_in_bytes: packed_layout.start_offset() * packed_chunk.dtype().size_in_bytes(),
        };
        let v = candle_metal_kernels::BufferOffset {
            buffer: value.buffer(),
            offset_in_bytes: value_layout.start_offset() * value.dtype().size_in_bytes(),
        };
        candle_metal_kernels::call_delta_chunk_fused(
            device.metal_device(),
            &encoder,
            device.kernels(),
            dtype,
            batch_heads,
            chunk_size,
            k_head_dim,
            v_head_dim,
            prev,
            packed,
            v,
            &output,
        )
        .map_err(MetalError::from)?;
        let storage =
            candle::MetalStorage::new(output, device.clone(), elem_count, prev_state.dtype());
        Ok((storage, out_shape))
    }

    #[cfg(feature = "hip")]
    fn hip_fwd(
        &self,
        prev_state: &candle::HipStorage,
        prev_layout: &candle::Layout,
        packed_chunk: &candle::HipStorage,
        packed_layout: &candle::Layout,
        value: &candle::HipStorage,
        value_layout: &candle::Layout,
    ) -> Result<(candle::HipStorage, candle::Shape)> {
        use candle::backend::BackendStorage;
        use std::ffi::c_void;

        if !(prev_layout.is_contiguous()
            && packed_layout.is_contiguous()
            && value_layout.is_contiguous())
        {
            candle::bail!("delta-chunk-fused requires contiguous inputs")
        }

        let (batch_heads, k_head_dim, v_head_dim) = prev_layout.shape().dims3()?;
        let (packed_bh, chunk_size, packed_width) = packed_layout.shape().dims3()?;
        let (value_bh, value_chunk_size, value_v_head_dim) = value_layout.shape().dims3()?;
        if packed_bh != batch_heads
            || value_bh != batch_heads
            || value_chunk_size != chunk_size
            || value_v_head_dim != v_head_dim
            || packed_width != 3 * k_head_dim + 1
        {
            candle::bail!(
                "delta-chunk-fused shape mismatch: prev={:?} packed={:?} value={:?}",
                prev_layout.shape().dims(),
                packed_layout.shape().dims(),
                value_layout.shape().dims()
            )
        }

        let device = prev_state.device().clone();
        let storage_dtype = prev_state.dtype();
        let dtype_code = candle::hip::qwen35_dtype_code(storage_dtype)?;
        let out_shape =
            candle::Shape::from_dims(&[batch_heads, 2 * chunk_size + k_head_dim, v_head_dim]);
        let elem_count = out_shape.elem_count();

        macro_rules! launch {
            ($ty:ty, $zero:expr) => {{
                let prev_state = prev_state.cpu_storage().as_slice::<$ty>()?;
                let prev_state = match prev_layout.contiguous_offsets() {
                    Some((o1, o2)) => &prev_state[o1..o2],
                    None => candle::bail!("delta-chunk-fused requires contiguous inputs"),
                };
                let packed_chunk = packed_chunk.cpu_storage().as_slice::<$ty>()?;
                let packed_chunk = match packed_layout.contiguous_offsets() {
                    Some((o1, o2)) => &packed_chunk[o1..o2],
                    None => candle::bail!("delta-chunk-fused requires contiguous inputs"),
                };
                let value = value.cpu_storage().as_slice::<$ty>()?;
                let value = match value_layout.contiguous_offsets() {
                    Some((o1, o2)) => &value[o1..o2],
                    None => candle::bail!("delta-chunk-fused requires contiguous inputs"),
                };
                let mut output = vec![$zero; elem_count];
                let status = unsafe {
                    candle::hip::ffi::qwen35_hip_delta_chunk_fused(
                        dtype_code,
                        batch_heads,
                        chunk_size,
                        k_head_dim,
                        v_head_dim,
                        prev_state.as_ptr() as *const c_void,
                        packed_chunk.as_ptr() as *const c_void,
                        value.as_ptr() as *const c_void,
                        output.as_mut_ptr() as *mut c_void,
                    )
                };
                if status != 0 {
                    return Err(candle::hip::qwen35_error(self.name(), status));
                }
                let storage = <$ty as candle::WithDType>::to_cpu_storage_owned(output);
                Ok((
                    candle::HipStorage::wrap_cpu_storage(storage, device.clone()),
                    out_shape.clone(),
                ))
            }};
        }

        match storage_dtype {
            DType::F16 => launch!(half::f16, half::f16::from_bits(0)),
            DType::F32 => launch!(f32, 0.0f32),
            DType::BF16 => launch!(half::bf16, half::bf16::from_bits(0)),
            other => candle::bail!("delta-chunk-fused unsupported dtype {other:?}"),
        }
    }
}

fn delta_chunk_fused(prev_state: &Tensor, packed_chunk: &Tensor, value: &Tensor) -> Result<Tensor> {
    prev_state.apply_op3_no_bwd(packed_chunk, value, &DeltaChunkFused)
}

#[derive(Debug, Clone, Copy)]
struct DeltaRecurrentPrefill;

impl candle::CustomOp6 for DeltaRecurrentPrefill {
    fn name(&self) -> &'static str {
        "delta-recurrent-prefill"
    }

    fn cpu_fwd(
        &self,
        _s1: &candle::CpuStorage,
        _l1: &candle::Layout,
        _s2: &candle::CpuStorage,
        _l2: &candle::Layout,
        _s3: &candle::CpuStorage,
        _l3: &candle::Layout,
        _s4: &candle::CpuStorage,
        _l4: &candle::Layout,
        _s5: &candle::CpuStorage,
        _l5: &candle::Layout,
        _s6: &candle::CpuStorage,
        _l6: &candle::Layout,
    ) -> Result<(candle::CpuStorage, candle::Shape)> {
        candle::bail!("delta-recurrent-prefill has no cpu implementation")
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        initial_state: &candle::CudaStorage,
        initial_layout: &candle::Layout,
        query: &candle::CudaStorage,
        query_layout: &candle::Layout,
        key: &candle::CudaStorage,
        key_layout: &candle::Layout,
        value: &candle::CudaStorage,
        value_layout: &candle::Layout,
        beta: &candle::CudaStorage,
        beta_layout: &candle::Layout,
        g: &candle::CudaStorage,
        g_layout: &candle::Layout,
    ) -> Result<(candle::CudaStorage, candle::Shape)> {
        use candle::backend::BackendStorage;
        use candle::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
        use candle::cuda_backend::WrapErr;

        if !(initial_layout.is_contiguous()
            && query_layout.is_contiguous()
            && key_layout.is_contiguous()
            && value_layout.is_contiguous()
            && beta_layout.is_contiguous()
            && g_layout.is_contiguous())
        {
            candle::bail!("delta-recurrent-prefill requires contiguous inputs")
        }

        let (batch_heads, k_head_dim, v_head_dim) = initial_layout.shape().dims3()?;
        let (query_bh, seq_len, query_k) = query_layout.shape().dims3()?;
        let (key_bh, key_seq, key_k) = key_layout.shape().dims3()?;
        let (value_bh, value_seq, value_v) = value_layout.shape().dims3()?;
        let (beta_bh, beta_seq) = beta_layout.shape().dims2()?;
        let (g_bh, g_seq) = g_layout.shape().dims2()?;
        if query_bh != batch_heads
            || key_bh != batch_heads
            || value_bh != batch_heads
            || beta_bh != batch_heads
            || g_bh != batch_heads
            || key_seq != seq_len
            || value_seq != seq_len
            || beta_seq != seq_len
            || g_seq != seq_len
            || query_k != k_head_dim
            || key_k != k_head_dim
            || value_v != v_head_dim
        {
            candle::bail!(
                "delta-recurrent-prefill shape mismatch: initial={:?} query={:?} key={:?} value={:?} beta={:?} g={:?}",
                initial_layout.shape().dims(),
                query_layout.shape().dims(),
                key_layout.shape().dims(),
                value_layout.shape().dims(),
                beta_layout.shape().dims(),
                g_layout.shape().dims()
            )
        }

        let device = initial_state.device().clone();
        let out_shape = candle::Shape::from_dims(&[batch_heads, seq_len + k_head_dim, v_head_dim]);
        let elem_count = out_shape.elem_count();
        let total_threads = batch_heads * v_head_dim;
        let cfg = LaunchConfig::for_num_elems(total_threads as u32);

        macro_rules! launch {
            ($ty:ty, $kernel:expr) => {{
                let initial_state = initial_state.as_cuda_slice::<$ty>()?;
                let initial_state = match initial_layout.contiguous_offsets() {
                    Some((o1, o2)) => initial_state.slice(o1..o2),
                    None => candle::bail!("delta-recurrent-prefill requires contiguous inputs"),
                };
                let query = query.as_cuda_slice::<$ty>()?;
                let query = match query_layout.contiguous_offsets() {
                    Some((o1, o2)) => query.slice(o1..o2),
                    None => candle::bail!("delta-recurrent-prefill requires contiguous inputs"),
                };
                let key = key.as_cuda_slice::<$ty>()?;
                let key = match key_layout.contiguous_offsets() {
                    Some((o1, o2)) => key.slice(o1..o2),
                    None => candle::bail!("delta-recurrent-prefill requires contiguous inputs"),
                };
                let value = value.as_cuda_slice::<$ty>()?;
                let value = match value_layout.contiguous_offsets() {
                    Some((o1, o2)) => value.slice(o1..o2),
                    None => candle::bail!("delta-recurrent-prefill requires contiguous inputs"),
                };
                let beta = beta.as_cuda_slice::<$ty>()?;
                let beta = match beta_layout.contiguous_offsets() {
                    Some((o1, o2)) => beta.slice(o1..o2),
                    None => candle::bail!("delta-recurrent-prefill requires contiguous inputs"),
                };
                let g = g.as_cuda_slice::<$ty>()?;
                let g = match g_layout.contiguous_offsets() {
                    Some((o1, o2)) => g.slice(o1..o2),
                    None => candle::bail!("delta-recurrent-prefill requires contiguous inputs"),
                };
                let output = unsafe { device.alloc::<$ty>(elem_count) }?;
                let func = device
                    .get_or_load_func($kernel, &candle::cuda_backend::kernels::QWEN35_DELTA)?;
                let mut builder = func.builder();
                candle::builder_arg!(builder, batch_heads, seq_len, k_head_dim, v_head_dim);
                builder.arg(&initial_state);
                builder.arg(&query);
                builder.arg(&key);
                builder.arg(&value);
                builder.arg(&beta);
                builder.arg(&g);
                builder.arg(&output);
                unsafe { builder.launch(cfg) }.w()?;
                let storage = candle::CudaStorage::wrap_cuda_slice(output, device.clone());
                Ok((storage, out_shape.clone()))
            }};
        }

        match initial_state.dtype() {
            DType::F16 => launch!(half::f16, "delta_recurrent_prefill_f16"),
            DType::F32 => launch!(f32, "delta_recurrent_prefill_f32"),
            DType::BF16 => launch!(half::bf16, "delta_recurrent_prefill_bf16"),
            other => candle::bail!("delta-recurrent-prefill unsupported dtype {other:?}"),
        }
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        initial_state: &candle::MetalStorage,
        initial_layout: &candle::Layout,
        query: &candle::MetalStorage,
        query_layout: &candle::Layout,
        key: &candle::MetalStorage,
        key_layout: &candle::Layout,
        value: &candle::MetalStorage,
        value_layout: &candle::Layout,
        beta: &candle::MetalStorage,
        beta_layout: &candle::Layout,
        g: &candle::MetalStorage,
        g_layout: &candle::Layout,
    ) -> Result<(candle::MetalStorage, candle::Shape)> {
        use candle::backend::BackendStorage;
        use candle::MetalError;

        if !(initial_layout.is_contiguous()
            && query_layout.is_contiguous()
            && key_layout.is_contiguous()
            && value_layout.is_contiguous()
            && beta_layout.is_contiguous()
            && g_layout.is_contiguous())
        {
            candle::bail!("delta-recurrent-prefill requires contiguous inputs")
        }

        let (batch_heads, k_head_dim, v_head_dim) = initial_layout.shape().dims3()?;
        let (query_bh, seq_len, query_k) = query_layout.shape().dims3()?;
        let (key_bh, key_seq, key_k) = key_layout.shape().dims3()?;
        let (value_bh, value_seq, value_v) = value_layout.shape().dims3()?;
        let (beta_bh, beta_seq) = beta_layout.shape().dims2()?;
        let (g_bh, g_seq) = g_layout.shape().dims2()?;
        if query_bh != batch_heads
            || key_bh != batch_heads
            || value_bh != batch_heads
            || beta_bh != batch_heads
            || g_bh != batch_heads
            || key_seq != seq_len
            || value_seq != seq_len
            || beta_seq != seq_len
            || g_seq != seq_len
            || query_k != k_head_dim
            || key_k != k_head_dim
            || value_v != v_head_dim
        {
            candle::bail!(
                "delta-recurrent-prefill shape mismatch: initial={:?} query={:?} key={:?} value={:?} beta={:?} g={:?}",
                initial_layout.shape().dims(),
                query_layout.shape().dims(),
                key_layout.shape().dims(),
                value_layout.shape().dims(),
                beta_layout.shape().dims(),
                g_layout.shape().dims()
            )
        }

        let device = initial_state.device();
        let dtype = match initial_state.dtype() {
            DType::F16 => candle_metal_kernels::DType::F16,
            DType::F32 => candle_metal_kernels::DType::F32,
            DType::BF16 => candle_metal_kernels::DType::BF16,
            other => candle::bail!("delta-recurrent-prefill unsupported dtype {other:?}"),
        };
        let out_shape = candle::Shape::from_dims(&[batch_heads, seq_len + k_head_dim, v_head_dim]);
        let elem_count = out_shape.elem_count();
        let output =
            device.new_buffer(elem_count, initial_state.dtype(), "delta-recurrent-prefill")?;
        let encoder = device.command_encoder()?;
        encoder.set_label("delta-recurrent-prefill");
        let initial = candle_metal_kernels::BufferOffset {
            buffer: initial_state.buffer(),
            offset_in_bytes: initial_layout.start_offset() * initial_state.dtype().size_in_bytes(),
        };
        let query = candle_metal_kernels::BufferOffset {
            buffer: query.buffer(),
            offset_in_bytes: query_layout.start_offset() * query.dtype().size_in_bytes(),
        };
        let key = candle_metal_kernels::BufferOffset {
            buffer: key.buffer(),
            offset_in_bytes: key_layout.start_offset() * key.dtype().size_in_bytes(),
        };
        let value = candle_metal_kernels::BufferOffset {
            buffer: value.buffer(),
            offset_in_bytes: value_layout.start_offset() * value.dtype().size_in_bytes(),
        };
        let beta = candle_metal_kernels::BufferOffset {
            buffer: beta.buffer(),
            offset_in_bytes: beta_layout.start_offset() * beta.dtype().size_in_bytes(),
        };
        let g = candle_metal_kernels::BufferOffset {
            buffer: g.buffer(),
            offset_in_bytes: g_layout.start_offset() * g.dtype().size_in_bytes(),
        };
        candle_metal_kernels::call_delta_recurrent_prefill(
            device.metal_device(),
            &encoder,
            device.kernels(),
            dtype,
            batch_heads,
            seq_len,
            k_head_dim,
            v_head_dim,
            initial,
            query,
            key,
            value,
            beta,
            g,
            &output,
        )
        .map_err(MetalError::from)?;
        let storage =
            candle::MetalStorage::new(output, device.clone(), elem_count, initial_state.dtype());
        Ok((storage, out_shape))
    }

    #[cfg(feature = "hip")]
    fn hip_fwd(
        &self,
        initial_state: &candle::HipStorage,
        initial_layout: &candle::Layout,
        query: &candle::HipStorage,
        query_layout: &candle::Layout,
        key: &candle::HipStorage,
        key_layout: &candle::Layout,
        value: &candle::HipStorage,
        value_layout: &candle::Layout,
        beta: &candle::HipStorage,
        beta_layout: &candle::Layout,
        g: &candle::HipStorage,
        g_layout: &candle::Layout,
    ) -> Result<(candle::HipStorage, candle::Shape)> {
        use candle::backend::BackendStorage;
        use std::ffi::c_void;

        if !(initial_layout.is_contiguous()
            && query_layout.is_contiguous()
            && key_layout.is_contiguous()
            && value_layout.is_contiguous()
            && beta_layout.is_contiguous()
            && g_layout.is_contiguous())
        {
            candle::bail!("delta-recurrent-prefill requires contiguous inputs")
        }

        let (batch_heads, k_head_dim, v_head_dim) = initial_layout.shape().dims3()?;
        let (query_bh, seq_len, query_k) = query_layout.shape().dims3()?;
        let (key_bh, key_seq, key_k) = key_layout.shape().dims3()?;
        let (value_bh, value_seq, value_v) = value_layout.shape().dims3()?;
        let (beta_bh, beta_seq) = beta_layout.shape().dims2()?;
        let (g_bh, g_seq) = g_layout.shape().dims2()?;
        if query_bh != batch_heads
            || key_bh != batch_heads
            || value_bh != batch_heads
            || beta_bh != batch_heads
            || g_bh != batch_heads
            || key_seq != seq_len
            || value_seq != seq_len
            || beta_seq != seq_len
            || g_seq != seq_len
            || query_k != k_head_dim
            || key_k != k_head_dim
            || value_v != v_head_dim
        {
            candle::bail!(
                "delta-recurrent-prefill shape mismatch: initial={:?} query={:?} key={:?} value={:?} beta={:?} g={:?}",
                initial_layout.shape().dims(),
                query_layout.shape().dims(),
                key_layout.shape().dims(),
                value_layout.shape().dims(),
                beta_layout.shape().dims(),
                g_layout.shape().dims()
            )
        }

        let device = initial_state.device().clone();
        let storage_dtype = initial_state.dtype();
        let dtype_code = candle::hip::qwen35_dtype_code(storage_dtype)?;
        let out_shape = candle::Shape::from_dims(&[batch_heads, seq_len + k_head_dim, v_head_dim]);
        let elem_count = out_shape.elem_count();

        macro_rules! launch {
            ($ty:ty, $zero:expr) => {{
                let initial_state = initial_state.cpu_storage().as_slice::<$ty>()?;
                let initial_state = match initial_layout.contiguous_offsets() {
                    Some((o1, o2)) => &initial_state[o1..o2],
                    None => candle::bail!("delta-recurrent-prefill requires contiguous inputs"),
                };
                let query = query.cpu_storage().as_slice::<$ty>()?;
                let query = match query_layout.contiguous_offsets() {
                    Some((o1, o2)) => &query[o1..o2],
                    None => candle::bail!("delta-recurrent-prefill requires contiguous inputs"),
                };
                let key = key.cpu_storage().as_slice::<$ty>()?;
                let key = match key_layout.contiguous_offsets() {
                    Some((o1, o2)) => &key[o1..o2],
                    None => candle::bail!("delta-recurrent-prefill requires contiguous inputs"),
                };
                let value = value.cpu_storage().as_slice::<$ty>()?;
                let value = match value_layout.contiguous_offsets() {
                    Some((o1, o2)) => &value[o1..o2],
                    None => candle::bail!("delta-recurrent-prefill requires contiguous inputs"),
                };
                let beta = beta.cpu_storage().as_slice::<$ty>()?;
                let beta = match beta_layout.contiguous_offsets() {
                    Some((o1, o2)) => &beta[o1..o2],
                    None => candle::bail!("delta-recurrent-prefill requires contiguous inputs"),
                };
                let g = g.cpu_storage().as_slice::<$ty>()?;
                let g = match g_layout.contiguous_offsets() {
                    Some((o1, o2)) => &g[o1..o2],
                    None => candle::bail!("delta-recurrent-prefill requires contiguous inputs"),
                };
                let mut output = vec![$zero; elem_count];
                let status = unsafe {
                    candle::hip::ffi::qwen35_hip_delta_recurrent_prefill(
                        dtype_code,
                        batch_heads,
                        seq_len,
                        k_head_dim,
                        v_head_dim,
                        initial_state.as_ptr() as *const c_void,
                        query.as_ptr() as *const c_void,
                        key.as_ptr() as *const c_void,
                        value.as_ptr() as *const c_void,
                        beta.as_ptr() as *const c_void,
                        g.as_ptr() as *const c_void,
                        output.as_mut_ptr() as *mut c_void,
                    )
                };
                if status != 0 {
                    return Err(candle::hip::qwen35_error(self.name(), status));
                }
                let storage = <$ty as candle::WithDType>::to_cpu_storage_owned(output);
                Ok((
                    candle::HipStorage::wrap_cpu_storage(storage, device.clone()),
                    out_shape.clone(),
                ))
            }};
        }

        match storage_dtype {
            DType::F16 => launch!(half::f16, half::f16::from_bits(0)),
            DType::F32 => launch!(f32, 0.0f32),
            DType::BF16 => launch!(half::bf16, half::bf16::from_bits(0)),
            other => candle::bail!("delta-recurrent-prefill unsupported dtype {other:?}"),
        }
    }
}

fn delta_recurrent_prefill(
    initial_state: &Tensor,
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    beta: &Tensor,
    g: &Tensor,
) -> Result<Tensor> {
    initial_state.apply_op6_no_bwd(query, key, value, beta, g, &DeltaRecurrentPrefill)
}

#[derive(Debug, Clone, Copy)]
struct DeltaChunkStepRaw;

impl candle::CustomOp6 for DeltaChunkStepRaw {
    fn name(&self) -> &'static str {
        "delta-chunk-step-raw"
    }

    fn cpu_fwd(
        &self,
        _s1: &candle::CpuStorage,
        _l1: &candle::Layout,
        _s2: &candle::CpuStorage,
        _l2: &candle::Layout,
        _s3: &candle::CpuStorage,
        _l3: &candle::Layout,
        _s4: &candle::CpuStorage,
        _l4: &candle::Layout,
        _s5: &candle::CpuStorage,
        _l5: &candle::Layout,
        _s6: &candle::CpuStorage,
        _l6: &candle::Layout,
    ) -> Result<(candle::CpuStorage, candle::Shape)> {
        candle::bail!("delta-chunk-step-raw has no cpu implementation")
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        prev_state: &candle::CudaStorage,
        prev_layout: &candle::Layout,
        query: &candle::CudaStorage,
        query_layout: &candle::Layout,
        key: &candle::CudaStorage,
        key_layout: &candle::Layout,
        value: &candle::CudaStorage,
        value_layout: &candle::Layout,
        beta: &candle::CudaStorage,
        beta_layout: &candle::Layout,
        g: &candle::CudaStorage,
        g_layout: &candle::Layout,
    ) -> Result<(candle::CudaStorage, candle::Shape)> {
        use candle::backend::BackendStorage;
        use candle::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
        use candle::cuda_backend::WrapErr;

        if !(prev_layout.is_contiguous()
            && query_layout.is_contiguous()
            && key_layout.is_contiguous()
            && value_layout.is_contiguous()
            && beta_layout.is_contiguous()
            && g_layout.is_contiguous())
        {
            candle::bail!("delta-chunk-step-raw requires contiguous inputs")
        }

        let (batch_heads, k_head_dim, v_head_dim) = prev_layout.shape().dims3()?;
        let (query_bh, chunk_size, query_k) = query_layout.shape().dims3()?;
        let (key_bh, key_chunk, key_k) = key_layout.shape().dims3()?;
        let (value_bh, value_chunk, value_v) = value_layout.shape().dims3()?;
        let (beta_bh, beta_chunk) = beta_layout.shape().dims2()?;
        let (g_bh, g_chunk) = g_layout.shape().dims2()?;
        if query_bh != batch_heads
            || key_bh != batch_heads
            || value_bh != batch_heads
            || beta_bh != batch_heads
            || g_bh != batch_heads
            || key_chunk != chunk_size
            || value_chunk != chunk_size
            || beta_chunk != chunk_size
            || g_chunk != chunk_size
            || query_k != k_head_dim
            || key_k != k_head_dim
            || value_v != v_head_dim
        {
            candle::bail!(
                "delta-chunk-step-raw shape mismatch: prev={:?} query={:?} key={:?} value={:?} beta={:?} g={:?}",
                prev_layout.shape().dims(),
                query_layout.shape().dims(),
                key_layout.shape().dims(),
                value_layout.shape().dims(),
                beta_layout.shape().dims(),
                g_layout.shape().dims()
            )
        }

        let device = prev_state.device().clone();
        let out_shape =
            candle::Shape::from_dims(&[batch_heads, chunk_size + k_head_dim, v_head_dim]);
        let elem_count = out_shape.elem_count();
        let total_threads = batch_heads * v_head_dim;
        let cfg = LaunchConfig::for_num_elems(total_threads as u32);

        macro_rules! launch {
            ($ty:ty, $kernel:expr) => {{
                let prev_state = prev_state.as_cuda_slice::<$ty>()?;
                let prev_state = match prev_layout.contiguous_offsets() {
                    Some((o1, o2)) => prev_state.slice(o1..o2),
                    None => candle::bail!("delta-chunk-step-raw requires contiguous inputs"),
                };
                let query = query.as_cuda_slice::<$ty>()?;
                let query = match query_layout.contiguous_offsets() {
                    Some((o1, o2)) => query.slice(o1..o2),
                    None => candle::bail!("delta-chunk-step-raw requires contiguous inputs"),
                };
                let key = key.as_cuda_slice::<$ty>()?;
                let key = match key_layout.contiguous_offsets() {
                    Some((o1, o2)) => key.slice(o1..o2),
                    None => candle::bail!("delta-chunk-step-raw requires contiguous inputs"),
                };
                let value = value.as_cuda_slice::<$ty>()?;
                let value = match value_layout.contiguous_offsets() {
                    Some((o1, o2)) => value.slice(o1..o2),
                    None => candle::bail!("delta-chunk-step-raw requires contiguous inputs"),
                };
                let beta = beta.as_cuda_slice::<$ty>()?;
                let beta = match beta_layout.contiguous_offsets() {
                    Some((o1, o2)) => beta.slice(o1..o2),
                    None => candle::bail!("delta-chunk-step-raw requires contiguous inputs"),
                };
                let g = g.as_cuda_slice::<$ty>()?;
                let g = match g_layout.contiguous_offsets() {
                    Some((o1, o2)) => g.slice(o1..o2),
                    None => candle::bail!("delta-chunk-step-raw requires contiguous inputs"),
                };
                let output = unsafe { device.alloc::<$ty>(elem_count) }?;
                let func = device
                    .get_or_load_func($kernel, &candle::cuda_backend::kernels::QWEN35_DELTA)?;
                let mut builder = func.builder();
                candle::builder_arg!(builder, batch_heads, chunk_size, k_head_dim, v_head_dim);
                builder.arg(&prev_state);
                builder.arg(&query);
                builder.arg(&key);
                builder.arg(&value);
                builder.arg(&beta);
                builder.arg(&g);
                builder.arg(&output);
                unsafe { builder.launch(cfg) }.w()?;
                let storage = candle::CudaStorage::wrap_cuda_slice(output, device.clone());
                Ok((storage, out_shape.clone()))
            }};
        }

        match prev_state.dtype() {
            DType::F16 => launch!(half::f16, "delta_chunk_step_f16"),
            DType::F32 => launch!(f32, "delta_chunk_step_f32"),
            DType::BF16 => launch!(half::bf16, "delta_chunk_step_bf16"),
            other => candle::bail!("delta-chunk-step-raw unsupported dtype {other:?}"),
        }
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        prev_state: &candle::MetalStorage,
        prev_layout: &candle::Layout,
        query: &candle::MetalStorage,
        query_layout: &candle::Layout,
        key: &candle::MetalStorage,
        key_layout: &candle::Layout,
        value: &candle::MetalStorage,
        value_layout: &candle::Layout,
        beta: &candle::MetalStorage,
        beta_layout: &candle::Layout,
        g: &candle::MetalStorage,
        g_layout: &candle::Layout,
    ) -> Result<(candle::MetalStorage, candle::Shape)> {
        use candle::backend::BackendStorage;
        use candle::MetalError;

        if !(prev_layout.is_contiguous()
            && query_layout.is_contiguous()
            && key_layout.is_contiguous()
            && value_layout.is_contiguous()
            && beta_layout.is_contiguous()
            && g_layout.is_contiguous())
        {
            candle::bail!("delta-chunk-step-raw requires contiguous inputs")
        }

        let (batch_heads, k_head_dim, v_head_dim) = prev_layout.shape().dims3()?;
        let (query_bh, chunk_size, query_k) = query_layout.shape().dims3()?;
        let (key_bh, key_chunk, key_k) = key_layout.shape().dims3()?;
        let (value_bh, value_chunk, value_v) = value_layout.shape().dims3()?;
        let (beta_bh, beta_chunk) = beta_layout.shape().dims2()?;
        let (g_bh, g_chunk) = g_layout.shape().dims2()?;
        if query_bh != batch_heads
            || key_bh != batch_heads
            || value_bh != batch_heads
            || beta_bh != batch_heads
            || g_bh != batch_heads
            || key_chunk != chunk_size
            || value_chunk != chunk_size
            || beta_chunk != chunk_size
            || g_chunk != chunk_size
            || query_k != k_head_dim
            || key_k != k_head_dim
            || value_v != v_head_dim
        {
            candle::bail!(
                "delta-chunk-step-raw shape mismatch: prev={:?} query={:?} key={:?} value={:?} beta={:?} g={:?}",
                prev_layout.shape().dims(),
                query_layout.shape().dims(),
                key_layout.shape().dims(),
                value_layout.shape().dims(),
                beta_layout.shape().dims(),
                g_layout.shape().dims()
            )
        }

        let device = prev_state.device();
        let dtype = match prev_state.dtype() {
            DType::F16 => candle_metal_kernels::DType::F16,
            DType::F32 => candle_metal_kernels::DType::F32,
            DType::BF16 => candle_metal_kernels::DType::BF16,
            other => candle::bail!("delta-chunk-step-raw unsupported dtype {other:?}"),
        };
        let out_shape =
            candle::Shape::from_dims(&[batch_heads, chunk_size + k_head_dim, v_head_dim]);
        let elem_count = out_shape.elem_count();
        let output = device.new_buffer(elem_count, prev_state.dtype(), "delta-chunk-step-raw")?;
        let encoder = device.command_encoder()?;
        encoder.set_label("delta-chunk-step-raw");
        let prev_offset = candle_metal_kernels::BufferOffset {
            buffer: prev_state.buffer(),
            offset_in_bytes: prev_layout.start_offset() * prev_state.dtype().size_in_bytes(),
        };
        let query_offset = candle_metal_kernels::BufferOffset {
            buffer: query.buffer(),
            offset_in_bytes: query_layout.start_offset() * query.dtype().size_in_bytes(),
        };
        let key_offset = candle_metal_kernels::BufferOffset {
            buffer: key.buffer(),
            offset_in_bytes: key_layout.start_offset() * key.dtype().size_in_bytes(),
        };
        let value_offset = candle_metal_kernels::BufferOffset {
            buffer: value.buffer(),
            offset_in_bytes: value_layout.start_offset() * value.dtype().size_in_bytes(),
        };
        let beta_offset = candle_metal_kernels::BufferOffset {
            buffer: beta.buffer(),
            offset_in_bytes: beta_layout.start_offset() * beta.dtype().size_in_bytes(),
        };
        let g_offset = candle_metal_kernels::BufferOffset {
            buffer: g.buffer(),
            offset_in_bytes: g_layout.start_offset() * g.dtype().size_in_bytes(),
        };
        let use_split_kernel = matches!(
            std::env::var("CANDLE_QWEN35_DELTA_CHUNK_SPLIT_KERNEL").as_deref(),
            Ok("1" | "true" | "TRUE" | "yes" | "YES")
        );
        if use_split_kernel {
            let v_new_shape = candle::Shape::from_dims(&[batch_heads, chunk_size, v_head_dim]);
            let v_new_elem_count = v_new_shape.elem_count();
            let v_new_output =
                device.new_buffer(v_new_elem_count, prev_state.dtype(), "delta-chunk-v-new")?;
            candle_metal_kernels::call_delta_chunk_readout_split(
                device.metal_device(),
                &encoder,
                device.kernels(),
                dtype,
                batch_heads,
                chunk_size,
                k_head_dim,
                v_head_dim,
                prev_offset,
                query_offset,
                key_offset,
                value_offset,
                beta_offset,
                g_offset,
                &output,
                &v_new_output,
            )
            .map_err(MetalError::from)?;
            candle_metal_kernels::call_delta_chunk_state_update_raw(
                device.metal_device(),
                &encoder,
                device.kernels(),
                dtype,
                batch_heads,
                chunk_size,
                k_head_dim,
                v_head_dim,
                candle_metal_kernels::BufferOffset {
                    buffer: prev_state.buffer(),
                    offset_in_bytes: prev_layout.start_offset()
                        * prev_state.dtype().size_in_bytes(),
                },
                candle_metal_kernels::BufferOffset {
                    buffer: key.buffer(),
                    offset_in_bytes: key_layout.start_offset() * key.dtype().size_in_bytes(),
                },
                candle_metal_kernels::BufferOffset {
                    buffer: &v_new_output,
                    offset_in_bytes: 0,
                },
                candle_metal_kernels::BufferOffset {
                    buffer: g.buffer(),
                    offset_in_bytes: g_layout.start_offset() * g.dtype().size_in_bytes(),
                },
                chunk_size,
                &output,
            )
            .map_err(MetalError::from)?;
            let storage =
                candle::MetalStorage::new(output, device.clone(), elem_count, prev_state.dtype());
            return Ok((storage, out_shape));
        }
        let use_2d_kernel = delta_chunk_step_2d_enabled() && chunk_size <= 16;
        if use_2d_kernel {
            candle_metal_kernels::call_delta_chunk_step_2d(
                device.metal_device(),
                &encoder,
                device.kernels(),
                dtype,
                batch_heads,
                chunk_size,
                k_head_dim,
                v_head_dim,
                prev_offset,
                query_offset,
                key_offset,
                value_offset,
                beta_offset,
                g_offset,
                &output,
            )
            .map_err(MetalError::from)?;
        } else {
            candle_metal_kernels::call_delta_chunk_step(
                device.metal_device(),
                &encoder,
                device.kernels(),
                dtype,
                batch_heads,
                chunk_size,
                k_head_dim,
                v_head_dim,
                prev_offset,
                query_offset,
                key_offset,
                value_offset,
                beta_offset,
                g_offset,
                &output,
            )
            .map_err(MetalError::from)?;
        }
        let storage =
            candle::MetalStorage::new(output, device.clone(), elem_count, prev_state.dtype());
        Ok((storage, out_shape))
    }

    #[cfg(feature = "hip")]
    fn hip_fwd(
        &self,
        prev_state: &candle::HipStorage,
        prev_layout: &candle::Layout,
        query: &candle::HipStorage,
        query_layout: &candle::Layout,
        key: &candle::HipStorage,
        key_layout: &candle::Layout,
        value: &candle::HipStorage,
        value_layout: &candle::Layout,
        beta: &candle::HipStorage,
        beta_layout: &candle::Layout,
        g: &candle::HipStorage,
        g_layout: &candle::Layout,
    ) -> Result<(candle::HipStorage, candle::Shape)> {
        use candle::backend::BackendStorage;
        use std::ffi::c_void;

        if !(prev_layout.is_contiguous()
            && query_layout.is_contiguous()
            && key_layout.is_contiguous()
            && value_layout.is_contiguous()
            && beta_layout.is_contiguous()
            && g_layout.is_contiguous())
        {
            candle::bail!("delta-chunk-step-raw requires contiguous inputs")
        }

        let (batch_heads, k_head_dim, v_head_dim) = prev_layout.shape().dims3()?;
        let (query_bh, chunk_size, query_k) = query_layout.shape().dims3()?;
        let (key_bh, key_chunk, key_k) = key_layout.shape().dims3()?;
        let (value_bh, value_chunk, value_v) = value_layout.shape().dims3()?;
        let (beta_bh, beta_chunk) = beta_layout.shape().dims2()?;
        let (g_bh, g_chunk) = g_layout.shape().dims2()?;
        if query_bh != batch_heads
            || key_bh != batch_heads
            || value_bh != batch_heads
            || beta_bh != batch_heads
            || g_bh != batch_heads
            || key_chunk != chunk_size
            || value_chunk != chunk_size
            || beta_chunk != chunk_size
            || g_chunk != chunk_size
            || query_k != k_head_dim
            || key_k != k_head_dim
            || value_v != v_head_dim
        {
            candle::bail!(
                "delta-chunk-step-raw shape mismatch: prev={:?} query={:?} key={:?} value={:?} beta={:?} g={:?}",
                prev_layout.shape().dims(),
                query_layout.shape().dims(),
                key_layout.shape().dims(),
                value_layout.shape().dims(),
                beta_layout.shape().dims(),
                g_layout.shape().dims()
            )
        }

        let device = prev_state.device().clone();
        let storage_dtype = prev_state.dtype();
        let dtype_code = candle::hip::qwen35_dtype_code(storage_dtype)?;
        let out_shape =
            candle::Shape::from_dims(&[batch_heads, chunk_size + k_head_dim, v_head_dim]);
        let elem_count = out_shape.elem_count();

        macro_rules! launch {
            ($ty:ty, $zero:expr) => {{
                let prev_state = prev_state.cpu_storage().as_slice::<$ty>()?;
                let prev_state = match prev_layout.contiguous_offsets() {
                    Some((o1, o2)) => &prev_state[o1..o2],
                    None => candle::bail!("delta-chunk-step-raw requires contiguous inputs"),
                };
                let query = query.cpu_storage().as_slice::<$ty>()?;
                let query = match query_layout.contiguous_offsets() {
                    Some((o1, o2)) => &query[o1..o2],
                    None => candle::bail!("delta-chunk-step-raw requires contiguous inputs"),
                };
                let key = key.cpu_storage().as_slice::<$ty>()?;
                let key = match key_layout.contiguous_offsets() {
                    Some((o1, o2)) => &key[o1..o2],
                    None => candle::bail!("delta-chunk-step-raw requires contiguous inputs"),
                };
                let value = value.cpu_storage().as_slice::<$ty>()?;
                let value = match value_layout.contiguous_offsets() {
                    Some((o1, o2)) => &value[o1..o2],
                    None => candle::bail!("delta-chunk-step-raw requires contiguous inputs"),
                };
                let beta = beta.cpu_storage().as_slice::<$ty>()?;
                let beta = match beta_layout.contiguous_offsets() {
                    Some((o1, o2)) => &beta[o1..o2],
                    None => candle::bail!("delta-chunk-step-raw requires contiguous inputs"),
                };
                let g = g.cpu_storage().as_slice::<$ty>()?;
                let g = match g_layout.contiguous_offsets() {
                    Some((o1, o2)) => &g[o1..o2],
                    None => candle::bail!("delta-chunk-step-raw requires contiguous inputs"),
                };
                let mut output = vec![$zero; elem_count];
                let status = unsafe {
                    candle::hip::ffi::qwen35_hip_delta_chunk_step(
                        dtype_code,
                        batch_heads,
                        chunk_size,
                        k_head_dim,
                        v_head_dim,
                        prev_state.as_ptr() as *const c_void,
                        query.as_ptr() as *const c_void,
                        key.as_ptr() as *const c_void,
                        value.as_ptr() as *const c_void,
                        beta.as_ptr() as *const c_void,
                        g.as_ptr() as *const c_void,
                        output.as_mut_ptr() as *mut c_void,
                    )
                };
                if status != 0 {
                    return Err(candle::hip::qwen35_error(self.name(), status));
                }
                let storage = <$ty as candle::WithDType>::to_cpu_storage_owned(output);
                Ok((
                    candle::HipStorage::wrap_cpu_storage(storage, device.clone()),
                    out_shape.clone(),
                ))
            }};
        }

        match storage_dtype {
            DType::F16 => launch!(half::f16, half::f16::from_bits(0)),
            DType::F32 => launch!(f32, 0.0f32),
            DType::BF16 => launch!(half::bf16, half::bf16::from_bits(0)),
            other => candle::bail!("delta-chunk-step-raw unsupported dtype {other:?}"),
        }
    }
}

fn delta_chunk_step_raw(
    prev_state: &Tensor,
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    beta: &Tensor,
    g: &Tensor,
) -> Result<Tensor> {
    prev_state.apply_op6_no_bwd(query, key, value, beta, g, &DeltaChunkStepRaw)
}

#[derive(Debug, Clone, Copy)]
struct DeltaChunkStepWindowedRaw;

impl candle::CustomOp6 for DeltaChunkStepWindowedRaw {
    fn name(&self) -> &'static str {
        "delta-chunk-step-windowed-raw"
    }

    fn cpu_fwd(
        &self,
        _s1: &candle::CpuStorage,
        _l1: &candle::Layout,
        _s2: &candle::CpuStorage,
        _l2: &candle::Layout,
        _s3: &candle::CpuStorage,
        _l3: &candle::Layout,
        _s4: &candle::CpuStorage,
        _l4: &candle::Layout,
        _s5: &candle::CpuStorage,
        _l5: &candle::Layout,
        _s6: &candle::CpuStorage,
        _l6: &candle::Layout,
    ) -> Result<(candle::CpuStorage, candle::Shape)> {
        candle::bail!("delta-chunk-step-windowed-raw has no cpu implementation")
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        prev_state: &candle::CudaStorage,
        prev_layout: &candle::Layout,
        query: &candle::CudaStorage,
        query_layout: &candle::Layout,
        key: &candle::CudaStorage,
        key_layout: &candle::Layout,
        value: &candle::CudaStorage,
        value_layout: &candle::Layout,
        beta: &candle::CudaStorage,
        beta_layout: &candle::Layout,
        g: &candle::CudaStorage,
        g_layout: &candle::Layout,
    ) -> Result<(candle::CudaStorage, candle::Shape)> {
        use candle::backend::BackendStorage;
        use candle::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
        use candle::cuda_backend::WrapErr;

        if !(prev_layout.is_contiguous()
            && query_layout.is_contiguous()
            && key_layout.is_contiguous()
            && value_layout.is_contiguous()
            && beta_layout.is_contiguous()
            && g_layout.is_contiguous())
        {
            candle::bail!("delta-chunk-step-windowed-raw requires contiguous inputs")
        }

        let (batch_heads, k_head_dim, v_head_dim) = prev_layout.shape().dims3()?;
        let (query_bh, num_chunks, chunk_size, query_k) = query_layout.shape().dims4()?;
        let (key_bh, key_num_chunks, key_chunk, key_k) = key_layout.shape().dims4()?;
        let (value_bh, value_num_chunks, value_chunk, value_v) = value_layout.shape().dims4()?;
        let (beta_bh, beta_num_chunks, beta_chunk) = beta_layout.shape().dims3()?;
        let (g_bh, g_num_chunks, g_chunk) = g_layout.shape().dims3()?;
        if query_bh != batch_heads
            || key_bh != batch_heads
            || value_bh != batch_heads
            || beta_bh != batch_heads
            || g_bh != batch_heads
            || key_num_chunks != num_chunks
            || value_num_chunks != num_chunks
            || beta_num_chunks != num_chunks
            || g_num_chunks != num_chunks
            || key_chunk != chunk_size
            || value_chunk != chunk_size
            || beta_chunk != chunk_size
            || g_chunk != chunk_size
            || query_k != k_head_dim
            || key_k != k_head_dim
            || value_v != v_head_dim
        {
            candle::bail!(
                "delta-chunk-step-windowed-raw shape mismatch: prev={:?} query={:?} key={:?} value={:?} beta={:?} g={:?}",
                prev_layout.shape().dims(),
                query_layout.shape().dims(),
                key_layout.shape().dims(),
                value_layout.shape().dims(),
                beta_layout.shape().dims(),
                g_layout.shape().dims()
            )
        }

        let device = prev_state.device().clone();
        let total_tokens = num_chunks * chunk_size;
        let total_rows = total_tokens + k_head_dim;
        let out_shape = candle::Shape::from_dims(&[batch_heads, total_rows, v_head_dim]);
        let elem_count = out_shape.elem_count();
        let total_threads = batch_heads * v_head_dim;
        let cfg = LaunchConfig::for_num_elems(total_threads as u32);

        macro_rules! launch {
            ($ty:ty, $kernel:expr) => {{
                let prev_state = prev_state.as_cuda_slice::<$ty>()?;
                let prev_state = match prev_layout.contiguous_offsets() {
                    Some((o1, o2)) => prev_state.slice(o1..o2),
                    None => {
                        candle::bail!("delta-chunk-step-windowed-raw requires contiguous inputs")
                    }
                };
                let query = query.as_cuda_slice::<$ty>()?;
                let query = match query_layout.contiguous_offsets() {
                    Some((o1, o2)) => query.slice(o1..o2),
                    None => {
                        candle::bail!("delta-chunk-step-windowed-raw requires contiguous inputs")
                    }
                };
                let key = key.as_cuda_slice::<$ty>()?;
                let key = match key_layout.contiguous_offsets() {
                    Some((o1, o2)) => key.slice(o1..o2),
                    None => {
                        candle::bail!("delta-chunk-step-windowed-raw requires contiguous inputs")
                    }
                };
                let value = value.as_cuda_slice::<$ty>()?;
                let value = match value_layout.contiguous_offsets() {
                    Some((o1, o2)) => value.slice(o1..o2),
                    None => {
                        candle::bail!("delta-chunk-step-windowed-raw requires contiguous inputs")
                    }
                };
                let beta = beta.as_cuda_slice::<$ty>()?;
                let beta = match beta_layout.contiguous_offsets() {
                    Some((o1, o2)) => beta.slice(o1..o2),
                    None => {
                        candle::bail!("delta-chunk-step-windowed-raw requires contiguous inputs")
                    }
                };
                let g = g.as_cuda_slice::<$ty>()?;
                let g = match g_layout.contiguous_offsets() {
                    Some((o1, o2)) => g.slice(o1..o2),
                    None => {
                        candle::bail!("delta-chunk-step-windowed-raw requires contiguous inputs")
                    }
                };
                let output = unsafe { device.alloc::<$ty>(elem_count) }?;
                let func = device
                    .get_or_load_func($kernel, &candle::cuda_backend::kernels::QWEN35_DELTA)?;
                let mut builder = func.builder();
                candle::builder_arg!(
                    builder,
                    batch_heads,
                    num_chunks,
                    chunk_size,
                    k_head_dim,
                    v_head_dim
                );
                builder.arg(&prev_state);
                builder.arg(&query);
                builder.arg(&key);
                builder.arg(&value);
                builder.arg(&beta);
                builder.arg(&g);
                builder.arg(&output);
                unsafe { builder.launch(cfg) }.w()?;
                let storage = candle::CudaStorage::wrap_cuda_slice(output, device.clone());
                Ok((storage, out_shape.clone()))
            }};
        }

        match prev_state.dtype() {
            DType::F16 => launch!(half::f16, "delta_chunk_step_windowed_f16"),
            DType::F32 => launch!(f32, "delta_chunk_step_windowed_f32"),
            DType::BF16 => launch!(half::bf16, "delta_chunk_step_windowed_bf16"),
            other => candle::bail!("delta-chunk-step-windowed-raw unsupported dtype {other:?}"),
        }
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        prev_state: &candle::MetalStorage,
        prev_layout: &candle::Layout,
        query: &candle::MetalStorage,
        query_layout: &candle::Layout,
        key: &candle::MetalStorage,
        key_layout: &candle::Layout,
        value: &candle::MetalStorage,
        value_layout: &candle::Layout,
        beta: &candle::MetalStorage,
        beta_layout: &candle::Layout,
        g: &candle::MetalStorage,
        g_layout: &candle::Layout,
    ) -> Result<(candle::MetalStorage, candle::Shape)> {
        use candle::backend::BackendStorage;
        use candle::MetalError;

        if !(prev_layout.is_contiguous()
            && query_layout.is_contiguous()
            && key_layout.is_contiguous()
            && value_layout.is_contiguous()
            && beta_layout.is_contiguous()
            && g_layout.is_contiguous())
        {
            candle::bail!("delta-chunk-step-windowed-raw requires contiguous inputs")
        }

        let (batch_heads, k_head_dim, v_head_dim) = prev_layout.shape().dims3()?;
        let (query_bh, num_chunks, chunk_size, query_k) = query_layout.shape().dims4()?;
        let (key_bh, key_num_chunks, key_chunk, key_k) = key_layout.shape().dims4()?;
        let (value_bh, value_num_chunks, value_chunk, value_v) = value_layout.shape().dims4()?;
        let (beta_bh, beta_num_chunks, beta_chunk) = beta_layout.shape().dims3()?;
        let (g_bh, g_num_chunks, g_chunk) = g_layout.shape().dims3()?;
        if query_bh != batch_heads
            || key_bh != batch_heads
            || value_bh != batch_heads
            || beta_bh != batch_heads
            || g_bh != batch_heads
            || key_num_chunks != num_chunks
            || value_num_chunks != num_chunks
            || beta_num_chunks != num_chunks
            || g_num_chunks != num_chunks
            || key_chunk != chunk_size
            || value_chunk != chunk_size
            || beta_chunk != chunk_size
            || g_chunk != chunk_size
            || query_k != k_head_dim
            || key_k != k_head_dim
            || value_v != v_head_dim
        {
            candle::bail!(
                "delta-chunk-step-windowed-raw shape mismatch: prev={:?} query={:?} key={:?} value={:?} beta={:?} g={:?}",
                prev_layout.shape().dims(),
                query_layout.shape().dims(),
                key_layout.shape().dims(),
                value_layout.shape().dims(),
                beta_layout.shape().dims(),
                g_layout.shape().dims()
            )
        }

        let device = prev_state.device();
        let dtype = match prev_state.dtype() {
            DType::F16 => candle_metal_kernels::DType::F16,
            DType::F32 => candle_metal_kernels::DType::F32,
            DType::BF16 => candle_metal_kernels::DType::BF16,
            other => candle::bail!("delta-chunk-step-windowed-raw unsupported dtype {other:?}"),
        };
        let total_tokens = num_chunks * chunk_size;
        let total_rows = total_tokens + k_head_dim;
        let out_shape = candle::Shape::from_dims(&[batch_heads, total_rows, v_head_dim]);
        let elem_count = out_shape.elem_count();
        let output = device.new_buffer(
            elem_count,
            prev_state.dtype(),
            "delta-chunk-step-windowed-raw",
        )?;
        let encoder = device.command_encoder()?;
        encoder.set_label("delta-chunk-step-windowed-raw");

        let elem_bytes = prev_state.dtype().size_in_bytes();
        let query_chunk_elems = chunk_size * k_head_dim;
        let value_chunk_elems = chunk_size * v_head_dim;
        let scalar_chunk_elems = chunk_size;
        let query_bh_stride = num_chunks * query_chunk_elems;
        let value_bh_stride = num_chunks * value_chunk_elems;
        let scalar_bh_stride = num_chunks * scalar_chunk_elems;
        let initial_prev_offset = prev_layout.start_offset() * elem_bytes;
        let output_state_offset = total_tokens * v_head_dim * elem_bytes;
        let output_state_bh_stride = total_rows * v_head_dim;

        for chunk_idx in 0..num_chunks {
            let query_offset =
                (query_layout.start_offset() + chunk_idx * query_chunk_elems) * elem_bytes;
            let key_offset =
                (key_layout.start_offset() + chunk_idx * query_chunk_elems) * elem_bytes;
            let value_offset =
                (value_layout.start_offset() + chunk_idx * value_chunk_elems) * elem_bytes;
            let scalar_offset =
                (beta_layout.start_offset() + chunk_idx * scalar_chunk_elems) * elem_bytes;
            let g_scalar_offset =
                (g_layout.start_offset() + chunk_idx * scalar_chunk_elems) * elem_bytes;
            let prev = if chunk_idx == 0 {
                candle_metal_kernels::BufferOffset {
                    buffer: prev_state.buffer(),
                    offset_in_bytes: initial_prev_offset,
                }
            } else {
                candle_metal_kernels::BufferOffset {
                    buffer: &output,
                    offset_in_bytes: output_state_offset,
                }
            };
            let prev_state_bh_stride = if chunk_idx == 0 {
                k_head_dim * v_head_dim
            } else {
                output_state_bh_stride
            };
            let query = candle_metal_kernels::BufferOffset {
                buffer: query.buffer(),
                offset_in_bytes: query_offset,
            };
            let key = candle_metal_kernels::BufferOffset {
                buffer: key.buffer(),
                offset_in_bytes: key_offset,
            };
            let value = candle_metal_kernels::BufferOffset {
                buffer: value.buffer(),
                offset_in_bytes: value_offset,
            };
            let beta = candle_metal_kernels::BufferOffset {
                buffer: beta.buffer(),
                offset_in_bytes: scalar_offset,
            };
            let g = candle_metal_kernels::BufferOffset {
                buffer: g.buffer(),
                offset_in_bytes: g_scalar_offset,
            };
            if delta_chunk_step_windowed_2d_enabled() {
                candle_metal_kernels::call_delta_chunk_step_windowed_2d(
                    device.metal_device(),
                    &encoder,
                    device.kernels(),
                    dtype,
                    batch_heads,
                    chunk_size,
                    k_head_dim,
                    v_head_dim,
                    prev_state_bh_stride,
                    query_bh_stride,
                    value_bh_stride,
                    scalar_bh_stride,
                    total_rows,
                    chunk_idx * chunk_size,
                    total_tokens,
                    prev,
                    query,
                    key,
                    value,
                    beta,
                    g,
                    &output,
                )
                .map_err(MetalError::from)?;
            } else {
                candle_metal_kernels::call_delta_chunk_step_windowed(
                    device.metal_device(),
                    &encoder,
                    device.kernels(),
                    dtype,
                    batch_heads,
                    chunk_size,
                    k_head_dim,
                    v_head_dim,
                    prev_state_bh_stride,
                    query_bh_stride,
                    value_bh_stride,
                    scalar_bh_stride,
                    total_rows,
                    chunk_idx * chunk_size,
                    total_tokens,
                    prev,
                    query,
                    key,
                    value,
                    beta,
                    g,
                    &output,
                )
                .map_err(MetalError::from)?;
            }
        }

        let storage =
            candle::MetalStorage::new(output, device.clone(), elem_count, prev_state.dtype());
        Ok((storage, out_shape))
    }

    #[cfg(feature = "hip")]
    fn hip_fwd(
        &self,
        prev_state: &candle::HipStorage,
        prev_layout: &candle::Layout,
        query: &candle::HipStorage,
        query_layout: &candle::Layout,
        key: &candle::HipStorage,
        key_layout: &candle::Layout,
        value: &candle::HipStorage,
        value_layout: &candle::Layout,
        beta: &candle::HipStorage,
        beta_layout: &candle::Layout,
        g: &candle::HipStorage,
        g_layout: &candle::Layout,
    ) -> Result<(candle::HipStorage, candle::Shape)> {
        use candle::backend::BackendStorage;
        use std::ffi::c_void;

        if !(prev_layout.is_contiguous()
            && query_layout.is_contiguous()
            && key_layout.is_contiguous()
            && value_layout.is_contiguous()
            && beta_layout.is_contiguous()
            && g_layout.is_contiguous())
        {
            candle::bail!("delta-chunk-step-windowed-raw requires contiguous inputs")
        }

        let (batch_heads, k_head_dim, v_head_dim) = prev_layout.shape().dims3()?;
        let (query_bh, num_chunks, chunk_size, query_k) = query_layout.shape().dims4()?;
        let (key_bh, key_num_chunks, key_chunk, key_k) = key_layout.shape().dims4()?;
        let (value_bh, value_num_chunks, value_chunk, value_v) = value_layout.shape().dims4()?;
        let (beta_bh, beta_num_chunks, beta_chunk) = beta_layout.shape().dims3()?;
        let (g_bh, g_num_chunks, g_chunk) = g_layout.shape().dims3()?;
        if query_bh != batch_heads
            || key_bh != batch_heads
            || value_bh != batch_heads
            || beta_bh != batch_heads
            || g_bh != batch_heads
            || key_num_chunks != num_chunks
            || value_num_chunks != num_chunks
            || beta_num_chunks != num_chunks
            || g_num_chunks != num_chunks
            || key_chunk != chunk_size
            || value_chunk != chunk_size
            || beta_chunk != chunk_size
            || g_chunk != chunk_size
            || query_k != k_head_dim
            || key_k != k_head_dim
            || value_v != v_head_dim
        {
            candle::bail!(
                "delta-chunk-step-windowed-raw shape mismatch: prev={:?} query={:?} key={:?} value={:?} beta={:?} g={:?}",
                prev_layout.shape().dims(),
                query_layout.shape().dims(),
                key_layout.shape().dims(),
                value_layout.shape().dims(),
                beta_layout.shape().dims(),
                g_layout.shape().dims()
            )
        }

        let device = prev_state.device().clone();
        let storage_dtype = prev_state.dtype();
        let dtype_code = candle::hip::qwen35_dtype_code(storage_dtype)?;
        let total_tokens = num_chunks * chunk_size;
        let total_rows = total_tokens + k_head_dim;
        let out_shape = candle::Shape::from_dims(&[batch_heads, total_rows, v_head_dim]);
        let elem_count = out_shape.elem_count();

        macro_rules! launch {
            ($ty:ty, $zero:expr) => {{
                let prev_state = prev_state.cpu_storage().as_slice::<$ty>()?;
                let prev_state = match prev_layout.contiguous_offsets() {
                    Some((o1, o2)) => &prev_state[o1..o2],
                    None => {
                        candle::bail!("delta-chunk-step-windowed-raw requires contiguous inputs")
                    }
                };
                let query = query.cpu_storage().as_slice::<$ty>()?;
                let query = match query_layout.contiguous_offsets() {
                    Some((o1, o2)) => &query[o1..o2],
                    None => {
                        candle::bail!("delta-chunk-step-windowed-raw requires contiguous inputs")
                    }
                };
                let key = key.cpu_storage().as_slice::<$ty>()?;
                let key = match key_layout.contiguous_offsets() {
                    Some((o1, o2)) => &key[o1..o2],
                    None => {
                        candle::bail!("delta-chunk-step-windowed-raw requires contiguous inputs")
                    }
                };
                let value = value.cpu_storage().as_slice::<$ty>()?;
                let value = match value_layout.contiguous_offsets() {
                    Some((o1, o2)) => &value[o1..o2],
                    None => {
                        candle::bail!("delta-chunk-step-windowed-raw requires contiguous inputs")
                    }
                };
                let beta = beta.cpu_storage().as_slice::<$ty>()?;
                let beta = match beta_layout.contiguous_offsets() {
                    Some((o1, o2)) => &beta[o1..o2],
                    None => {
                        candle::bail!("delta-chunk-step-windowed-raw requires contiguous inputs")
                    }
                };
                let g = g.cpu_storage().as_slice::<$ty>()?;
                let g = match g_layout.contiguous_offsets() {
                    Some((o1, o2)) => &g[o1..o2],
                    None => {
                        candle::bail!("delta-chunk-step-windowed-raw requires contiguous inputs")
                    }
                };
                let mut output = vec![$zero; elem_count];
                let status = unsafe {
                    candle::hip::ffi::qwen35_hip_delta_chunk_windowed(
                        dtype_code,
                        batch_heads,
                        num_chunks,
                        chunk_size,
                        k_head_dim,
                        v_head_dim,
                        prev_state.as_ptr() as *const c_void,
                        query.as_ptr() as *const c_void,
                        key.as_ptr() as *const c_void,
                        value.as_ptr() as *const c_void,
                        beta.as_ptr() as *const c_void,
                        g.as_ptr() as *const c_void,
                        output.as_mut_ptr() as *mut c_void,
                    )
                };
                if status != 0 {
                    return Err(candle::hip::qwen35_error(self.name(), status));
                }
                let storage = <$ty as candle::WithDType>::to_cpu_storage_owned(output);
                Ok((
                    candle::HipStorage::wrap_cpu_storage(storage, device.clone()),
                    out_shape.clone(),
                ))
            }};
        }

        match storage_dtype {
            DType::F16 => launch!(half::f16, half::f16::from_bits(0)),
            DType::F32 => launch!(f32, 0.0f32),
            DType::BF16 => launch!(half::bf16, half::bf16::from_bits(0)),
            other => candle::bail!("delta-chunk-step-windowed-raw unsupported dtype {other:?}"),
        }
    }
}

fn delta_chunk_step_windowed_raw(
    prev_state: &Tensor,
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    beta: &Tensor,
    g: &Tensor,
) -> Result<Tensor> {
    prev_state.apply_op6_no_bwd(query, key, value, beta, g, &DeltaChunkStepWindowedRaw)
}

#[allow(dead_code)]
#[derive(Debug, Clone, Copy)]
struct DeltaChunkReadoutRaw;

impl candle::CustomOp6 for DeltaChunkReadoutRaw {
    fn name(&self) -> &'static str {
        "delta-chunk-readout-raw"
    }

    fn cpu_fwd(
        &self,
        _s1: &candle::CpuStorage,
        _l1: &candle::Layout,
        _s2: &candle::CpuStorage,
        _l2: &candle::Layout,
        _s3: &candle::CpuStorage,
        _l3: &candle::Layout,
        _s4: &candle::CpuStorage,
        _l4: &candle::Layout,
        _s5: &candle::CpuStorage,
        _l5: &candle::Layout,
        _s6: &candle::CpuStorage,
        _l6: &candle::Layout,
    ) -> Result<(candle::CpuStorage, candle::Shape)> {
        candle::bail!("delta-chunk-readout-raw has no cpu implementation")
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        prev_state: &candle::MetalStorage,
        prev_layout: &candle::Layout,
        query: &candle::MetalStorage,
        query_layout: &candle::Layout,
        key: &candle::MetalStorage,
        key_layout: &candle::Layout,
        value: &candle::MetalStorage,
        value_layout: &candle::Layout,
        beta: &candle::MetalStorage,
        beta_layout: &candle::Layout,
        g: &candle::MetalStorage,
        g_layout: &candle::Layout,
    ) -> Result<(candle::MetalStorage, candle::Shape)> {
        use candle::backend::BackendStorage;
        use candle::MetalError;

        if !(prev_layout.is_contiguous()
            && query_layout.is_contiguous()
            && key_layout.is_contiguous()
            && value_layout.is_contiguous()
            && beta_layout.is_contiguous()
            && g_layout.is_contiguous())
        {
            candle::bail!("delta-chunk-readout-raw requires contiguous inputs")
        }

        let (batch_heads, k_head_dim, v_head_dim) = prev_layout.shape().dims3()?;
        let (query_bh, chunk_size, query_k) = query_layout.shape().dims3()?;
        let (key_bh, key_chunk, key_k) = key_layout.shape().dims3()?;
        let (value_bh, value_chunk, value_v) = value_layout.shape().dims3()?;
        let (beta_bh, beta_chunk) = beta_layout.shape().dims2()?;
        let (g_bh, g_chunk) = g_layout.shape().dims2()?;
        if query_bh != batch_heads
            || key_bh != batch_heads
            || value_bh != batch_heads
            || beta_bh != batch_heads
            || g_bh != batch_heads
            || key_chunk != chunk_size
            || value_chunk != chunk_size
            || beta_chunk != chunk_size
            || g_chunk != chunk_size
            || query_k != k_head_dim
            || key_k != k_head_dim
            || value_v != v_head_dim
        {
            candle::bail!(
                "delta-chunk-readout-raw shape mismatch: prev={:?} query={:?} key={:?} value={:?} beta={:?} g={:?}",
                prev_layout.shape().dims(),
                query_layout.shape().dims(),
                key_layout.shape().dims(),
                value_layout.shape().dims(),
                beta_layout.shape().dims(),
                g_layout.shape().dims()
            )
        }

        let device = prev_state.device();
        let dtype = match prev_state.dtype() {
            DType::F16 => candle_metal_kernels::DType::F16,
            DType::F32 => candle_metal_kernels::DType::F32,
            DType::BF16 => candle_metal_kernels::DType::BF16,
            other => candle::bail!("delta-chunk-readout-raw unsupported dtype {other:?}"),
        };
        let out_shape = candle::Shape::from_dims(&[batch_heads, 2 * chunk_size, v_head_dim]);
        let elem_count = out_shape.elem_count();
        let output =
            device.new_buffer(elem_count, prev_state.dtype(), "delta-chunk-readout-raw")?;
        let encoder = device.command_encoder()?;
        encoder.set_label("delta-chunk-readout-raw");
        let prev = candle_metal_kernels::BufferOffset {
            buffer: prev_state.buffer(),
            offset_in_bytes: prev_layout.start_offset() * prev_state.dtype().size_in_bytes(),
        };
        let query = candle_metal_kernels::BufferOffset {
            buffer: query.buffer(),
            offset_in_bytes: query_layout.start_offset() * query.dtype().size_in_bytes(),
        };
        let key = candle_metal_kernels::BufferOffset {
            buffer: key.buffer(),
            offset_in_bytes: key_layout.start_offset() * key.dtype().size_in_bytes(),
        };
        let value = candle_metal_kernels::BufferOffset {
            buffer: value.buffer(),
            offset_in_bytes: value_layout.start_offset() * value.dtype().size_in_bytes(),
        };
        let beta = candle_metal_kernels::BufferOffset {
            buffer: beta.buffer(),
            offset_in_bytes: beta_layout.start_offset() * beta.dtype().size_in_bytes(),
        };
        let g = candle_metal_kernels::BufferOffset {
            buffer: g.buffer(),
            offset_in_bytes: g_layout.start_offset() * g.dtype().size_in_bytes(),
        };
        candle_metal_kernels::call_delta_chunk_readout(
            device.metal_device(),
            &encoder,
            device.kernels(),
            dtype,
            batch_heads,
            chunk_size,
            k_head_dim,
            v_head_dim,
            prev,
            query,
            key,
            value,
            beta,
            g,
            &output,
        )
        .map_err(MetalError::from)?;
        let storage =
            candle::MetalStorage::new(output, device.clone(), elem_count, prev_state.dtype());
        Ok((storage, out_shape))
    }
}

#[allow(dead_code)]
fn delta_chunk_readout_raw(
    prev_state: &Tensor,
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    beta: &Tensor,
    g: &Tensor,
) -> Result<Tensor> {
    prev_state.apply_op6_no_bwd(query, key, value, beta, g, &DeltaChunkReadoutRaw)
}

#[allow(dead_code)]
#[derive(Debug, Clone, Copy)]
struct DeltaChunkStateUpdateRaw;

impl candle::CustomOp4 for DeltaChunkStateUpdateRaw {
    fn name(&self) -> &'static str {
        "delta-chunk-state-update-raw"
    }

    fn cpu_fwd(
        &self,
        _s1: &candle::CpuStorage,
        _l1: &candle::Layout,
        _s2: &candle::CpuStorage,
        _l2: &candle::Layout,
        _s3: &candle::CpuStorage,
        _l3: &candle::Layout,
        _s4: &candle::CpuStorage,
        _l4: &candle::Layout,
    ) -> Result<(candle::CpuStorage, candle::Shape)> {
        candle::bail!("delta-chunk-state-update-raw has no cpu implementation")
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        prev_state: &candle::MetalStorage,
        prev_layout: &candle::Layout,
        key: &candle::MetalStorage,
        key_layout: &candle::Layout,
        v_new: &candle::MetalStorage,
        v_new_layout: &candle::Layout,
        g: &candle::MetalStorage,
        g_layout: &candle::Layout,
    ) -> Result<(candle::MetalStorage, candle::Shape)> {
        use candle::backend::BackendStorage;
        use candle::MetalError;

        if !(prev_layout.is_contiguous()
            && key_layout.is_contiguous()
            && v_new_layout.is_contiguous()
            && g_layout.is_contiguous())
        {
            candle::bail!("delta-chunk-state-update-raw requires contiguous inputs")
        }

        let (batch_heads, k_head_dim, v_head_dim) = prev_layout.shape().dims3()?;
        let (key_bh, chunk_size, key_k) = key_layout.shape().dims3()?;
        let (v_new_bh, v_new_chunk, v_new_v) = v_new_layout.shape().dims3()?;
        let (g_bh, g_chunk) = g_layout.shape().dims2()?;
        if key_bh != batch_heads
            || v_new_bh != batch_heads
            || g_bh != batch_heads
            || v_new_chunk != chunk_size
            || g_chunk != chunk_size
            || key_k != k_head_dim
            || v_new_v != v_head_dim
        {
            candle::bail!(
                "delta-chunk-state-update-raw shape mismatch: prev={:?} key={:?} v_new={:?} g={:?}",
                prev_layout.shape().dims(),
                key_layout.shape().dims(),
                v_new_layout.shape().dims(),
                g_layout.shape().dims()
            )
        }

        let device = prev_state.device();
        let dtype = match prev_state.dtype() {
            DType::F16 => candle_metal_kernels::DType::F16,
            DType::F32 => candle_metal_kernels::DType::F32,
            DType::BF16 => candle_metal_kernels::DType::BF16,
            other => candle::bail!("delta-chunk-state-update-raw unsupported dtype {other:?}"),
        };
        let out_shape = candle::Shape::from_dims(&[batch_heads, k_head_dim, v_head_dim]);
        let elem_count = out_shape.elem_count();
        let output = device.new_buffer(
            elem_count,
            prev_state.dtype(),
            "delta-chunk-state-update-raw",
        )?;
        let encoder = device.command_encoder()?;
        encoder.set_label("delta-chunk-state-update-raw");
        let prev = candle_metal_kernels::BufferOffset {
            buffer: prev_state.buffer(),
            offset_in_bytes: prev_layout.start_offset() * prev_state.dtype().size_in_bytes(),
        };
        let key = candle_metal_kernels::BufferOffset {
            buffer: key.buffer(),
            offset_in_bytes: key_layout.start_offset() * key.dtype().size_in_bytes(),
        };
        let v_new = candle_metal_kernels::BufferOffset {
            buffer: v_new.buffer(),
            offset_in_bytes: v_new_layout.start_offset() * v_new.dtype().size_in_bytes(),
        };
        let g = candle_metal_kernels::BufferOffset {
            buffer: g.buffer(),
            offset_in_bytes: g_layout.start_offset() * g.dtype().size_in_bytes(),
        };
        candle_metal_kernels::call_delta_chunk_state_update_raw(
            device.metal_device(),
            &encoder,
            device.kernels(),
            dtype,
            batch_heads,
            chunk_size,
            k_head_dim,
            v_head_dim,
            prev,
            key,
            v_new,
            g,
            0,
            &output,
        )
        .map_err(MetalError::from)?;
        let storage =
            candle::MetalStorage::new(output, device.clone(), elem_count, prev_state.dtype());
        Ok((storage, out_shape))
    }
}

#[allow(dead_code)]
fn delta_chunk_state_update_raw(
    prev_state: &Tensor,
    key: &Tensor,
    v_new: &Tensor,
    g: &Tensor,
) -> Result<Tensor> {
    prev_state.apply_op4_no_bwd(key, v_new, g, &DeltaChunkStateUpdateRaw)
}

#[derive(Debug, Clone, Copy)]
struct DeltaChunkScanRaw;

impl candle::CustomOp6 for DeltaChunkScanRaw {
    fn name(&self) -> &'static str {
        "delta-chunk-scan-raw"
    }

    fn cpu_fwd(
        &self,
        _s1: &candle::CpuStorage,
        _l1: &candle::Layout,
        _s2: &candle::CpuStorage,
        _l2: &candle::Layout,
        _s3: &candle::CpuStorage,
        _l3: &candle::Layout,
        _s4: &candle::CpuStorage,
        _l4: &candle::Layout,
        _s5: &candle::CpuStorage,
        _l5: &candle::Layout,
        _s6: &candle::CpuStorage,
        _l6: &candle::Layout,
    ) -> Result<(candle::CpuStorage, candle::Shape)> {
        candle::bail!("delta-chunk-scan-raw has no cpu implementation")
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        initial_state: &candle::CudaStorage,
        initial_layout: &candle::Layout,
        query: &candle::CudaStorage,
        query_layout: &candle::Layout,
        key: &candle::CudaStorage,
        key_layout: &candle::Layout,
        value: &candle::CudaStorage,
        value_layout: &candle::Layout,
        beta: &candle::CudaStorage,
        beta_layout: &candle::Layout,
        g: &candle::CudaStorage,
        g_layout: &candle::Layout,
    ) -> Result<(candle::CudaStorage, candle::Shape)> {
        use candle::backend::BackendStorage;
        use candle::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
        use candle::cuda_backend::WrapErr;

        if !(initial_layout.is_contiguous()
            && query_layout.is_contiguous()
            && key_layout.is_contiguous()
            && value_layout.is_contiguous()
            && beta_layout.is_contiguous()
            && g_layout.is_contiguous())
        {
            candle::bail!("delta-chunk-scan-raw requires contiguous inputs")
        }

        let (batch_heads, k_head_dim, v_head_dim) = initial_layout.shape().dims3()?;
        let (query_bh, num_chunks, chunk_size, query_k) = query_layout.shape().dims4()?;
        let (key_bh, key_num_chunks, key_chunk, key_k) = key_layout.shape().dims4()?;
        let (value_bh, value_num_chunks, value_chunk, value_v) = value_layout.shape().dims4()?;
        let (beta_bh, beta_num_chunks, beta_chunk) = beta_layout.shape().dims3()?;
        let (g_bh, g_num_chunks, g_chunk) = g_layout.shape().dims3()?;
        if query_bh != batch_heads
            || key_bh != batch_heads
            || value_bh != batch_heads
            || beta_bh != batch_heads
            || g_bh != batch_heads
            || key_num_chunks != num_chunks
            || value_num_chunks != num_chunks
            || beta_num_chunks != num_chunks
            || g_num_chunks != num_chunks
            || key_chunk != chunk_size
            || value_chunk != chunk_size
            || beta_chunk != chunk_size
            || g_chunk != chunk_size
            || query_k != k_head_dim
            || key_k != k_head_dim
            || value_v != v_head_dim
        {
            candle::bail!(
                "delta-chunk-scan-raw shape mismatch: initial={:?} query={:?} key={:?} value={:?} beta={:?} g={:?}",
                initial_layout.shape().dims(),
                query_layout.shape().dims(),
                key_layout.shape().dims(),
                value_layout.shape().dims(),
                beta_layout.shape().dims(),
                g_layout.shape().dims()
            )
        }

        let device = initial_state.device().clone();
        let out_shape = candle::Shape::from_dims(&[
            batch_heads,
            num_chunks * chunk_size + k_head_dim,
            v_head_dim,
        ]);
        let elem_count = out_shape.elem_count();
        let total_threads = batch_heads * v_head_dim;
        let cfg = LaunchConfig::for_num_elems(total_threads as u32);

        macro_rules! launch {
            ($ty:ty, $kernel:expr) => {{
                let initial_state = initial_state.as_cuda_slice::<$ty>()?;
                let initial_state = match initial_layout.contiguous_offsets() {
                    Some((o1, o2)) => initial_state.slice(o1..o2),
                    None => candle::bail!("delta-chunk-scan-raw requires contiguous inputs"),
                };
                let query = query.as_cuda_slice::<$ty>()?;
                let query = match query_layout.contiguous_offsets() {
                    Some((o1, o2)) => query.slice(o1..o2),
                    None => candle::bail!("delta-chunk-scan-raw requires contiguous inputs"),
                };
                let key = key.as_cuda_slice::<$ty>()?;
                let key = match key_layout.contiguous_offsets() {
                    Some((o1, o2)) => key.slice(o1..o2),
                    None => candle::bail!("delta-chunk-scan-raw requires contiguous inputs"),
                };
                let value = value.as_cuda_slice::<$ty>()?;
                let value = match value_layout.contiguous_offsets() {
                    Some((o1, o2)) => value.slice(o1..o2),
                    None => candle::bail!("delta-chunk-scan-raw requires contiguous inputs"),
                };
                let beta = beta.as_cuda_slice::<$ty>()?;
                let beta = match beta_layout.contiguous_offsets() {
                    Some((o1, o2)) => beta.slice(o1..o2),
                    None => candle::bail!("delta-chunk-scan-raw requires contiguous inputs"),
                };
                let g = g.as_cuda_slice::<$ty>()?;
                let g = match g_layout.contiguous_offsets() {
                    Some((o1, o2)) => g.slice(o1..o2),
                    None => candle::bail!("delta-chunk-scan-raw requires contiguous inputs"),
                };
                let output = unsafe { device.alloc::<$ty>(elem_count) }?;
                let func = device
                    .get_or_load_func($kernel, &candle::cuda_backend::kernels::QWEN35_DELTA)?;
                let mut builder = func.builder();
                candle::builder_arg!(
                    builder,
                    batch_heads,
                    num_chunks,
                    chunk_size,
                    k_head_dim,
                    v_head_dim
                );
                builder.arg(&initial_state);
                builder.arg(&query);
                builder.arg(&key);
                builder.arg(&value);
                builder.arg(&beta);
                builder.arg(&g);
                builder.arg(&output);
                unsafe { builder.launch(cfg) }.w()?;
                let storage = candle::CudaStorage::wrap_cuda_slice(output, device.clone());
                Ok((storage, out_shape.clone()))
            }};
        }

        match initial_state.dtype() {
            DType::F16 => launch!(half::f16, "delta_chunk_scan_raw_f16"),
            DType::F32 => launch!(f32, "delta_chunk_scan_raw_f32"),
            DType::BF16 => launch!(half::bf16, "delta_chunk_scan_raw_bf16"),
            other => candle::bail!("delta-chunk-scan-raw unsupported dtype {other:?}"),
        }
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        initial_state: &candle::MetalStorage,
        initial_layout: &candle::Layout,
        query: &candle::MetalStorage,
        query_layout: &candle::Layout,
        key: &candle::MetalStorage,
        key_layout: &candle::Layout,
        value: &candle::MetalStorage,
        value_layout: &candle::Layout,
        beta: &candle::MetalStorage,
        beta_layout: &candle::Layout,
        g: &candle::MetalStorage,
        g_layout: &candle::Layout,
    ) -> Result<(candle::MetalStorage, candle::Shape)> {
        use candle::backend::BackendStorage;
        use candle::MetalError;

        if !(initial_layout.is_contiguous()
            && query_layout.is_contiguous()
            && key_layout.is_contiguous()
            && value_layout.is_contiguous()
            && beta_layout.is_contiguous()
            && g_layout.is_contiguous())
        {
            candle::bail!("delta-chunk-scan-raw requires contiguous inputs")
        }

        let (batch_heads, k_head_dim, v_head_dim) = initial_layout.shape().dims3()?;
        let (query_bh, num_chunks, chunk_size, query_k) = query_layout.shape().dims4()?;
        let (key_bh, key_num_chunks, key_chunk, key_k) = key_layout.shape().dims4()?;
        let (value_bh, value_num_chunks, value_chunk, value_v) = value_layout.shape().dims4()?;
        let (beta_bh, beta_num_chunks, beta_chunk) = beta_layout.shape().dims3()?;
        let (g_bh, g_num_chunks, g_chunk) = g_layout.shape().dims3()?;
        if query_bh != batch_heads
            || key_bh != batch_heads
            || value_bh != batch_heads
            || beta_bh != batch_heads
            || g_bh != batch_heads
            || key_num_chunks != num_chunks
            || value_num_chunks != num_chunks
            || beta_num_chunks != num_chunks
            || g_num_chunks != num_chunks
            || key_chunk != chunk_size
            || value_chunk != chunk_size
            || beta_chunk != chunk_size
            || g_chunk != chunk_size
            || query_k != k_head_dim
            || key_k != k_head_dim
            || value_v != v_head_dim
        {
            candle::bail!(
                "delta-chunk-scan-raw shape mismatch: initial={:?} query={:?} key={:?} value={:?} beta={:?} g={:?}",
                initial_layout.shape().dims(),
                query_layout.shape().dims(),
                key_layout.shape().dims(),
                value_layout.shape().dims(),
                beta_layout.shape().dims(),
                g_layout.shape().dims()
            )
        }

        let device = initial_state.device();
        let dtype = match initial_state.dtype() {
            DType::F16 => candle_metal_kernels::DType::F16,
            DType::F32 => candle_metal_kernels::DType::F32,
            DType::BF16 => candle_metal_kernels::DType::BF16,
            other => candle::bail!("delta-chunk-scan-raw unsupported dtype {other:?}"),
        };
        let out_shape = candle::Shape::from_dims(&[
            batch_heads,
            num_chunks * chunk_size + k_head_dim,
            v_head_dim,
        ]);
        let elem_count = out_shape.elem_count();
        let output =
            device.new_buffer(elem_count, initial_state.dtype(), "delta-chunk-scan-raw")?;
        let encoder = device.command_encoder()?;
        encoder.set_label("delta-chunk-scan-raw");
        let initial = candle_metal_kernels::BufferOffset {
            buffer: initial_state.buffer(),
            offset_in_bytes: initial_layout.start_offset() * initial_state.dtype().size_in_bytes(),
        };
        let query = candle_metal_kernels::BufferOffset {
            buffer: query.buffer(),
            offset_in_bytes: query_layout.start_offset() * query.dtype().size_in_bytes(),
        };
        let key = candle_metal_kernels::BufferOffset {
            buffer: key.buffer(),
            offset_in_bytes: key_layout.start_offset() * key.dtype().size_in_bytes(),
        };
        let value = candle_metal_kernels::BufferOffset {
            buffer: value.buffer(),
            offset_in_bytes: value_layout.start_offset() * value.dtype().size_in_bytes(),
        };
        let beta = candle_metal_kernels::BufferOffset {
            buffer: beta.buffer(),
            offset_in_bytes: beta_layout.start_offset() * beta.dtype().size_in_bytes(),
        };
        let g = candle_metal_kernels::BufferOffset {
            buffer: g.buffer(),
            offset_in_bytes: g_layout.start_offset() * g.dtype().size_in_bytes(),
        };
        candle_metal_kernels::call_delta_chunk_scan_raw(
            device.metal_device(),
            &encoder,
            device.kernels(),
            dtype,
            batch_heads,
            num_chunks,
            chunk_size,
            k_head_dim,
            v_head_dim,
            initial,
            query,
            key,
            value,
            beta,
            g,
            &output,
        )
        .map_err(MetalError::from)?;
        let storage =
            candle::MetalStorage::new(output, device.clone(), elem_count, initial_state.dtype());
        Ok((storage, out_shape))
    }

    #[cfg(feature = "hip")]
    fn hip_fwd(
        &self,
        initial_state: &candle::HipStorage,
        initial_layout: &candle::Layout,
        query: &candle::HipStorage,
        query_layout: &candle::Layout,
        key: &candle::HipStorage,
        key_layout: &candle::Layout,
        value: &candle::HipStorage,
        value_layout: &candle::Layout,
        beta: &candle::HipStorage,
        beta_layout: &candle::Layout,
        g: &candle::HipStorage,
        g_layout: &candle::Layout,
    ) -> Result<(candle::HipStorage, candle::Shape)> {
        use candle::backend::BackendStorage;
        use std::ffi::c_void;

        if !(initial_layout.is_contiguous()
            && query_layout.is_contiguous()
            && key_layout.is_contiguous()
            && value_layout.is_contiguous()
            && beta_layout.is_contiguous()
            && g_layout.is_contiguous())
        {
            candle::bail!("delta-chunk-scan-raw requires contiguous inputs")
        }

        let (batch_heads, k_head_dim, v_head_dim) = initial_layout.shape().dims3()?;
        let (query_bh, num_chunks, chunk_size, query_k) = query_layout.shape().dims4()?;
        let (key_bh, key_num_chunks, key_chunk, key_k) = key_layout.shape().dims4()?;
        let (value_bh, value_num_chunks, value_chunk, value_v) = value_layout.shape().dims4()?;
        let (beta_bh, beta_num_chunks, beta_chunk) = beta_layout.shape().dims3()?;
        let (g_bh, g_num_chunks, g_chunk) = g_layout.shape().dims3()?;
        if query_bh != batch_heads
            || key_bh != batch_heads
            || value_bh != batch_heads
            || beta_bh != batch_heads
            || g_bh != batch_heads
            || key_num_chunks != num_chunks
            || value_num_chunks != num_chunks
            || beta_num_chunks != num_chunks
            || g_num_chunks != num_chunks
            || key_chunk != chunk_size
            || value_chunk != chunk_size
            || beta_chunk != chunk_size
            || g_chunk != chunk_size
            || query_k != k_head_dim
            || key_k != k_head_dim
            || value_v != v_head_dim
        {
            candle::bail!(
                "delta-chunk-scan-raw shape mismatch: initial={:?} query={:?} key={:?} value={:?} beta={:?} g={:?}",
                initial_layout.shape().dims(),
                query_layout.shape().dims(),
                key_layout.shape().dims(),
                value_layout.shape().dims(),
                beta_layout.shape().dims(),
                g_layout.shape().dims()
            )
        }

        let device = initial_state.device().clone();
        let storage_dtype = initial_state.dtype();
        let dtype_code = candle::hip::qwen35_dtype_code(storage_dtype)?;
        let out_shape = candle::Shape::from_dims(&[
            batch_heads,
            num_chunks * chunk_size + k_head_dim,
            v_head_dim,
        ]);
        let elem_count = out_shape.elem_count();

        macro_rules! launch {
            ($ty:ty, $zero:expr) => {{
                let initial_state = initial_state.cpu_storage().as_slice::<$ty>()?;
                let initial_state = match initial_layout.contiguous_offsets() {
                    Some((o1, o2)) => &initial_state[o1..o2],
                    None => candle::bail!("delta-chunk-scan-raw requires contiguous inputs"),
                };
                let query = query.cpu_storage().as_slice::<$ty>()?;
                let query = match query_layout.contiguous_offsets() {
                    Some((o1, o2)) => &query[o1..o2],
                    None => candle::bail!("delta-chunk-scan-raw requires contiguous inputs"),
                };
                let key = key.cpu_storage().as_slice::<$ty>()?;
                let key = match key_layout.contiguous_offsets() {
                    Some((o1, o2)) => &key[o1..o2],
                    None => candle::bail!("delta-chunk-scan-raw requires contiguous inputs"),
                };
                let value = value.cpu_storage().as_slice::<$ty>()?;
                let value = match value_layout.contiguous_offsets() {
                    Some((o1, o2)) => &value[o1..o2],
                    None => candle::bail!("delta-chunk-scan-raw requires contiguous inputs"),
                };
                let beta = beta.cpu_storage().as_slice::<$ty>()?;
                let beta = match beta_layout.contiguous_offsets() {
                    Some((o1, o2)) => &beta[o1..o2],
                    None => candle::bail!("delta-chunk-scan-raw requires contiguous inputs"),
                };
                let g = g.cpu_storage().as_slice::<$ty>()?;
                let g = match g_layout.contiguous_offsets() {
                    Some((o1, o2)) => &g[o1..o2],
                    None => candle::bail!("delta-chunk-scan-raw requires contiguous inputs"),
                };
                let mut output = vec![$zero; elem_count];
                let status = unsafe {
                    candle::hip::ffi::qwen35_hip_delta_chunk_scan_raw(
                        dtype_code,
                        batch_heads,
                        num_chunks,
                        chunk_size,
                        k_head_dim,
                        v_head_dim,
                        initial_state.as_ptr() as *const c_void,
                        query.as_ptr() as *const c_void,
                        key.as_ptr() as *const c_void,
                        value.as_ptr() as *const c_void,
                        beta.as_ptr() as *const c_void,
                        g.as_ptr() as *const c_void,
                        output.as_mut_ptr() as *mut c_void,
                    )
                };
                if status != 0 {
                    return Err(candle::hip::qwen35_error(self.name(), status));
                }
                let storage = <$ty as candle::WithDType>::to_cpu_storage_owned(output);
                Ok((
                    candle::HipStorage::wrap_cpu_storage(storage, device.clone()),
                    out_shape.clone(),
                ))
            }};
        }

        match storage_dtype {
            DType::F16 => launch!(half::f16, half::f16::from_bits(0)),
            DType::F32 => launch!(f32, 0.0f32),
            DType::BF16 => launch!(half::bf16, half::bf16::from_bits(0)),
            other => candle::bail!("delta-chunk-scan-raw unsupported dtype {other:?}"),
        }
    }
}

fn delta_chunk_scan_raw(
    initial_state: &Tensor,
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    beta: &Tensor,
    g: &Tensor,
) -> Result<Tensor> {
    initial_state.apply_op6_no_bwd(query, key, value, beta, g, &DeltaChunkScanRaw)
}

#[derive(Debug, Clone, Copy)]
struct DeltaFullScan;

impl candle::CustomOp7 for DeltaFullScan {
    fn name(&self) -> &'static str {
        "delta-full-scan"
    }

    fn cpu_fwd(
        &self,
        _s1: &candle::CpuStorage,
        _l1: &candle::Layout,
        _s2: &candle::CpuStorage,
        _l2: &candle::Layout,
        _s3: &candle::CpuStorage,
        _l3: &candle::Layout,
        _s4: &candle::CpuStorage,
        _l4: &candle::Layout,
        _s5: &candle::CpuStorage,
        _l5: &candle::Layout,
        _s6: &candle::CpuStorage,
        _l6: &candle::Layout,
        _s7: &candle::CpuStorage,
        _l7: &candle::Layout,
    ) -> Result<(candle::CpuStorage, candle::Shape)> {
        candle::bail!("delta-full-scan has no cpu implementation")
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        initial_state: &candle::CudaStorage,
        initial_layout: &candle::Layout,
        weighted_key_scan: &candle::CudaStorage,
        weighted_key_layout: &candle::Layout,
        k_cumdecay_scan: &candle::CudaStorage,
        k_cumdecay_layout: &candle::Layout,
        q_state_scan: &candle::CudaStorage,
        q_state_layout: &candle::Layout,
        local_attn_scan: &candle::CudaStorage,
        local_attn_layout: &candle::Layout,
        state_decay_scan: &candle::CudaStorage,
        state_decay_layout: &candle::Layout,
        value: &candle::CudaStorage,
        value_layout: &candle::Layout,
    ) -> Result<(candle::CudaStorage, candle::Shape)> {
        use candle::backend::BackendStorage;
        use candle::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
        use candle::cuda_backend::WrapErr;

        if !(initial_layout.is_contiguous()
            && weighted_key_layout.is_contiguous()
            && k_cumdecay_layout.is_contiguous()
            && q_state_layout.is_contiguous()
            && local_attn_layout.is_contiguous()
            && state_decay_layout.is_contiguous()
            && value_layout.is_contiguous())
        {
            candle::bail!("delta-full-scan requires contiguous inputs")
        }

        let (batch_heads, k_head_dim, v_head_dim) = initial_layout.shape().dims3()?;
        let (weighted_key_bh, num_chunks, chunk_size, weighted_key_width) =
            weighted_key_layout.shape().dims4()?;
        let (k_cumdecay_bh, k_cumdecay_num_chunks, k_cumdecay_chunk_size, k_cumdecay_width) =
            k_cumdecay_layout.shape().dims4()?;
        let (q_state_bh, q_state_num_chunks, q_state_chunk_size, q_state_width) =
            q_state_layout.shape().dims4()?;
        let (local_attn_bh, local_attn_num_chunks, local_attn_chunk_size, local_attn_width) =
            local_attn_layout.shape().dims4()?;
        let (state_decay_bh, state_decay_num_chunks) = state_decay_layout.shape().dims2()?;
        let (value_bh, value_num_chunks, value_chunk_size, value_v_head_dim) =
            value_layout.shape().dims4()?;
        if weighted_key_bh != batch_heads
            || k_cumdecay_bh != batch_heads
            || q_state_bh != batch_heads
            || local_attn_bh != batch_heads
            || state_decay_bh != batch_heads
            || value_bh != batch_heads
            || k_cumdecay_num_chunks != num_chunks
            || q_state_num_chunks != num_chunks
            || local_attn_num_chunks != num_chunks
            || state_decay_num_chunks != num_chunks
            || value_num_chunks != num_chunks
            || k_cumdecay_chunk_size != chunk_size
            || q_state_chunk_size != chunk_size
            || local_attn_chunk_size != chunk_size
            || value_chunk_size != chunk_size
            || weighted_key_width != k_head_dim
            || k_cumdecay_width != k_head_dim
            || q_state_width != k_head_dim
            || local_attn_width != chunk_size
            || value_v_head_dim != v_head_dim
        {
            candle::bail!(
                "delta-full-scan shape mismatch: initial={:?} weighted_key={:?} k_cumdecay={:?} q_state={:?} local_attn={:?} state_decay={:?} value={:?}",
                initial_layout.shape().dims(),
                weighted_key_layout.shape().dims(),
                k_cumdecay_layout.shape().dims(),
                q_state_layout.shape().dims(),
                local_attn_layout.shape().dims(),
                state_decay_layout.shape().dims(),
                value_layout.shape().dims()
            )
        }

        let device = initial_state.device().clone();
        let out_shape = candle::Shape::from_dims(&[
            batch_heads,
            num_chunks * chunk_size + k_head_dim,
            v_head_dim,
        ]);
        let elem_count = out_shape.elem_count();
        let total_threads = batch_heads * v_head_dim;
        let cfg = LaunchConfig::for_num_elems(total_threads as u32);

        macro_rules! launch {
            ($ty:ty, $kernel:expr) => {{
                let initial_state = initial_state.as_cuda_slice::<$ty>()?;
                let initial_state = match initial_layout.contiguous_offsets() {
                    Some((o1, o2)) => initial_state.slice(o1..o2),
                    None => candle::bail!("delta-full-scan requires contiguous inputs"),
                };
                let weighted_key_scan = weighted_key_scan.as_cuda_slice::<$ty>()?;
                let weighted_key_scan = match weighted_key_layout.contiguous_offsets() {
                    Some((o1, o2)) => weighted_key_scan.slice(o1..o2),
                    None => candle::bail!("delta-full-scan requires contiguous inputs"),
                };
                let k_cumdecay_scan = k_cumdecay_scan.as_cuda_slice::<$ty>()?;
                let k_cumdecay_scan = match k_cumdecay_layout.contiguous_offsets() {
                    Some((o1, o2)) => k_cumdecay_scan.slice(o1..o2),
                    None => candle::bail!("delta-full-scan requires contiguous inputs"),
                };
                let q_state_scan = q_state_scan.as_cuda_slice::<$ty>()?;
                let q_state_scan = match q_state_layout.contiguous_offsets() {
                    Some((o1, o2)) => q_state_scan.slice(o1..o2),
                    None => candle::bail!("delta-full-scan requires contiguous inputs"),
                };
                let local_attn_scan = local_attn_scan.as_cuda_slice::<$ty>()?;
                let local_attn_scan = match local_attn_layout.contiguous_offsets() {
                    Some((o1, o2)) => local_attn_scan.slice(o1..o2),
                    None => candle::bail!("delta-full-scan requires contiguous inputs"),
                };
                let state_decay_scan = state_decay_scan.as_cuda_slice::<$ty>()?;
                let state_decay_scan = match state_decay_layout.contiguous_offsets() {
                    Some((o1, o2)) => state_decay_scan.slice(o1..o2),
                    None => candle::bail!("delta-full-scan requires contiguous inputs"),
                };
                let value = value.as_cuda_slice::<$ty>()?;
                let value = match value_layout.contiguous_offsets() {
                    Some((o1, o2)) => value.slice(o1..o2),
                    None => candle::bail!("delta-full-scan requires contiguous inputs"),
                };
                let output = unsafe { device.alloc::<$ty>(elem_count) }?;
                let func = device
                    .get_or_load_func($kernel, &candle::cuda_backend::kernels::QWEN35_DELTA)?;
                let mut builder = func.builder();
                candle::builder_arg!(
                    builder,
                    batch_heads,
                    num_chunks,
                    chunk_size,
                    k_head_dim,
                    v_head_dim
                );
                builder.arg(&initial_state);
                builder.arg(&weighted_key_scan);
                builder.arg(&k_cumdecay_scan);
                builder.arg(&q_state_scan);
                builder.arg(&local_attn_scan);
                builder.arg(&state_decay_scan);
                builder.arg(&value);
                builder.arg(&output);
                unsafe { builder.launch(cfg) }.w()?;
                let storage = candle::CudaStorage::wrap_cuda_slice(output, device.clone());
                Ok((storage, out_shape.clone()))
            }};
        }

        match initial_state.dtype() {
            DType::F16 => launch!(half::f16, "delta_full_scan_f16"),
            DType::F32 => launch!(f32, "delta_full_scan_f32"),
            DType::BF16 => launch!(half::bf16, "delta_full_scan_bf16"),
            other => candle::bail!("delta-full-scan unsupported dtype {other:?}"),
        }
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        initial_state: &candle::MetalStorage,
        initial_layout: &candle::Layout,
        weighted_key_scan: &candle::MetalStorage,
        weighted_key_layout: &candle::Layout,
        k_cumdecay_scan: &candle::MetalStorage,
        k_cumdecay_layout: &candle::Layout,
        q_state_scan: &candle::MetalStorage,
        q_state_layout: &candle::Layout,
        local_attn_scan: &candle::MetalStorage,
        local_attn_layout: &candle::Layout,
        state_decay_scan: &candle::MetalStorage,
        state_decay_layout: &candle::Layout,
        value: &candle::MetalStorage,
        value_layout: &candle::Layout,
    ) -> Result<(candle::MetalStorage, candle::Shape)> {
        use candle::backend::BackendStorage;
        use candle::MetalError;

        if !(initial_layout.is_contiguous()
            && weighted_key_layout.is_contiguous()
            && k_cumdecay_layout.is_contiguous()
            && q_state_layout.is_contiguous()
            && local_attn_layout.is_contiguous()
            && state_decay_layout.is_contiguous()
            && value_layout.is_contiguous())
        {
            candle::bail!("delta-full-scan requires contiguous inputs")
        }

        let (batch_heads, k_head_dim, v_head_dim) = initial_layout.shape().dims3()?;
        let (weighted_key_bh, num_chunks, chunk_size, weighted_key_width) =
            weighted_key_layout.shape().dims4()?;
        let (k_cumdecay_bh, k_cumdecay_num_chunks, k_cumdecay_chunk_size, k_cumdecay_width) =
            k_cumdecay_layout.shape().dims4()?;
        let (q_state_bh, q_state_num_chunks, q_state_chunk_size, q_state_width) =
            q_state_layout.shape().dims4()?;
        let (local_attn_bh, local_attn_num_chunks, local_attn_chunk_size, local_attn_width) =
            local_attn_layout.shape().dims4()?;
        let (state_decay_bh, state_decay_num_chunks) = state_decay_layout.shape().dims2()?;
        let (value_bh, value_num_chunks, value_chunk_size, value_v_head_dim) =
            value_layout.shape().dims4()?;
        if weighted_key_bh != batch_heads
            || k_cumdecay_bh != batch_heads
            || q_state_bh != batch_heads
            || local_attn_bh != batch_heads
            || state_decay_bh != batch_heads
            || value_bh != batch_heads
            || k_cumdecay_num_chunks != num_chunks
            || q_state_num_chunks != num_chunks
            || local_attn_num_chunks != num_chunks
            || state_decay_num_chunks != num_chunks
            || value_num_chunks != num_chunks
            || k_cumdecay_chunk_size != chunk_size
            || q_state_chunk_size != chunk_size
            || local_attn_chunk_size != chunk_size
            || value_chunk_size != chunk_size
            || weighted_key_width != k_head_dim
            || k_cumdecay_width != k_head_dim
            || q_state_width != k_head_dim
            || local_attn_width != chunk_size
            || value_v_head_dim != v_head_dim
        {
            candle::bail!(
                "delta-full-scan shape mismatch: initial={:?} weighted_key={:?} k_cumdecay={:?} q_state={:?} local_attn={:?} state_decay={:?} value={:?}",
                initial_layout.shape().dims(),
                weighted_key_layout.shape().dims(),
                k_cumdecay_layout.shape().dims(),
                q_state_layout.shape().dims(),
                local_attn_layout.shape().dims(),
                state_decay_layout.shape().dims(),
                value_layout.shape().dims()
            )
        }

        let device = initial_state.device();
        let dtype = match initial_state.dtype() {
            DType::F16 => candle_metal_kernels::DType::F16,
            DType::F32 => candle_metal_kernels::DType::F32,
            DType::BF16 => candle_metal_kernels::DType::BF16,
            other => candle::bail!("delta-full-scan unsupported dtype {other:?}"),
        };
        let out_shape = candle::Shape::from_dims(&[
            batch_heads,
            num_chunks * chunk_size + k_head_dim,
            v_head_dim,
        ]);
        let elem_count = out_shape.elem_count();
        let output = device.new_buffer(elem_count, initial_state.dtype(), "delta-full-scan")?;
        let encoder = device.command_encoder()?;
        encoder.set_label("delta-full-scan");
        let initial = candle_metal_kernels::BufferOffset {
            buffer: initial_state.buffer(),
            offset_in_bytes: initial_layout.start_offset() * initial_state.dtype().size_in_bytes(),
        };
        let weighted_key = candle_metal_kernels::BufferOffset {
            buffer: weighted_key_scan.buffer(),
            offset_in_bytes: weighted_key_layout.start_offset()
                * weighted_key_scan.dtype().size_in_bytes(),
        };
        let k_cumdecay = candle_metal_kernels::BufferOffset {
            buffer: k_cumdecay_scan.buffer(),
            offset_in_bytes: k_cumdecay_layout.start_offset()
                * k_cumdecay_scan.dtype().size_in_bytes(),
        };
        let q_state = candle_metal_kernels::BufferOffset {
            buffer: q_state_scan.buffer(),
            offset_in_bytes: q_state_layout.start_offset() * q_state_scan.dtype().size_in_bytes(),
        };
        let local_attn = candle_metal_kernels::BufferOffset {
            buffer: local_attn_scan.buffer(),
            offset_in_bytes: local_attn_layout.start_offset()
                * local_attn_scan.dtype().size_in_bytes(),
        };
        let state_decay = candle_metal_kernels::BufferOffset {
            buffer: state_decay_scan.buffer(),
            offset_in_bytes: state_decay_layout.start_offset()
                * state_decay_scan.dtype().size_in_bytes(),
        };
        let v = candle_metal_kernels::BufferOffset {
            buffer: value.buffer(),
            offset_in_bytes: value_layout.start_offset() * value.dtype().size_in_bytes(),
        };
        candle_metal_kernels::call_delta_full_scan(
            device.metal_device(),
            &encoder,
            device.kernels(),
            dtype,
            batch_heads,
            num_chunks,
            chunk_size,
            k_head_dim,
            v_head_dim,
            initial,
            weighted_key,
            k_cumdecay,
            q_state,
            local_attn,
            state_decay,
            v,
            &output,
        )
        .map_err(MetalError::from)?;
        let storage =
            candle::MetalStorage::new(output, device.clone(), elem_count, initial_state.dtype());
        Ok((storage, out_shape))
    }

    #[cfg(feature = "hip")]
    fn hip_fwd(
        &self,
        initial_state: &candle::HipStorage,
        initial_layout: &candle::Layout,
        weighted_key_scan: &candle::HipStorage,
        weighted_key_layout: &candle::Layout,
        k_cumdecay_scan: &candle::HipStorage,
        k_cumdecay_layout: &candle::Layout,
        q_state_scan: &candle::HipStorage,
        q_state_layout: &candle::Layout,
        local_attn_scan: &candle::HipStorage,
        local_attn_layout: &candle::Layout,
        state_decay_scan: &candle::HipStorage,
        state_decay_layout: &candle::Layout,
        value: &candle::HipStorage,
        value_layout: &candle::Layout,
    ) -> Result<(candle::HipStorage, candle::Shape)> {
        use candle::backend::BackendStorage;
        use std::ffi::c_void;

        if !(initial_layout.is_contiguous()
            && weighted_key_layout.is_contiguous()
            && k_cumdecay_layout.is_contiguous()
            && q_state_layout.is_contiguous()
            && local_attn_layout.is_contiguous()
            && state_decay_layout.is_contiguous()
            && value_layout.is_contiguous())
        {
            candle::bail!("delta-full-scan requires contiguous inputs")
        }

        let (batch_heads, k_head_dim, v_head_dim) = initial_layout.shape().dims3()?;
        let (weighted_key_bh, num_chunks, chunk_size, weighted_key_width) =
            weighted_key_layout.shape().dims4()?;
        let (k_cumdecay_bh, k_cumdecay_num_chunks, k_cumdecay_chunk_size, k_cumdecay_width) =
            k_cumdecay_layout.shape().dims4()?;
        let (q_state_bh, q_state_num_chunks, q_state_chunk_size, q_state_width) =
            q_state_layout.shape().dims4()?;
        let (local_attn_bh, local_attn_num_chunks, local_attn_chunk_size, local_attn_width) =
            local_attn_layout.shape().dims4()?;
        let (state_decay_bh, state_decay_num_chunks) = state_decay_layout.shape().dims2()?;
        let (value_bh, value_num_chunks, value_chunk_size, value_v_head_dim) =
            value_layout.shape().dims4()?;
        if weighted_key_bh != batch_heads
            || k_cumdecay_bh != batch_heads
            || q_state_bh != batch_heads
            || local_attn_bh != batch_heads
            || state_decay_bh != batch_heads
            || value_bh != batch_heads
            || k_cumdecay_num_chunks != num_chunks
            || q_state_num_chunks != num_chunks
            || local_attn_num_chunks != num_chunks
            || state_decay_num_chunks != num_chunks
            || value_num_chunks != num_chunks
            || k_cumdecay_chunk_size != chunk_size
            || q_state_chunk_size != chunk_size
            || local_attn_chunk_size != chunk_size
            || value_chunk_size != chunk_size
            || weighted_key_width != k_head_dim
            || k_cumdecay_width != k_head_dim
            || q_state_width != k_head_dim
            || local_attn_width != chunk_size
            || value_v_head_dim != v_head_dim
        {
            candle::bail!(
                "delta-full-scan shape mismatch: initial={:?} weighted_key={:?} k_cumdecay={:?} q_state={:?} local_attn={:?} state_decay={:?} value={:?}",
                initial_layout.shape().dims(),
                weighted_key_layout.shape().dims(),
                k_cumdecay_layout.shape().dims(),
                q_state_layout.shape().dims(),
                local_attn_layout.shape().dims(),
                state_decay_layout.shape().dims(),
                value_layout.shape().dims()
            )
        }

        let device = initial_state.device().clone();
        let storage_dtype = initial_state.dtype();
        let dtype_code = candle::hip::qwen35_dtype_code(storage_dtype)?;
        let out_shape = candle::Shape::from_dims(&[
            batch_heads,
            num_chunks * chunk_size + k_head_dim,
            v_head_dim,
        ]);
        let elem_count = out_shape.elem_count();

        macro_rules! launch {
            ($ty:ty, $zero:expr) => {{
                let initial_state = initial_state.cpu_storage().as_slice::<$ty>()?;
                let initial_state = match initial_layout.contiguous_offsets() {
                    Some((o1, o2)) => &initial_state[o1..o2],
                    None => candle::bail!("delta-full-scan requires contiguous inputs"),
                };
                let weighted_key_scan = weighted_key_scan.cpu_storage().as_slice::<$ty>()?;
                let weighted_key_scan = match weighted_key_layout.contiguous_offsets() {
                    Some((o1, o2)) => &weighted_key_scan[o1..o2],
                    None => candle::bail!("delta-full-scan requires contiguous inputs"),
                };
                let k_cumdecay_scan = k_cumdecay_scan.cpu_storage().as_slice::<$ty>()?;
                let k_cumdecay_scan = match k_cumdecay_layout.contiguous_offsets() {
                    Some((o1, o2)) => &k_cumdecay_scan[o1..o2],
                    None => candle::bail!("delta-full-scan requires contiguous inputs"),
                };
                let q_state_scan = q_state_scan.cpu_storage().as_slice::<$ty>()?;
                let q_state_scan = match q_state_layout.contiguous_offsets() {
                    Some((o1, o2)) => &q_state_scan[o1..o2],
                    None => candle::bail!("delta-full-scan requires contiguous inputs"),
                };
                let local_attn_scan = local_attn_scan.cpu_storage().as_slice::<$ty>()?;
                let local_attn_scan = match local_attn_layout.contiguous_offsets() {
                    Some((o1, o2)) => &local_attn_scan[o1..o2],
                    None => candle::bail!("delta-full-scan requires contiguous inputs"),
                };
                let state_decay_scan = state_decay_scan.cpu_storage().as_slice::<$ty>()?;
                let state_decay_scan = match state_decay_layout.contiguous_offsets() {
                    Some((o1, o2)) => &state_decay_scan[o1..o2],
                    None => candle::bail!("delta-full-scan requires contiguous inputs"),
                };
                let value = value.cpu_storage().as_slice::<$ty>()?;
                let value = match value_layout.contiguous_offsets() {
                    Some((o1, o2)) => &value[o1..o2],
                    None => candle::bail!("delta-full-scan requires contiguous inputs"),
                };
                let mut output = vec![$zero; elem_count];
                let status = unsafe {
                    candle::hip::ffi::qwen35_hip_delta_full_scan(
                        dtype_code,
                        batch_heads,
                        num_chunks,
                        chunk_size,
                        k_head_dim,
                        v_head_dim,
                        initial_state.as_ptr() as *const c_void,
                        weighted_key_scan.as_ptr() as *const c_void,
                        k_cumdecay_scan.as_ptr() as *const c_void,
                        q_state_scan.as_ptr() as *const c_void,
                        local_attn_scan.as_ptr() as *const c_void,
                        state_decay_scan.as_ptr() as *const c_void,
                        value.as_ptr() as *const c_void,
                        output.as_mut_ptr() as *mut c_void,
                    )
                };
                if status != 0 {
                    return Err(candle::hip::qwen35_error(self.name(), status));
                }
                let storage = <$ty as candle::WithDType>::to_cpu_storage_owned(output);
                Ok((
                    candle::HipStorage::wrap_cpu_storage(storage, device.clone()),
                    out_shape.clone(),
                ))
            }};
        }

        match storage_dtype {
            DType::F16 => launch!(half::f16, half::f16::from_bits(0)),
            DType::F32 => launch!(f32, 0.0f32),
            DType::BF16 => launch!(half::bf16, half::bf16::from_bits(0)),
            other => candle::bail!("delta-full-scan unsupported dtype {other:?}"),
        }
    }
}

fn delta_full_scan(
    initial_state: &Tensor,
    weighted_key_scan: &Tensor,
    k_cumdecay_scan: &Tensor,
    q_state_scan: &Tensor,
    local_attn_scan: &Tensor,
    state_decay_scan: &Tensor,
    value: &Tensor,
) -> Result<Tensor> {
    initial_state.apply_op7_no_bwd(
        weighted_key_scan,
        k_cumdecay_scan,
        q_state_scan,
        local_attn_scan,
        state_decay_scan,
        value,
        &DeltaFullScan,
    )
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DeltaNetScanMode {
    Flat3d,
    HoistedDecays,
    PrebatchedLocal,
    TorchLike,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct DeltaNetExecutionPolicy {
    scan_mode: DeltaNetScanMode,
    use_flattened_solve: bool,
}

fn parse_delta_net_scan_mode(raw_value: &str) -> Option<DeltaNetScanMode> {
    match raw_value.trim() {
        "flat3d" => Some(DeltaNetScanMode::Flat3d),
        "hoisted-decays" => Some(DeltaNetScanMode::HoistedDecays),
        "prebatched-local" => Some(DeltaNetScanMode::PrebatchedLocal),
        "torch-like" => Some(DeltaNetScanMode::TorchLike),
        _ => None,
    }
}

fn debug_delta_scan_policy(sequence_length: usize, policy: DeltaNetExecutionPolicy) {
    static LOGGED: AtomicBool = AtomicBool::new(false);
    if std::env::var("CANDLE_QWEN35_DEBUG_DELTA_SCAN").is_ok()
        && !LOGGED.swap(true, Ordering::Relaxed)
    {
        eprintln!(
            "qwen3.5 delta scan policy: sequence_length={} mode={:?} flattened_solve={}",
            sequence_length, policy.scan_mode, policy.use_flattened_solve
        );
    }
}

fn recommended_delta_net_execution_policy(
    device: &Device,
    sequence_length: usize,
    num_chunks: usize,
) -> DeltaNetExecutionPolicy {
    let long_metal_context = matches!(device.location(), DeviceLocation::Metal { .. })
        && (sequence_length >= 2048 || num_chunks >= 64);
    DeltaNetExecutionPolicy {
        scan_mode: if long_metal_context {
            DeltaNetScanMode::PrebatchedLocal
        } else {
            DeltaNetScanMode::Flat3d
        },
        use_flattened_solve: long_metal_context,
    }
}

fn delta_net_execution_policy(
    device: &Device,
    sequence_length: usize,
    num_chunks: usize,
) -> DeltaNetExecutionPolicy {
    let mut policy = recommended_delta_net_execution_policy(device, sequence_length, num_chunks);
    if let Ok(raw_value) = std::env::var("CANDLE_QWEN35_DELTA_SCAN_MODE") {
        if let Some(mode) = parse_delta_net_scan_mode(&raw_value) {
            policy.scan_mode = mode;
        }
    }
    debug_delta_scan_policy(sequence_length, policy);
    policy
}

fn parse_usize_env(name: &str) -> Option<usize> {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
}

fn full_attention_blockwise_tiles(
    device: &Device,
    q_len: usize,
    kv_len: usize,
) -> Option<(usize, usize)> {
    if !matches!(device.location(), DeviceLocation::Metal { .. }) || q_len <= 1 || kv_len <= 1 {
        return None;
    }
    let enabled = matches!(
        std::env::var("CANDLE_QWEN35_FULL_BLOCKWISE_ATTN").as_deref(),
        Ok("1" | "true" | "TRUE" | "yes" | "YES")
    );
    if !enabled {
        return None;
    }
    let q_block = parse_usize_env("CANDLE_QWEN35_FULL_ATTN_Q_BLOCK").unwrap_or(128);
    let k_block = parse_usize_env("CANDLE_QWEN35_FULL_ATTN_K_BLOCK").unwrap_or(512);
    Some((q_block.min(q_len), k_block.min(kv_len)))
}

fn full_attention_sdpa_q_block(device: &Device, q_len: usize) -> Option<usize> {
    if !matches!(device.location(), DeviceLocation::Metal { .. }) || q_len <= 1 {
        return None;
    }
    let enabled = matches!(
        std::env::var("CANDLE_QWEN35_FULL_SDPA_CHUNKED").as_deref(),
        Ok("1" | "true" | "TRUE" | "yes" | "YES")
    );
    if !enabled {
        return None;
    }
    Some(
        parse_usize_env("CANDLE_QWEN35_FULL_SDPA_Q_BLOCK")
            .unwrap_or(128)
            .min(q_len),
    )
}

fn use_full_attention_torchlike_eager(device: &Device) -> bool {
    matches!(device.location(), DeviceLocation::Metal { .. })
        && matches!(
            std::env::var("CANDLE_QWEN35_FULL_EAGER_TORCHLIKE").as_deref(),
            Ok("1" | "true" | "TRUE" | "yes" | "YES")
        )
}

fn delta_net_compute_dtype(scan_mode: DeltaNetScanMode, initial_dtype: DType) -> DType {
    match scan_mode {
        DeltaNetScanMode::TorchLike => DType::F32,
        _ => initial_dtype,
    }
}

#[derive(Debug, Clone)]
struct FullAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: Qwen35RmsNorm,
    k_norm: Qwen35RmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    attention_size: usize,
    rotary_emb: Arc<RotaryEmbedding>,
    kv_cache: Option<(Tensor, Tensor)>,
}

impl FullAttention {
    fn causal_block_mask(
        device: &Device,
        q_start: usize,
        q_len: usize,
        k_start: usize,
        k_len: usize,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        let query_base = seqlen_offset + q_start;
        let mut mask = Vec::with_capacity(q_len * k_len);
        for q_idx in 0..q_len {
            let q_abs = query_base + q_idx;
            for k_idx in 0..k_len {
                let k_abs = k_start + k_idx;
                mask.push(if k_abs > q_abs {
                    f32::NEG_INFINITY
                } else {
                    0.0
                });
            }
        }
        Tensor::from_slice(&mask, (q_len, k_len), device)?.reshape((1, 1, q_len, k_len))
    }

    fn blockwise_attention_profiled(
        &self,
        query_states_f: &Tensor,
        key_states_f: &Tensor,
        value_states_f: &Tensor,
        scale: f64,
        seqlen_offset: usize,
        q_block_size: usize,
        k_block_size: usize,
        profile: &mut RuntimeProfile,
    ) -> Result<Tensor> {
        let device = query_states_f.device();
        let (b_sz, q_heads, q_len, _head_dim) = query_states_f.dims4()?;
        let (_, _, kv_len, value_dim) = value_states_f.dims4()?;
        let mut q_outputs = Vec::with_capacity(q_len.div_ceil(q_block_size));
        for q_start in (0..q_len).step_by(q_block_size) {
            let q_block_len = (q_len - q_start).min(q_block_size);
            let q_block = query_states_f.narrow(2, q_start, q_block_len)?;
            let mut running_max =
                Tensor::full(f32::NEG_INFINITY, (b_sz, q_heads, q_block_len, 1), device)?;
            let mut running_sum =
                Tensor::zeros((b_sz, q_heads, q_block_len, 1), DType::F32, device)?;
            let mut running_acc =
                Tensor::zeros((b_sz, q_heads, q_block_len, value_dim), DType::F32, device)?;

            for k_start in (0..kv_len).step_by(k_block_size) {
                let k_block_len = (kv_len - k_start).min(k_block_size);
                let q_abs_min = seqlen_offset + q_start;
                let q_abs_max = q_abs_min + q_block_len - 1;
                let k_abs_min = k_start;
                let k_abs_max = k_start + k_block_len - 1;
                if k_abs_min > q_abs_max {
                    break;
                }

                let score_start = profile_start(device)?;
                let k_block = key_states_f.narrow(2, k_start, k_block_len)?;
                let mut scores = (q_block.matmul(&k_block.transpose(2, 3)?)? * scale)?;
                let needs_partial_mask = !(k_abs_max <= q_abs_min);
                if needs_partial_mask {
                    let mask = Self::causal_block_mask(
                        device,
                        q_start,
                        q_block_len,
                        k_start,
                        k_block_len,
                        seqlen_offset,
                    )?;
                    scores = scores.broadcast_add(&mask)?;
                }
                profile.attention_score_millis += profile_elapsed(score_start, device)?;

                let softmax_start = profile_start(device)?;
                let block_max = scores.max_keepdim(D::Minus1)?;
                let new_max = running_max.maximum(&block_max)?;
                let prev_scale = running_max.broadcast_sub(&new_max)?.exp()?;
                let exp_scores = scores.broadcast_sub(&new_max)?.exp()?;
                let new_sum = running_sum
                    .broadcast_mul(&prev_scale)?
                    .broadcast_add(&exp_scores.sum_keepdim(D::Minus1)?)?;
                profile.attention_softmax_millis += profile_elapsed(softmax_start, device)?;

                let mix_start = profile_start(device)?;
                let v_block = value_states_f.narrow(2, k_start, k_block_len)?;
                let new_acc = running_acc
                    .broadcast_mul(&prev_scale)?
                    .broadcast_add(&exp_scores.matmul(&v_block)?)?;
                running_max = new_max;
                running_sum = new_sum;
                running_acc = new_acc;
                profile.attention_mix_millis += profile_elapsed(mix_start, device)?;
            }

            q_outputs.push(running_acc.broadcast_div(&running_sum)?);
        }

        Tensor::cat(&q_outputs.iter().collect::<Vec<_>>(), 2)
    }

    fn sdpa_chunked_attention_profiled(
        &self,
        query_states: &Tensor,
        key_states: &Tensor,
        value_states: &Tensor,
        scale: f32,
        seqlen_offset: usize,
        q_block_size: usize,
        profile: &mut RuntimeProfile,
    ) -> Result<Tensor> {
        let device = query_states.device();
        let (b_sz, q_heads, q_len, _) = query_states.dims4()?;
        let (_, kv_heads, kv_len, _) = key_states.dims4()?;
        let base_head_chunk = parse_usize_env("CANDLE_QWEN35_FULL_SDPA_Q_HEADS")
            .unwrap_or(q_heads)
            .min(q_heads)
            .max(self.num_kv_groups);
        let q_head_chunk =
            ((base_head_chunk / self.num_kv_groups).max(1) * self.num_kv_groups).min(q_heads);
        let kv_head_chunk = q_head_chunk / self.num_kv_groups;
        if kv_head_chunk == 0 || kv_head_chunk > kv_heads {
            candle::bail!("invalid sdpa q_head chunk for grouped attention")
        }

        let mut q_outputs = Vec::with_capacity(q_len.div_ceil(q_block_size));
        for q_start in (0..q_len).step_by(q_block_size) {
            let q_block_len = (q_len - q_start).min(q_block_size);
            let mask_base =
                Self::causal_block_mask(device, q_start, q_block_len, 0, kv_len, seqlen_offset)?
                    .to_dtype(query_states.dtype())?;
            let mut head_outputs = Vec::with_capacity(q_heads.div_ceil(q_head_chunk));
            for q_head_start in (0..q_heads).step_by(q_head_chunk) {
                let q_head_len = (q_heads - q_head_start).min(q_head_chunk);
                let kv_head_start = q_head_start / self.num_kv_groups;
                let kv_head_len = q_head_len / self.num_kv_groups;
                let q_chunk = query_states
                    .narrow(1, q_head_start, q_head_len)?
                    .narrow(2, q_start, q_block_len)?
                    .contiguous()?;
                let k_chunk = key_states
                    .narrow(1, kv_head_start, kv_head_len)?
                    .contiguous()?;
                let v_chunk = value_states
                    .narrow(1, kv_head_start, kv_head_len)?
                    .contiguous()?;
                let mask = mask_base.broadcast_as((b_sz, q_head_len, q_block_len, kv_len))?;
                let fused_start = profile_start(device)?;
                let output =
                    ops::sdpa(&q_chunk, &k_chunk, &v_chunk, Some(&mask), false, scale, 1.0)?;
                let fused_elapsed = profile_elapsed(fused_start, device)?;
                profile.attention_mix_millis += fused_elapsed;
                head_outputs.push(output);
            }
            q_outputs.push(Tensor::cat(&head_outputs.iter().collect::<Vec<_>>(), 1)?);
        }

        Tensor::cat(&q_outputs.iter().collect::<Vec<_>>(), 2)
    }

    fn grouped_torchlike_eager_attention_profiled(
        &self,
        query_states: &Tensor,
        key_states: &Tensor,
        value_states: &Tensor,
        attention_mask: Option<&Tensor>,
        scale: f64,
        profile: &mut RuntimeProfile,
    ) -> Result<Tensor> {
        let device = query_states.device();
        let compute_dtype = query_states.dtype();
        let (_, q_heads, _, _) = query_states.dims4()?;
        let (_, kv_heads, _, _) = key_states.dims4()?;
        let mask_start = profile_start(device)?;
        let mask = attention_mask
            .map(|mask| mask.to_dtype(compute_dtype))
            .transpose()?;
        profile.full_attention_mask_prepare_millis += profile_elapsed(mask_start, device)?;
        let mut outputs = Vec::with_capacity(kv_heads);

        for kv_head_idx in 0..kv_heads {
            let q_head_start = kv_head_idx * self.num_kv_groups;
            let q_head_len = (q_heads - q_head_start).min(self.num_kv_groups);
            if q_head_len == 0 {
                break;
            }
            let query_layout_start = profile_start(device)?;
            let q_chunk = query_states
                .narrow(1, q_head_start, q_head_len)?
                .contiguous()?;
            profile.full_attention_input_layout_millis +=
                profile_elapsed(query_layout_start, device)?;
            let kv_len = key_states.dim(2)?;
            let value_dim = value_states.dim(3)?;
            let kv_materialize_start = profile_start(device)?;
            let k_chunk = key_states.narrow(1, kv_head_idx, 1)?.broadcast_as((
                q_chunk.dim(0)?,
                q_head_len,
                kv_len,
                self.head_dim,
            ))?;
            let v_chunk = value_states.narrow(1, kv_head_idx, 1)?.broadcast_as((
                q_chunk.dim(0)?,
                q_head_len,
                kv_len,
                value_dim,
            ))?;
            let key_states_t = k_chunk.transpose(2, 3)?.contiguous()?;
            profile.full_attention_kv_materialize_millis +=
                profile_elapsed(kv_materialize_start, device)?;

            let score_start = profile_start(device)?;
            let mut attn_weights = (q_chunk.matmul(&key_states_t)? * scale)?;
            if let Some(mask) = &mask {
                attn_weights = attn_weights.broadcast_add(mask)?;
            }
            profile.attention_score_millis += profile_elapsed(score_start, device)?;

            let softmax_start = profile_start(device)?;
            let attn_weights = ops::softmax_last_dim(&attn_weights.to_dtype(DType::F32)?)?
                .to_dtype(compute_dtype)?;
            profile.attention_softmax_millis += profile_elapsed(softmax_start, device)?;

            let mix_start = profile_start(device)?;
            let attn_output = attn_weights.matmul(&v_chunk)?;
            profile.attention_mix_millis += profile_elapsed(mix_start, device)?;
            outputs.push(attn_output);
        }

        let collect_start = profile_start(device)?;
        let output = Tensor::cat(&outputs.iter().collect::<Vec<_>>(), 1)?;
        profile.full_attention_output_collect_millis += profile_elapsed(collect_start, device)?;
        Ok(output)
    }

    fn new(cfg: &TextConfig, rotary_emb: Arc<RotaryEmbedding>, vb: VarBuilder) -> Result<Self> {
        let q_proj = linear_b(
            cfg.hidden_size,
            cfg.num_attention_heads * cfg.head_dim * 2,
            cfg.attention_bias,
            vb.pp("q_proj"),
        )?;
        let k_proj = linear_b(
            cfg.hidden_size,
            cfg.num_key_value_heads * cfg.head_dim,
            cfg.attention_bias,
            vb.pp("k_proj"),
        )?;
        let v_proj = linear_b(
            cfg.hidden_size,
            cfg.num_key_value_heads * cfg.head_dim,
            cfg.attention_bias,
            vb.pp("v_proj"),
        )?;
        let o_proj = linear_b(
            cfg.num_attention_heads * cfg.head_dim,
            cfg.hidden_size,
            cfg.attention_bias,
            vb.pp("o_proj"),
        )?;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm: Qwen35RmsNorm::new(cfg.head_dim, cfg.rms_norm_eps, vb.pp("q_norm"))?,
            k_norm: Qwen35RmsNorm::new(cfg.head_dim, cfg.rms_norm_eps, vb.pp("k_norm"))?,
            num_heads: cfg.num_attention_heads,
            num_kv_heads: cfg.num_key_value_heads,
            num_kv_groups: cfg.num_attention_heads / cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
            attention_size: cfg.num_attention_heads * cfg.head_dim,
            rotary_emb,
            kv_cache: None,
        })
    }

    fn forward_profiled(
        &mut self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
    ) -> Result<(Tensor, RuntimeProfile)> {
        let device = xs.device();
        let full_start = profile_start(device)?;
        let mut profile = RuntimeProfile::default();
        let (b_sz, q_len, _) = xs.dims3()?;
        let qkv_start = profile_start(device)?;
        let q_and_gate =
            self.q_proj
                .forward(xs)?
                .reshape((b_sz, q_len, self.num_heads, self.head_dim * 2))?;
        let query_states = q_and_gate
            .narrow(D::Minus1, 0, self.head_dim)?
            .apply(&self.q_norm)?
            .transpose(1, 2)?;
        let gate = q_and_gate
            .narrow(D::Minus1, self.head_dim, self.head_dim)?
            .reshape((b_sz, q_len, self.num_heads * self.head_dim))?;

        let key_states = self
            .k_proj
            .forward(xs)?
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .apply(&self.k_norm)?
            .transpose(1, 2)?;
        let value_states = self
            .v_proj
            .forward(xs)?
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        profile.qkv_projection_millis += profile_elapsed(qkv_start, device)?;

        let layout_start = profile_start(device)?;
        let (query_states, key_states) =
            self.rotary_emb
                .apply(&query_states, &key_states, seqlen_offset)?;
        profile.layout_prepare_millis += profile_elapsed(layout_start, device)?;

        let kv_append_start = profile_start(device)?;
        let (key_states, value_states) = match &self.kv_cache {
            Some((prev_k, prev_v)) => {
                let prev_k = if prev_k.dtype() == key_states.dtype() {
                    prev_k.clone()
                } else {
                    prev_k.to_dtype(key_states.dtype())?
                };
                let prev_v = if prev_v.dtype() == value_states.dtype() {
                    prev_v.clone()
                } else {
                    prev_v.to_dtype(value_states.dtype())?
                };
                (
                    Tensor::cat(&[&prev_k, &key_states], 2)?,
                    Tensor::cat(&[&prev_v, &value_states], 2)?,
                )
            }
            None => (key_states, value_states),
        };
        profile.kv_append_write_millis += profile_elapsed(kv_append_start, device)?;

        let input_layout_start = profile_start(device)?;
        let query_states = query_states.contiguous()?;
        let key_states = key_states.contiguous()?;
        let value_states = value_states.contiguous()?;
        let scale = 1f64 / f64::sqrt(self.head_dim as f64);
        let input_layout_elapsed = profile_elapsed(input_layout_start, device)?;
        profile.layout_prepare_millis += input_layout_elapsed;
        profile.full_attention_input_layout_millis += input_layout_elapsed;

        let attn_output = if use_full_attention_prefill_megakernel(
            device,
            q_len,
            key_states.dim(2)?,
            seqlen_offset,
        ) {
            let kernel_start = profile_start(device)?;
            let output = full_attention_prefill_megakernel(
                &query_states,
                &key_states,
                &value_states,
                self.num_kv_groups,
                scale as f32,
                seqlen_offset,
            )?
            .to_dtype(DType::F32)?;
            profile.full_attention_kernel_execute_millis += profile_elapsed(kernel_start, device)?;
            output
        } else if use_full_attention_torchlike_eager(device) {
            self.grouped_torchlike_eager_attention_profiled(
                &query_states,
                &key_states,
                &value_states,
                attention_mask,
                scale,
                &mut profile,
            )?
            .to_dtype(DType::F32)?
        } else if let Some(q_block_size) = full_attention_sdpa_q_block(device, q_len) {
            self.sdpa_chunked_attention_profiled(
                &query_states,
                &key_states,
                &value_states,
                scale as f32,
                seqlen_offset,
                q_block_size,
                &mut profile,
            )?
            .to_dtype(DType::F32)?
        } else if matches!(device.location(), DeviceLocation::Metal { .. }) {
            self.grouped_torchlike_eager_attention_profiled(
                &query_states,
                &key_states,
                &value_states,
                attention_mask,
                scale,
                &mut profile,
            )?
            .to_dtype(DType::F32)?
        } else {
            let kv_materialize_start = profile_start(device)?;
            let key_states =
                crate::utils::repeat_kv(key_states.clone(), self.num_kv_groups)?.contiguous()?;
            let value_states =
                crate::utils::repeat_kv(value_states.clone(), self.num_kv_groups)?.contiguous()?;
            let kv_materialize_elapsed = profile_elapsed(kv_materialize_start, device)?;
            profile.layout_prepare_millis += kv_materialize_elapsed;
            profile.full_attention_kv_materialize_millis += kv_materialize_elapsed;

            let query_states_f = query_states.to_dtype(DType::F32)?;
            let key_states_f = key_states.to_dtype(DType::F32)?;
            let value_states_f = value_states.to_dtype(DType::F32)?;
            if let Some((q_block_size, k_block_size)) =
                full_attention_blockwise_tiles(device, q_len, key_states.dim(2)?)
            {
                self.blockwise_attention_profiled(
                    &query_states_f,
                    &key_states_f,
                    &value_states_f,
                    scale,
                    seqlen_offset,
                    q_block_size,
                    k_block_size,
                    &mut profile,
                )?
            } else {
                let key_states_t = key_states_f.transpose(2, 3)?.contiguous()?;
                let score_start = profile_start(device)?;
                let mut attn_weights = (query_states_f.matmul(&key_states_t)? * scale)?;
                if let Some(mask) = attention_mask {
                    attn_weights = attn_weights.broadcast_add(&mask.to_dtype(DType::F32)?)?;
                }
                profile.attention_score_millis += profile_elapsed(score_start, device)?;

                let softmax_start = profile_start(device)?;
                let attn_weights = ops::softmax_last_dim(&attn_weights)?;
                profile.attention_softmax_millis += profile_elapsed(softmax_start, device)?;

                let mix_start = profile_start(device)?;
                let attn_output = attn_weights.matmul(&value_states_f)?;
                profile.attention_mix_millis += profile_elapsed(mix_start, device)?;
                attn_output
            }
        };

        let output_reshape_start = profile_start(device)?;
        let attn_output = attn_output
            .transpose(1, 2)?
            .reshape((b_sz, q_len, self.attention_size))?
            .to_dtype(xs.dtype())?;
        profile.full_attention_output_reshape_millis +=
            profile_elapsed(output_reshape_start, device)?;
        self.kv_cache = Some((key_states, value_states));
        let gate_start = profile_start(device)?;
        let gated = attn_output.broadcast_mul(&ops::sigmoid(&gate)?)?;
        profile.full_attention_gate_millis += profile_elapsed(gate_start, device)?;
        let output_start = profile_start(device)?;
        let output = gated.apply(&self.o_proj)?;
        profile.output_projection_millis += profile_elapsed(output_start, device)?;
        profile.full_attention_millis += profile_elapsed(full_start, device)?;
        Ok((output, profile))
    }

    #[allow(dead_code)]
    fn forward(
        &mut self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        self.forward_profiled(xs, attention_mask, seqlen_offset)
            .map(|(output, _)| output)
    }

    fn clear_kv_cache(&mut self) {
        self.kv_cache = None;
    }
}

#[derive(Debug, Clone)]
struct GatedDeltaNet {
    in_proj_qkv: Linear,
    in_proj_z: Linear,
    in_proj_b: Linear,
    in_proj_a: Linear,
    conv1d: Conv1d,
    dt_bias: Tensor,
    a_log: Tensor,
    norm: Qwen35RmsNormGated,
    out_proj: Linear,
    num_v_heads: usize,
    num_k_heads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
    key_dim: usize,
    value_dim: usize,
    conv_kernel_size: usize,
    conv_state: Option<Tensor>,
    recurrent_state: Option<Tensor>,
    chunk_cache: Option<LinearChunkCache>,
    value_cache: Option<LinearValueCache>,
}

#[derive(Debug, Clone)]
struct LinearChunkCache {
    chunk_size: usize,
    dtype: DType,
    device_location: DeviceLocation,
    lower: Tensor,
    eye: Tensor,
    strict_lower: Tensor,
    lower_2d: Tensor,
}

#[derive(Debug, Clone)]
struct LinearValueCache {
    dtype: DType,
    device_location: DeviceLocation,
    dt_bias: Tensor,
    a_log_exp: Tensor,
}

impl GatedDeltaNet {
    fn new(cfg: &TextConfig, vb: VarBuilder) -> Result<Self> {
        let key_dim = cfg.linear_num_key_heads * cfg.linear_key_head_dim;
        let value_dim = cfg.linear_num_value_heads * cfg.linear_value_head_dim;
        let conv_dim = key_dim * 2 + value_dim;
        let conv1d = conv1d_no_bias(
            conv_dim,
            conv_dim,
            cfg.linear_conv_kernel_dim,
            Conv1dConfig {
                padding: 0,
                groups: conv_dim,
                ..Default::default()
            },
            vb.pp("conv1d"),
        )?;
        Ok(Self {
            in_proj_qkv: linear_no_bias(cfg.hidden_size, conv_dim, vb.pp("in_proj_qkv"))?,
            in_proj_z: linear_no_bias(cfg.hidden_size, value_dim, vb.pp("in_proj_z"))?,
            in_proj_b: linear_no_bias(
                cfg.hidden_size,
                cfg.linear_num_value_heads,
                vb.pp("in_proj_b"),
            )?,
            in_proj_a: linear_no_bias(
                cfg.hidden_size,
                cfg.linear_num_value_heads,
                vb.pp("in_proj_a"),
            )?,
            conv1d,
            dt_bias: vb.get(cfg.linear_num_value_heads, "dt_bias")?,
            a_log: vb.get(cfg.linear_num_value_heads, "A_log")?,
            norm: Qwen35RmsNormGated::new(
                cfg.linear_value_head_dim,
                cfg.rms_norm_eps,
                vb.pp("norm"),
            )?,
            out_proj: linear_no_bias(value_dim, cfg.hidden_size, vb.pp("out_proj"))?,
            num_v_heads: cfg.linear_num_value_heads,
            num_k_heads: cfg.linear_num_key_heads,
            head_k_dim: cfg.linear_key_head_dim,
            head_v_dim: cfg.linear_value_head_dim,
            key_dim,
            value_dim,
            conv_kernel_size: cfg.linear_conv_kernel_dim,
            conv_state: None,
            recurrent_state: None,
            chunk_cache: None,
            value_cache: None,
        })
    }

    fn value_cache(&mut self, device: &Device, dtype: DType) -> Result<(Tensor, Tensor)> {
        let device_location = device.location();
        let rebuild = self
            .value_cache
            .as_ref()
            .map(|cache| cache.dtype != dtype || cache.device_location != device_location)
            .unwrap_or(true);
        if rebuild {
            let dt_bias = if self.dt_bias.dtype() == dtype {
                self.dt_bias.clone()
            } else {
                self.dt_bias.to_dtype(dtype)?
            }
            .reshape((1, 1, self.num_v_heads))?;
            let a_log = if self.a_log.dtype() == dtype {
                self.a_log.clone()
            } else {
                self.a_log.to_dtype(dtype)?
            };
            let a_log_exp = a_log.exp()?.reshape((1, 1, self.num_v_heads))?;
            self.value_cache = Some(LinearValueCache {
                dtype,
                device_location,
                dt_bias,
                a_log_exp,
            });
        }
        let cache = self
            .value_cache
            .as_ref()
            .expect("linear value cache must be initialized");
        Ok((cache.dt_bias.clone(), cache.a_log_exp.clone()))
    }

    fn chunk_cache(
        &mut self,
        device: &Device,
        dtype: DType,
        chunk_size: usize,
    ) -> Result<LinearChunkCache> {
        let device_location = device.location();
        if let Some(cache) = &self.chunk_cache {
            if cache.chunk_size == chunk_size
                && cache.dtype == dtype
                && cache.device_location == device_location
            {
                return Ok(cache.clone());
            }
        }

        let lower =
            Tensor::tril2(chunk_size, dtype, device)?.reshape((1, 1, 1, chunk_size, chunk_size))?;
        let eye =
            Tensor::eye(chunk_size, dtype, device)?.reshape((1, 1, 1, chunk_size, chunk_size))?;
        let strict_lower = lower.broadcast_sub(&eye)?;
        let lower_2d =
            Tensor::tril2(chunk_size, dtype, device)?.reshape((1, 1, chunk_size, chunk_size))?;
        let cache = LinearChunkCache {
            chunk_size,
            dtype,
            device_location,
            lower,
            eye,
            strict_lower,
            lower_2d,
        };
        self.chunk_cache = Some(cache.clone());
        Ok(cache)
    }

    fn chunk_gated_delta_rule_torch_like(
        &mut self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        g: &Tensor,
        beta: &Tensor,
        _sequence_length: usize,
    ) -> Result<(Tensor, Tensor, RuntimeProfile)> {
        let device = query.device();
        let total_start = profile_start(device)?;
        let mut profile = RuntimeProfile::default();
        let initial_dtype = query.dtype();
        let chunk_size = 64usize;
        let compute_dtype = DType::F32;
        let query_heads = query.dim(2)?;
        let key_heads = key.dim(2)?;
        let value_heads = value.dim(2)?;
        if query_heads != key_heads {
            candle::bail!(
                "chunk_gated_delta_rule_torch_like expected matching query/key head counts, got query_heads={query_heads} key_heads={key_heads}"
            );
        }
        if value_heads % query_heads != 0 {
            candle::bail!(
                "chunk_gated_delta_rule_torch_like expected value heads to be a multiple of query heads, got query_heads={query_heads} value_heads={value_heads}"
            );
        }
        let head_repeat = value_heads / query_heads;
        let query = if head_repeat > 1 {
            repeat_heads(query, head_repeat)?
        } else {
            query.clone()
        };
        let key = if head_repeat > 1 {
            repeat_heads(key, head_repeat)?
        } else {
            key.clone()
        };

        let mut query = query
            .transpose(1, 2)?
            .contiguous()?
            .to_dtype(compute_dtype)?;
        let mut key = key.transpose(1, 2)?.contiguous()?.to_dtype(compute_dtype)?;
        let mut value = value
            .transpose(1, 2)?
            .contiguous()?
            .to_dtype(compute_dtype)?;
        let mut beta = beta
            .transpose(1, 2)?
            .contiguous()?
            .to_dtype(compute_dtype)?;
        let mut g = g.transpose(1, 2)?.contiguous()?.to_dtype(compute_dtype)?;

        let (batch_size, num_heads, sequence_length, k_head_dim) = query.dims4()?;
        let v_head_dim = value.dim(D::Minus1)?;
        let pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size;
        if pad_size > 0 {
            query = query.pad_with_zeros(2, 0, pad_size)?;
            key = key.pad_with_zeros(2, 0, pad_size)?;
            value = value.pad_with_zeros(2, 0, pad_size)?;
            beta = beta.pad_with_zeros(2, 0, pad_size)?;
            g = g.pad_with_zeros(2, 0, pad_size)?;
        }
        let total_sequence_length = sequence_length + pad_size;
        let num_chunks = total_sequence_length / chunk_size;
        query = (query * (1f64 / f64::sqrt(k_head_dim as f64)))?;

        let prepare_start = profile_start(device)?;
        let batch_heads = batch_size * num_heads;
        let k_beta = key.broadcast_mul(&beta.unsqueeze(D::Minus1)?)?;
        let v_beta = value.broadcast_mul(&beta.unsqueeze(D::Minus1)?)?;
        let query = query.reshape((batch_heads, num_chunks, chunk_size, k_head_dim))?;
        let key = key.reshape((batch_heads, num_chunks, chunk_size, k_head_dim))?;
        let k_beta = k_beta.reshape((batch_heads, num_chunks, chunk_size, k_head_dim))?;
        let v_beta = v_beta.reshape((batch_heads, num_chunks, chunk_size, v_head_dim))?;
        let g = g
            .reshape((batch_heads, num_chunks, chunk_size))?
            .cumsum(D::Minus1)?;
        let cache = self.chunk_cache(query.device(), compute_dtype, chunk_size)?;
        let lower_2d = cache.lower_2d.reshape((1, chunk_size, chunk_size))?;
        let eye_2d = Tensor::eye(chunk_size, compute_dtype, query.device())?
            .reshape((1, chunk_size, chunk_size))?;
        let strict_lower_2d = lower_2d.broadcast_sub(&eye_2d)?;
        let decay_mask = g
            .unsqueeze(3)?
            .broadcast_sub(&g.unsqueeze(2)?)?
            .exp()?
            .broadcast_mul(&lower_2d)?;
        let exp_g = g.exp()?;
        profile.linear_chunk_prepare_millis += profile_elapsed(prepare_start, device)?;

        let solve_start = profile_start(device)?;
        let solve_batch = batch_heads * num_chunks;
        let base_attn = k_beta
            .matmul(&key.transpose(3, 2)?)?
            .broadcast_mul(&decay_mask)?
            .neg()?
            .broadcast_mul(&strict_lower_2d)?
            .reshape((solve_batch, chunk_size, chunk_size))?;
        let mut rows = Vec::with_capacity(chunk_size);
        rows.push(Tensor::zeros(
            (solve_batch, 1, chunk_size),
            compute_dtype,
            query.device(),
        )?);
        for i in 1..chunk_size {
            let row = base_attn
                .narrow(1, i, 1)?
                .narrow(2, 0, i)?
                .reshape((solve_batch, i))?;
            let sub = Tensor::cat(&rows[..i].iter().collect::<Vec<_>>(), 1)?.narrow(2, 0, i)?;
            let correction = row
                .unsqueeze(1)?
                .broadcast_mul(&sub)?
                .sum(1)?
                .reshape((solve_batch, i))?;
            let row = row.broadcast_add(&correction)?;
            let row =
                row.pad_with_zeros(1, 0, chunk_size - i)?
                    .reshape((solve_batch, 1, chunk_size))?;
            rows.push(row);
        }
        let attn = Tensor::cat(&rows.iter().collect::<Vec<_>>(), 1)?
            .broadcast_add(&eye_2d)?
            .reshape((batch_heads, num_chunks, chunk_size, chunk_size))?;
        let solved_value = attn.matmul(&v_beta)?;
        let k_cumdecay = attn.matmul(&k_beta.broadcast_mul(&exp_g.unsqueeze(D::Minus1)?)?)?;
        profile.linear_chunk_solve_millis += profile_elapsed(solve_start, device)?;

        let scan_start = profile_start(device)?;
        let mut last_recurrent_state = Tensor::zeros(
            (batch_heads, k_head_dim, v_head_dim),
            compute_dtype,
            query.device(),
        )?;
        let mut outputs = Vec::with_capacity(num_chunks);
        for chunk_idx in 0..num_chunks {
            let q_i = query.i((.., chunk_idx, .., ..))?;
            let k_i = key.i((.., chunk_idx, .., ..))?;
            let v_i = solved_value.i((.., chunk_idx, .., ..))?;
            let g_i = g.i((.., chunk_idx, ..))?;
            let decay_i = decay_mask.i((.., chunk_idx, .., ..))?;

            let recurrent_read_start = profile_start(device)?;
            let v_prime = k_cumdecay
                .i((.., chunk_idx, .., ..))?
                .matmul(&last_recurrent_state)?;
            let attn_inter = q_i
                .broadcast_mul(&g_i.exp()?.unsqueeze(D::Minus1)?)?
                .matmul(&last_recurrent_state)?;
            profile.linear_chunk_recurrent_read_millis +=
                profile_elapsed(recurrent_read_start, device)?;

            let v_new = v_i.broadcast_sub(&v_prime)?;

            let local_attn_start = profile_start(device)?;
            let local_attn = q_i
                .matmul(&k_i.transpose(2, 1)?)?
                .broadcast_mul(&decay_i)?
                .broadcast_mul(&lower_2d)?;
            let local_out = local_attn.matmul(&v_new)?;
            outputs.push(attn_inter.broadcast_add(&local_out)?.unsqueeze(1)?);
            profile.linear_chunk_local_attn_millis += profile_elapsed(local_attn_start, device)?;

            let state_update_start = profile_start(device)?;
            let g_last = g_i.i((.., chunk_size - 1))?;
            let state_decay = g_last.exp()?.unsqueeze(D::Minus1)?.unsqueeze(D::Minus1)?;
            let chunk_decay = g_last
                .unsqueeze(D::Minus1)?
                .broadcast_sub(&g_i)?
                .exp()?
                .unsqueeze(D::Minus1)?;
            last_recurrent_state = last_recurrent_state
                .broadcast_mul(&state_decay)?
                .broadcast_add(
                    &k_i.broadcast_mul(&chunk_decay)?
                        .transpose(2, 1)?
                        .matmul(&v_new)?,
                )?;
            profile.linear_chunk_state_update_millis +=
                profile_elapsed(state_update_start, device)?;
        }
        profile.linear_chunk_scan_millis += profile_elapsed(scan_start, device)?;

        let output = Tensor::cat(&outputs.iter().collect::<Vec<_>>(), 1)?
            .reshape((batch_size, num_heads, total_sequence_length, v_head_dim))?
            .narrow(2, 0, sequence_length)?
            .transpose(1, 2)?
            .contiguous()?
            .to_dtype(initial_dtype)?;
        profile.linear_attention_millis += profile_elapsed(total_start, device)?;
        Ok((output, last_recurrent_state, profile))
    }

    fn apply_mask_to_padding_states(
        &self,
        hidden_states: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        match attention_mask {
            Some(mask) if mask.dim(1)? > 1 && mask.dim(0)? > 1 => hidden_states
                .broadcast_mul(&mask.unsqueeze(D::Minus1)?.to_dtype(hidden_states.dtype())?),
            None => Ok(hidden_states.clone()),
            Some(_) => Ok(hidden_states.clone()),
        }
    }

    fn prepare_depthwise_conv_input(&mut self, mixed_qkv: &Tensor) -> Result<Tensor> {
        let kernel = self.conv_kernel_size;
        let mixed_qkv = match &self.conv_state {
            Some(conv_state) => {
                let conv_state = if conv_state.dtype() == mixed_qkv.dtype() {
                    conv_state.clone()
                } else {
                    conv_state.to_dtype(mixed_qkv.dtype())?
                };
                Tensor::cat(&[&conv_state, mixed_qkv], 2)?
            }
            None => mixed_qkv.pad_with_zeros(2, kernel.saturating_sub(1), 0)?,
        };
        let total_len = mixed_qkv.dim(2)?;
        let state_len = kernel.saturating_sub(1);
        self.conv_state = if state_len == 0 {
            None
        } else {
            Some(
                mixed_qkv
                    .narrow(2, total_len - state_len, state_len)?
                    .contiguous()?,
            )
        };
        Ok(mixed_qkv)
    }

    fn depthwise_conv_from_state(&mut self, mixed_qkv: &Tensor) -> Result<Tensor> {
        let kernel = self.conv_kernel_size;
        let seq_len = mixed_qkv.dim(2)?;
        let mixed_qkv = self.prepare_depthwise_conv_input(mixed_qkv)?;
        let weights = self.conv1d.weight().squeeze(1)?;
        let mut output: Option<Tensor> = None;
        for tap in 0..kernel {
            let xs = mixed_qkv.narrow(2, tap, seq_len)?;
            let w = weights.i((.., tap))?.reshape((1, self.conv_dim(), 1))?;
            let contrib = xs.broadcast_mul(&w)?;
            output = Some(match output {
                Some(acc) => acc.broadcast_add(&contrib)?,
                None => contrib,
            });
        }
        output
            .expect("depthwise conv produced at least one tap")
            .silu()
    }

    fn run_depthwise_conv(&mut self, mixed_qkv: &Tensor) -> Result<Tensor> {
        self.depthwise_conv_from_state(mixed_qkv)
    }

    fn run_depthwise_conv_update(&mut self, mixed_qkv: &Tensor) -> Result<Tensor> {
        self.depthwise_conv_from_state(mixed_qkv)
    }

    fn run_depthwise_conv_packed_prefill(&mut self, mixed_qkv: &Tensor) -> Result<Tensor> {
        let seq_len = mixed_qkv.dim(2)?;
        let mixed_qkv = self.prepare_depthwise_conv_input(mixed_qkv)?.contiguous()?;
        let weights = self.conv1d.weight().squeeze(1)?.contiguous()?;
        linear_prefill_conv_pack(&mixed_qkv, &weights, seq_len, self.conv_kernel_size)
    }

    fn conv_dim(&self) -> usize {
        self.key_dim * 2 + self.value_dim
    }

    fn recurrent_gated_delta_rule(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        g: &Tensor,
        beta: &Tensor,
        initial_state: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor, RuntimeProfile)> {
        let device = query.device();
        let total_start = profile_start(device)?;
        let mut profile = RuntimeProfile::default();
        let initial_dtype = query.dtype();
        let compute_dtype = initial_dtype;
        let query = query
            .transpose(1, 2)?
            .contiguous()?
            .to_dtype(compute_dtype)?;
        let key = key.transpose(1, 2)?.contiguous()?.to_dtype(compute_dtype)?;
        let value = value
            .transpose(1, 2)?
            .contiguous()?
            .to_dtype(compute_dtype)?;
        let beta = beta
            .transpose(1, 2)?
            .contiguous()?
            .to_dtype(compute_dtype)?;
        let g = g.transpose(1, 2)?.contiguous()?.to_dtype(compute_dtype)?;

        let (batch_size, num_heads, seq_len, k_head_dim) = key.dims4()?;
        let v_head_dim = value.dim(D::Minus1)?;
        let query = (query * (1f64 / f64::sqrt(k_head_dim as f64)))?;

        let mut recurrent_state = match initial_state {
            Some(state) => state.to_dtype(compute_dtype)?,
            None => Tensor::zeros(
                (batch_size, num_heads, k_head_dim, v_head_dim),
                compute_dtype,
                query.device(),
            )?,
        };

        let mut outputs = Vec::with_capacity(seq_len);
        let loop_start = profile_start(device)?;
        for step in 0..seq_len {
            let q_t = query.i((.., .., step, ..))?.contiguous()?;
            let k_t = key.i((.., .., step, ..))?.contiguous()?;
            let v_t = value.i((.., .., step, ..))?.contiguous()?;
            let beta_t = beta.i((.., .., step))?.unsqueeze(D::Minus1)?;
            let g_t = g
                .i((.., .., step))?
                .exp()?
                .unsqueeze(D::Minus1)?
                .unsqueeze(D::Minus1)?;

            recurrent_state = recurrent_state.broadcast_mul(&g_t)?;
            let kv_mem = recurrent_state
                .broadcast_mul(&k_t.unsqueeze(D::Minus1)?)?
                .sum_keepdim(2)?
                .squeeze(2)?;
            let delta = (v_t.broadcast_sub(&kv_mem)?).broadcast_mul(&beta_t)?;
            recurrent_state = recurrent_state.broadcast_add(
                &k_t.unsqueeze(D::Minus1)?
                    .broadcast_mul(&delta.unsqueeze(2)?)?,
            )?;
            let out_t = recurrent_state
                .broadcast_mul(&q_t.unsqueeze(D::Minus1)?)?
                .sum_keepdim(2)?
                .squeeze(2)?;
            outputs.push(out_t.unsqueeze(2)?);
        }
        profile.linear_recurrent_loop_millis += profile_elapsed(loop_start, device)?;

        let output = Tensor::cat(&outputs.iter().collect::<Vec<_>>(), 2)?
            .transpose(1, 2)?
            .contiguous()?
            .to_dtype(initial_dtype)?;
        profile.linear_attention_millis += profile_elapsed(total_start, device)?;
        Ok((output, recurrent_state, profile))
    }

    fn chunk_gated_delta_rule(
        &mut self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        g: &Tensor,
        beta: &Tensor,
        sequence_length: usize,
    ) -> Result<(Tensor, Tensor, RuntimeProfile)> {
        let device = query.device();
        let total_start = profile_start(device)?;
        let mut profile = RuntimeProfile::default();
        let initial_dtype = query.dtype();
        let chunk_size = linear_attention_chunk_size(query.device(), sequence_length);
        let estimated_pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size;
        let estimated_num_chunks = (sequence_length + estimated_pad_size) / chunk_size;
        let scan_policy =
            delta_net_execution_policy(query.device(), sequence_length, estimated_num_chunks);
        let scan_mode = scan_policy.scan_mode;
        if scan_mode == DeltaNetScanMode::TorchLike {
            return self.chunk_gated_delta_rule_torch_like(
                query,
                key,
                value,
                g,
                beta,
                sequence_length,
            );
        }
        let compute_dtype = delta_net_compute_dtype(scan_mode, initial_dtype);
        let query_heads = query.dim(2)?;
        let key_heads = key.dim(2)?;
        let value_heads = value.dim(2)?;
        if query_heads != key_heads {
            candle::bail!(
                "chunk_gated_delta_rule expected matching query/key head counts, got query_heads={query_heads} key_heads={key_heads}"
            );
        }
        if value_heads % query_heads != 0 {
            candle::bail!(
                "chunk_gated_delta_rule expected value heads to be a multiple of query heads, got query_heads={query_heads} value_heads={value_heads}"
            );
        }
        let head_repeat = value_heads / query_heads;
        let query = if head_repeat > 1 {
            repeat_heads(query, head_repeat)?
        } else {
            query.clone()
        };
        let key = if head_repeat > 1 {
            repeat_heads(key, head_repeat)?
        } else {
            key.clone()
        };

        let mut query = query
            .transpose(1, 2)?
            .contiguous()?
            .to_dtype(compute_dtype)?;
        let mut key = key.transpose(1, 2)?.contiguous()?.to_dtype(compute_dtype)?;
        let mut value = value
            .transpose(1, 2)?
            .contiguous()?
            .to_dtype(compute_dtype)?;
        let mut beta = beta
            .transpose(1, 2)?
            .contiguous()?
            .to_dtype(compute_dtype)?;
        let mut g = g.transpose(1, 2)?.contiguous()?.to_dtype(compute_dtype)?;

        let (batch_size, num_heads, sequence_length, k_head_dim) = query.dims4()?;
        let v_head_dim = value.dim(D::Minus1)?;
        let pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size;

        if pad_size > 0 {
            query = query.pad_with_zeros(2, 0, pad_size)?;
            key = key.pad_with_zeros(2, 0, pad_size)?;
            value = value.pad_with_zeros(2, 0, pad_size)?;
            beta = beta.pad_with_zeros(2, 0, pad_size)?;
            g = g.pad_with_zeros(2, 0, pad_size)?;
        }

        let total_sequence_length = sequence_length + pad_size;
        let num_chunks = total_sequence_length / chunk_size;
        query = (query * (1f64 / f64::sqrt(k_head_dim as f64)))?;

        if use_delta_recurrent_prefill_kernel(query.device(), sequence_length) {
            let batch_heads = batch_size * num_heads;
            let pack_start = profile_start(device)?;
            let query_scan = query
                .reshape((batch_heads, total_sequence_length, k_head_dim))?
                .contiguous()?;
            let key_scan = key
                .reshape((batch_heads, total_sequence_length, k_head_dim))?
                .contiguous()?;
            let value_scan = value
                .reshape((batch_heads, total_sequence_length, v_head_dim))?
                .contiguous()?;
            let beta_scan = beta
                .reshape((batch_heads, total_sequence_length))?
                .contiguous()?;
            let g_scan = g
                .reshape((batch_heads, total_sequence_length))?
                .contiguous()?;
            let initial_state = Tensor::zeros(
                (batch_heads, k_head_dim, v_head_dim),
                compute_dtype,
                query.device(),
            )?;
            let pack_elapsed = profile_elapsed(pack_start, device)?;
            profile.linear_full_kernel_pack_millis += pack_elapsed;
            profile.transfer_millis += pack_elapsed;

            let kernel_start = profile_start(device)?;
            let fused = delta_recurrent_prefill(
                &initial_state,
                &query_scan,
                &key_scan,
                &value_scan,
                &beta_scan,
                &g_scan,
            )?;
            profile.linear_full_kernel_execute_millis += profile_elapsed(kernel_start, device)?;

            let unpack_start = profile_start(device)?;
            let output_scan = fused.narrow(1, 0, total_sequence_length)?.reshape((
                batch_size,
                num_heads,
                total_sequence_length,
                v_head_dim,
            ))?;
            let recurrent_state = fused
                .narrow(1, total_sequence_length, k_head_dim)?
                .reshape((batch_heads, k_head_dim, v_head_dim))?
                .contiguous()?;
            let output = output_scan
                .narrow(2, 0, sequence_length)?
                .transpose(1, 2)?
                .contiguous()?
                .to_dtype(initial_dtype)?;
            let unpack_elapsed = profile_elapsed(unpack_start, device)?;
            profile.linear_full_kernel_unpack_millis += unpack_elapsed;
            profile.transfer_millis += unpack_elapsed;
            profile.linear_chunk_scan_millis += profile.linear_full_kernel_pack_millis
                + profile.linear_full_kernel_execute_millis
                + profile.linear_full_kernel_unpack_millis;
            profile.linear_attention_millis += profile_elapsed(total_start, device)?;
            return Ok((output, recurrent_state, profile));
        }

        let query = query.reshape((batch_size, num_heads, num_chunks, chunk_size, k_head_dim))?;
        let key = key.reshape((batch_size, num_heads, num_chunks, chunk_size, k_head_dim))?;
        let value = value.reshape((batch_size, num_heads, num_chunks, chunk_size, v_head_dim))?;
        let beta = beta.reshape((batch_size, num_heads, num_chunks, chunk_size))?;
        let g_raw = g.reshape((batch_size, num_heads, num_chunks, chunk_size))?;
        let batch_heads = batch_size * num_heads;
        let query_scan = query.reshape((batch_heads, num_chunks, chunk_size, k_head_dim))?;
        let key_scan = key.reshape((batch_heads, num_chunks, chunk_size, k_head_dim))?;
        let value_scan = value.reshape((batch_heads, num_chunks, chunk_size, v_head_dim))?;
        let beta_scan = beta.reshape((batch_heads, num_chunks, chunk_size))?;
        let g_scan = g_raw.reshape((batch_heads, num_chunks, chunk_size))?;

        if use_delta_chunk_scan_kernel(query.device(), scan_mode, sequence_length, chunk_size) {
            let pack_start = profile_start(device)?;
            let query_scan = query_scan.contiguous()?;
            let key_scan = key_scan.contiguous()?;
            let value_scan = value_scan.contiguous()?;
            let beta_scan = beta_scan.contiguous()?;
            let g_scan = g_scan.contiguous()?;
            let initial_state = Tensor::zeros(
                (batch_heads, k_head_dim, v_head_dim),
                compute_dtype,
                query.device(),
            )?;
            let pack_elapsed = profile_elapsed(pack_start, device)?;
            profile.linear_full_kernel_pack_millis += pack_elapsed;
            profile.transfer_millis += pack_elapsed;

            let kernel_start = profile_start(device)?;
            let fused = delta_chunk_scan_raw(
                &initial_state,
                &query_scan,
                &key_scan,
                &value_scan,
                &beta_scan,
                &g_scan,
            )?;
            profile.linear_full_kernel_execute_millis += profile_elapsed(kernel_start, device)?;

            let unpack_start = profile_start(device)?;
            let output_scan = fused.narrow(1, 0, total_sequence_length)?.reshape((
                batch_size,
                num_heads,
                total_sequence_length,
                v_head_dim,
            ))?;
            let last_recurrent_state = fused
                .narrow(1, total_sequence_length, k_head_dim)?
                .reshape((batch_heads, k_head_dim, v_head_dim))?
                .contiguous()?;
            let output = output_scan
                .narrow(2, 0, sequence_length)?
                .transpose(1, 2)?
                .contiguous()?
                .to_dtype(initial_dtype)?;
            let unpack_elapsed = profile_elapsed(unpack_start, device)?;
            profile.linear_full_kernel_unpack_millis += unpack_elapsed;
            profile.transfer_millis += unpack_elapsed;
            profile.linear_chunk_scan_millis += profile.linear_full_kernel_pack_millis
                + profile.linear_full_kernel_execute_millis
                + profile.linear_full_kernel_unpack_millis;
            profile.linear_attention_millis += profile_elapsed(total_start, device)?;
            return Ok((output, last_recurrent_state, profile));
        }

        if use_delta_chunk_step_kernel(query.device(), scan_mode, sequence_length, chunk_size) {
            if use_delta_chunk_windowed_kernel(
                query.device(),
                scan_mode,
                sequence_length,
                chunk_size,
            ) {
                let pack_start = profile_start(device)?;
                let initial_state = Tensor::zeros(
                    (batch_heads, k_head_dim, v_head_dim),
                    compute_dtype,
                    query.device(),
                )?;
                let pack_elapsed = profile_elapsed(pack_start, device)?;
                profile.linear_full_kernel_pack_millis += pack_elapsed;
                profile.transfer_millis += pack_elapsed;

                let scan_start = profile_start(device)?;
                let fused = delta_chunk_step_windowed_raw(
                    &initial_state,
                    &query_scan,
                    &key_scan,
                    &value_scan,
                    &beta_scan,
                    &g_scan,
                )?;
                profile.linear_full_kernel_execute_millis += profile_elapsed(scan_start, device)?;

                let unpack_start = profile_start(device)?;
                let output = fused
                    .narrow(1, 0, total_sequence_length)?
                    .reshape((batch_size, num_heads, total_sequence_length, v_head_dim))?
                    .narrow(2, 0, sequence_length)?
                    .transpose(1, 2)?
                    .contiguous()?
                    .to_dtype(initial_dtype)?;
                let last_recurrent_state = fused
                    .narrow(1, total_sequence_length, k_head_dim)?
                    .reshape((batch_heads, k_head_dim, v_head_dim))?
                    .contiguous()?;
                let unpack_elapsed = profile_elapsed(unpack_start, device)?;
                profile.linear_full_kernel_unpack_millis += unpack_elapsed;
                profile.transfer_millis += unpack_elapsed;
                profile.linear_chunk_scan_millis += profile.linear_full_kernel_pack_millis
                    + profile.linear_full_kernel_execute_millis
                    + profile.linear_full_kernel_unpack_millis;
                profile.linear_attention_millis += profile_elapsed(total_start, device)?;
                return Ok((output, last_recurrent_state, profile));
            }

            let mut last_recurrent_state = Tensor::zeros(
                (batch_heads, k_head_dim, v_head_dim),
                compute_dtype,
                query.device(),
            )?;
            let mut outputs = Vec::with_capacity(num_chunks);
            let scan_start = profile_start(device)?;
            for chunk_idx in 0..num_chunks {
                let pack_start = profile_start(device)?;
                let q_i = query_scan.i((.., chunk_idx, .., ..))?.contiguous()?;
                let k_i = key_scan.i((.., chunk_idx, .., ..))?.contiguous()?;
                let v_i = value_scan.i((.., chunk_idx, .., ..))?.contiguous()?;
                let beta_i = beta_scan.i((.., chunk_idx, ..))?.contiguous()?;
                let g_i = g_scan.i((.., chunk_idx, ..))?.contiguous()?;
                let prev_state_i = last_recurrent_state.contiguous()?;
                let pack_elapsed = profile_elapsed(pack_start, device)?;
                profile.linear_full_kernel_pack_millis += pack_elapsed;
                profile.transfer_millis += pack_elapsed;

                let kernel_start = profile_start(device)?;
                let fused = delta_chunk_step_raw(&prev_state_i, &q_i, &k_i, &v_i, &beta_i, &g_i)?;
                profile.linear_full_kernel_execute_millis += profile_elapsed(kernel_start, device)?;

                let unpack_start = profile_start(device)?;
                outputs.push(fused.narrow(1, 0, chunk_size)?.unsqueeze(1)?);
                last_recurrent_state = fused
                    .narrow(1, chunk_size, k_head_dim)?
                    .reshape((batch_heads, k_head_dim, v_head_dim))?
                    .contiguous()?;
                let unpack_elapsed = profile_elapsed(unpack_start, device)?;
                profile.linear_full_kernel_unpack_millis += unpack_elapsed;
                profile.transfer_millis += unpack_elapsed;
            }
            profile.linear_chunk_scan_millis += profile_elapsed(scan_start, device)?;
            let output = Tensor::cat(&outputs.iter().collect::<Vec<_>>(), 1)?
                .reshape((batch_size, num_heads, total_sequence_length, v_head_dim))?
                .narrow(2, 0, sequence_length)?
                .transpose(1, 2)?
                .contiguous()?
                .to_dtype(initial_dtype)?;
            profile.linear_attention_millis += profile_elapsed(total_start, device)?;
            return Ok((output, last_recurrent_state, profile));
        }

        let prepare_start = profile_start(device)?;
        let k_beta = key.broadcast_mul(&beta.unsqueeze(D::Minus1)?)?;
        let g = g_raw.cumsum(D::Minus1)?;
        let exp_g = g.exp()?;

        let cache = self.chunk_cache(query.device(), compute_dtype, chunk_size)?;
        let lower = cache.lower;
        let eye = cache.eye;
        let strict_lower = cache.strict_lower;

        let decay_mask = g
            .unsqueeze(4)?
            .broadcast_sub(&g.unsqueeze(3)?)?
            .exp()?
            .broadcast_mul(&lower)?;
        let base_attn = k_beta
            .matmul(&key.transpose(4, 3)?)?
            .broadcast_mul(&decay_mask)?
            .neg()?
            .broadcast_mul(&strict_lower)?;
        profile.linear_chunk_prepare_millis += profile_elapsed(prepare_start, device)?;

        let solve_start = profile_start(device)?;
        let attn = if scan_policy.use_flattened_solve {
            let solve_batch = batch_size * num_heads * num_chunks;
            let base_attn_flat = base_attn.reshape((solve_batch, chunk_size, chunk_size))?;
            let mut rows = Vec::with_capacity(chunk_size);
            rows.push(Tensor::zeros(
                (solve_batch, 1, chunk_size),
                compute_dtype,
                query.device(),
            )?);

            for i in 1..chunk_size {
                let row = base_attn_flat
                    .narrow(1, i, 1)?
                    .narrow(2, 0, i)?
                    .reshape((solve_batch, i))?;
                let sub = Tensor::cat(&rows[..i].iter().collect::<Vec<_>>(), 1)?.narrow(2, 0, i)?;
                let correction = row
                    .unsqueeze(1)?
                    .broadcast_mul(&sub)?
                    .sum(1)?
                    .reshape((solve_batch, i))?;
                let row = row.broadcast_add(&correction)?;
                let row = row.pad_with_zeros(1, 0, chunk_size - i)?.reshape((
                    solve_batch,
                    1,
                    chunk_size,
                ))?;
                rows.push(row);
            }

            Tensor::cat(&rows.iter().collect::<Vec<_>>(), 1)?
                .reshape((batch_size, num_heads, num_chunks, chunk_size, chunk_size))?
                .broadcast_add(&eye)?
        } else {
            let mut rows = Vec::with_capacity(chunk_size);
            rows.push(Tensor::zeros(
                (batch_size, num_heads, num_chunks, 1, chunk_size),
                compute_dtype,
                query.device(),
            )?);

            for i in 1..chunk_size {
                let row = base_attn.narrow(3, i, 1)?.narrow(4, 0, i)?.squeeze(3)?;
                let sub = Tensor::cat(&rows[..i].iter().collect::<Vec<_>>(), 3)?.narrow(4, 0, i)?;
                let correction = row.unsqueeze(4)?.broadcast_mul(&sub)?.sum(3)?;
                let row = (row + correction)?;
                let row = row.pad_with_zeros(3, 0, chunk_size - i)?.unsqueeze(3)?;
                rows.push(row);
            }

            Tensor::cat(&rows.iter().collect::<Vec<_>>(), 3)?.broadcast_add(&eye)?
        };
        let k_cumdecay = attn.matmul(&k_beta.broadcast_mul(&exp_g.unsqueeze(D::Minus1)?)?)?;
        profile.linear_chunk_solve_millis += profile_elapsed(solve_start, device)?;

        let lower_2d = cache.lower_2d;
        let decay_scan = decay_mask.reshape((batch_heads, num_chunks, chunk_size, chunk_size))?;
        let exp_g_scan = exp_g.reshape((batch_heads, num_chunks, chunk_size))?;
        let k_cumdecay_scan =
            k_cumdecay.reshape((batch_heads, num_chunks, chunk_size, k_head_dim))?;
        let q_state_scan = query_scan.broadcast_mul(&exp_g_scan.unsqueeze(D::Minus1)?)?;
        let (state_decay_scan, chunk_decay_scan) = match scan_mode {
            DeltaNetScanMode::HoistedDecays | DeltaNetScanMode::PrebatchedLocal => {
                let exp_g_last_scan = exp_g_scan.i((.., .., chunk_size - 1))?;
                (
                    Some(exp_g_last_scan.unsqueeze(D::Minus1)?.unsqueeze(D::Minus1)?),
                    Some(
                        exp_g_last_scan
                            .unsqueeze(D::Minus1)?
                            .broadcast_div(&exp_g_scan)?
                            .unsqueeze(D::Minus1)?,
                    ),
                )
            }
            DeltaNetScanMode::Flat3d | DeltaNetScanMode::TorchLike => (None, None),
        };
        let local_attn_scan = match scan_mode {
            DeltaNetScanMode::PrebatchedLocal => Some(
                query_scan
                    .matmul(&key_scan.transpose(3, 2)?)?
                    .broadcast_mul(&decay_scan)?
                    .broadcast_mul(&lower_2d.reshape((1, 1, chunk_size, chunk_size))?)?,
            ),
            _ => None,
        };
        let lower_2d = lower_2d.reshape((1, chunk_size, chunk_size))?;
        let mut last_recurrent_state = Tensor::zeros(
            (batch_heads, k_head_dim, v_head_dim),
            compute_dtype,
            query.device(),
        )?;
        let mut outputs = Vec::with_capacity(num_chunks);
        let use_state_kernel = use_delta_state_kernel(query.device(), scan_mode, sequence_length);
        let use_state_scan_kernel =
            use_delta_state_scan_kernel(query.device(), scan_mode, sequence_length);
        let use_chunk_fused_kernel =
            use_delta_chunk_fused_kernel(query.device(), scan_mode, sequence_length);
        let use_full_scan_kernel =
            use_delta_full_scan_kernel(query.device(), scan_mode, sequence_length);
        let full_scan = if use_full_scan_kernel {
            let full_pack_start = profile_start(device)?;
            let state_decay_scan = state_decay_scan
                .as_ref()
                .ok_or_else(|| {
                    candle::Error::Msg("delta-full-scan requires hoisted state decay".into())
                })?
                .squeeze(3)?
                .squeeze(2)?
                .contiguous()?;
            let weighted_key_scan = key_scan
                .broadcast_mul(chunk_decay_scan.as_ref().ok_or_else(|| {
                    candle::Error::Msg("delta-full-scan requires hoisted chunk decay".into())
                })?)?
                .contiguous()?;
            let k_cumdecay_scan = k_cumdecay_scan.contiguous()?;
            let q_state_scan = q_state_scan.contiguous()?;
            let local_attn_scan = local_attn_scan
                .as_ref()
                .ok_or_else(|| {
                    candle::Error::Msg("delta-full-scan requires prebatched local attention".into())
                })?
                .contiguous()?;
            let value_scan = value_scan.contiguous()?;
            let full_pack_elapsed = profile_elapsed(full_pack_start, device)?;
            profile.linear_full_kernel_pack_millis += full_pack_elapsed;
            profile.transfer_millis += full_pack_elapsed;
            let full_kernel_start = profile_start(device)?;
            let full_scan = delta_full_scan(
                &last_recurrent_state,
                &weighted_key_scan,
                &k_cumdecay_scan,
                &q_state_scan,
                &local_attn_scan,
                &state_decay_scan,
                &value_scan,
            )?;
            profile.linear_full_kernel_execute_millis +=
                profile_elapsed(full_kernel_start, device)?;
            Some(full_scan)
        } else {
            None
        };
        let state_scan = if use_state_scan_kernel {
            let state_decay_scan = state_decay_scan.as_ref().ok_or_else(|| {
                candle::Error::Msg("delta-state-scan requires hoisted state decay".into())
            })?;
            let chunk_decay_scan = chunk_decay_scan.as_ref().ok_or_else(|| {
                candle::Error::Msg("delta-state-scan requires hoisted chunk decay".into())
            })?;
            let weighted_key_scan = key_scan.broadcast_mul(chunk_decay_scan)?;
            let state_decay_feature =
                state_decay_scan.broadcast_as((batch_heads, num_chunks, chunk_size, 1))?;
            let packed_scan = Tensor::cat(
                &[&weighted_key_scan, &k_cumdecay_scan, &state_decay_feature],
                3,
            )?
            .contiguous()?;
            Some(delta_state_scan(
                &last_recurrent_state,
                &packed_scan,
                &value_scan.contiguous()?,
            )?)
        } else {
            None
        };
        if let Some(full_scan) = &full_scan {
            let full_unpack_start = profile_start(device)?;
            let output_scan = full_scan.narrow(1, 0, total_sequence_length)?.reshape((
                batch_size,
                num_heads,
                total_sequence_length,
                v_head_dim,
            ))?;
            let last_recurrent_state = full_scan
                .narrow(1, total_sequence_length, k_head_dim)?
                .reshape((batch_heads, k_head_dim, v_head_dim))?
                .contiguous()?;
            let output = output_scan
                .narrow(2, 0, sequence_length)?
                .transpose(1, 2)?
                .contiguous()?
                .to_dtype(initial_dtype)?;
            let full_unpack_elapsed = profile_elapsed(full_unpack_start, device)?;
            profile.linear_full_kernel_unpack_millis += full_unpack_elapsed;
            profile.transfer_millis += full_unpack_elapsed;
            profile.linear_chunk_scan_millis += profile.linear_full_kernel_pack_millis
                + profile.linear_full_kernel_execute_millis
                + profile.linear_full_kernel_unpack_millis;
            profile.linear_attention_millis += profile_elapsed(total_start, device)?;
            return Ok((output, last_recurrent_state, profile));
        }

        let scan_start = profile_start(device)?;
        for chunk_idx in 0..num_chunks {
            let index_start = profile_start(device)?;
            let q_i = query_scan.i((.., chunk_idx, .., ..))?;
            let k_i = key_scan.i((.., chunk_idx, .., ..))?;
            let v_i = value_scan.i((.., chunk_idx, .., ..))?;
            let g_i = g_scan.i((.., chunk_idx, ..))?;
            let prev_state_i = if let Some(state_scan) = &state_scan {
                state_scan.i((.., chunk_idx, .., ..))?
            } else {
                last_recurrent_state.clone()
            };
            profile.linear_chunk_index_millis += profile_elapsed(index_start, device)?;

            let local_attn_start = profile_start(device)?;
            let attn = if let Some(local_attn_scan) = &local_attn_scan {
                local_attn_scan.i((.., chunk_idx, .., ..))?
            } else {
                let decay_i = decay_scan.i((.., chunk_idx, .., ..))?;
                q_i.matmul(&k_i.transpose(2, 1)?)?
                    .broadcast_mul(&decay_i)?
                    .broadcast_mul(&lower_2d)?
            };
            profile.linear_chunk_local_attn_millis += profile_elapsed(local_attn_start, device)?;

            let recurrent_read_start = profile_start(device)?;
            let (v_new, attn_inter, fused_next_state) = if use_chunk_fused_kernel
                && state_scan.is_none()
            {
                let weighted_key = chunk_decay_scan
                    .as_ref()
                    .ok_or_else(|| {
                        candle::Error::Msg("delta-chunk-fused requires hoisted chunk decay".into())
                    })?
                    .i((.., chunk_idx, .., ..))?
                    .broadcast_mul(&k_i)?;
                let q_state = q_state_scan.i((.., chunk_idx, .., ..))?;
                let state_decay = state_decay_scan
                    .as_ref()
                    .ok_or_else(|| {
                        candle::Error::Msg("delta-chunk-fused requires hoisted state decay".into())
                    })?
                    .i((.., chunk_idx, .., ..))?
                    .broadcast_as((batch_heads, chunk_size, 1))?;
                let packed_chunk = Tensor::cat(
                    &[
                        &weighted_key,
                        &k_cumdecay_scan.i((.., chunk_idx, .., ..))?,
                        &q_state,
                        &state_decay,
                    ],
                    2,
                )?
                .contiguous()?;
                let fused = delta_chunk_fused(
                    &prev_state_i.contiguous()?,
                    &packed_chunk,
                    &v_i.contiguous()?,
                )?;
                (
                    fused.narrow(1, 0, chunk_size)?,
                    fused.narrow(1, chunk_size, chunk_size)?,
                    Some(fused.narrow(1, 2 * chunk_size, k_head_dim)?),
                )
            } else {
                let v_prime = k_cumdecay_scan
                    .i((.., chunk_idx, .., ..))?
                    .matmul(&prev_state_i)?;
                let v_new = v_i.broadcast_sub(&v_prime)?;
                let attn_inter = q_state_scan
                    .i((.., chunk_idx, .., ..))?
                    .matmul(&prev_state_i)?;
                (v_new, attn_inter, None)
            };
            profile.linear_chunk_recurrent_read_millis +=
                profile_elapsed(recurrent_read_start, device)?;

            let local_mix_start = profile_start(device)?;
            outputs.push(
                attn_inter
                    .broadcast_add(&attn.matmul(&v_new)?)?
                    .unsqueeze(1)?,
            );
            profile.linear_chunk_local_attn_millis += profile_elapsed(local_mix_start, device)?;

            let state_update_start = profile_start(device)?;
            let (state_decay, chunk_decay) =
                if let (Some(state_decay_scan), Some(chunk_decay_scan)) =
                    (&state_decay_scan, &chunk_decay_scan)
                {
                    (
                        state_decay_scan.i((.., chunk_idx, .., ..))?,
                        chunk_decay_scan.i((.., chunk_idx, .., ..))?,
                    )
                } else {
                    let g_last = g_i.i((.., chunk_size - 1))?;
                    (
                        g_last.exp()?.unsqueeze(D::Minus1)?.unsqueeze(D::Minus1)?,
                        g_last
                            .unsqueeze(D::Minus1)?
                            .broadcast_sub(&g_i)?
                            .exp()?
                            .unsqueeze(D::Minus1)?,
                    )
                };
            if let Some(fused_next_state) = fused_next_state {
                last_recurrent_state = fused_next_state.contiguous()?;
            } else if let Some(state_scan) = &state_scan {
                last_recurrent_state = state_scan.i((.., chunk_idx + 1, .., ..))?.contiguous()?;
            } else {
                let prev_state_scaled = last_recurrent_state
                    .broadcast_mul(&state_decay)?
                    .contiguous()?;
                let weighted_key = k_i.broadcast_mul(&chunk_decay)?.contiguous()?;
                last_recurrent_state = delta_state_update(
                    &prev_state_scaled,
                    &weighted_key,
                    &v_new,
                    use_state_kernel,
                )?
                .contiguous()?;
            }
            profile.linear_chunk_state_update_millis +=
                profile_elapsed(state_update_start, device)?;
        }
        profile.linear_chunk_scan_millis += profile_elapsed(scan_start, device)?;

        let output = Tensor::cat(&outputs.iter().collect::<Vec<_>>(), 1)?
            .reshape((batch_size, num_heads, total_sequence_length, v_head_dim))?
            .narrow(2, 0, sequence_length)?
            .transpose(1, 2)?
            .contiguous()?
            .to_dtype(initial_dtype)?;
        profile.linear_attention_millis += profile_elapsed(total_start, device)?;
        Ok((output, last_recurrent_state, profile))
    }

    fn forward_profiled_with_state(
        &mut self,
        hidden_states: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor, RuntimeProfile)> {
        let device = hidden_states.device();
        let total_start = profile_start(device)?;
        let mut profile = RuntimeProfile::default();
        let compute_dtype =
            linear_attention_compute_dtype(hidden_states.device(), hidden_states.dtype());
        let layout_start = profile_start(device)?;
        let hidden_states = self.apply_mask_to_padding_states(hidden_states, attention_mask)?;
        let (batch_size, seq_len, _) = hidden_states.dims3()?;
        profile.layout_prepare_millis += profile_elapsed(layout_start, device)?;

        let qkv_start = profile_start(device)?;
        let mixed_qkv = self.in_proj_qkv.forward(&hidden_states)?.transpose(1, 2)?;
        let z = self.in_proj_z.forward(&hidden_states)?.reshape((
            batch_size,
            seq_len,
            self.num_v_heads,
            self.head_v_dim,
        ))?;
        let beta = ops::sigmoid(&self.in_proj_b.forward(&hidden_states)?)?;
        let a = self
            .in_proj_a
            .forward(&hidden_states)?
            .to_dtype(compute_dtype)?;
        profile.qkv_projection_millis += profile_elapsed(qkv_start, device)?;

        let kv_append_start = profile_start(device)?;
        let mixed_qkv = if seq_len == 1 {
            self.run_depthwise_conv_update(&mixed_qkv)?
                .transpose(1, 2)?
        } else if use_linear_prefill_packed_kernel(device, seq_len) {
            self.run_depthwise_conv_packed_prefill(&mixed_qkv)?
        } else {
            self.run_depthwise_conv(&mixed_qkv)?.transpose(1, 2)?
        };
        let (dt_bias, a_log_exp) = self.value_cache(device, compute_dtype)?;
        let g = softplus(&a.broadcast_add(&dt_bias)?)?
            .broadcast_mul(&a_log_exp)?
            .neg()?;
        let kv_append_elapsed = profile_elapsed(kv_append_start, device)?;
        profile.linear_conv_millis += kv_append_elapsed;
        profile.kv_append_write_millis += kv_append_elapsed;

        let layout_start = profile_start(device)?;
        let query = mixed_qkv.narrow(D::Minus1, 0, self.key_dim)?.reshape((
            batch_size,
            seq_len,
            self.num_k_heads,
            self.head_k_dim,
        ))?;
        let key = mixed_qkv
            .narrow(D::Minus1, self.key_dim, self.key_dim)?
            .reshape((batch_size, seq_len, self.num_k_heads, self.head_k_dim))?;
        let value = mixed_qkv
            .narrow(D::Minus1, self.key_dim * 2, self.value_dim)?
            .reshape((batch_size, seq_len, self.num_v_heads, self.head_v_dim))?;

        let query = if query.dtype() == compute_dtype {
            query
        } else {
            query.to_dtype(compute_dtype)?
        };
        let key = if key.dtype() == compute_dtype {
            key
        } else {
            key.to_dtype(compute_dtype)?
        };
        let query = l2norm(&query, 1e-6)?;
        let key = l2norm(&key, 1e-6)?;
        let head_repeat = self.num_v_heads / self.num_k_heads;
        let (query, key) = if seq_len == 1 && head_repeat > 1 {
            (
                repeat_heads(&query, head_repeat)?,
                repeat_heads(&key, head_repeat)?,
            )
        } else {
            (query, key)
        };
        let value = if value.dtype() == compute_dtype {
            value
        } else {
            value.to_dtype(compute_dtype)?
        };
        let beta = if beta.dtype() == compute_dtype {
            beta
        } else {
            beta.to_dtype(compute_dtype)?
        };
        let g = if g.dtype() == compute_dtype {
            g
        } else {
            g.to_dtype(compute_dtype)?
        };
        profile.layout_prepare_millis += profile_elapsed(layout_start, device)?;

        let (core_attn_out, recurrent_state, linear_profile) =
            if seq_len == 1 && self.recurrent_state.is_some() {
                self.recurrent_gated_delta_rule(
                    &query,
                    &key,
                    &value,
                    &g,
                    &beta,
                    self.recurrent_state.as_ref(),
                )?
            } else if seq_len == 1 {
                self.recurrent_gated_delta_rule(&query, &key, &value, &g, &beta, None)?
            } else {
                self.chunk_gated_delta_rule(&query, &key, &value, &g, &beta, seq_len)?
            };
        profile.add_assign(&linear_profile);

        let output_start = profile_start(device)?;
        let core_attn_out = self
            .norm
            .forward(
                &core_attn_out
                    .reshape((batch_size * seq_len * self.num_v_heads, self.head_v_dim))?,
                &z.reshape((batch_size * seq_len * self.num_v_heads, self.head_v_dim))?,
            )?
            .reshape((batch_size, seq_len, self.value_dim))?;
        let core_attn_out = if core_attn_out.dtype() == hidden_states.dtype() {
            core_attn_out
        } else {
            core_attn_out.to_dtype(hidden_states.dtype())?
        };
        let output = self.out_proj.forward(&core_attn_out)?;
        profile.output_projection_millis += profile_elapsed(output_start, device)?;
        profile.linear_attention_millis +=
            profile_elapsed(total_start, device)? - linear_profile.linear_attention_millis;
        Ok((output, recurrent_state, profile))
    }

    fn forward_profiled(
        &mut self,
        hidden_states: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<(Tensor, RuntimeProfile)> {
        let (output, recurrent_state, profile) =
            self.forward_profiled_with_state(hidden_states, attention_mask)?;
        self.recurrent_state = Some(recurrent_state);
        Ok((output, profile))
    }

    fn trace_profiled(
        &mut self,
        hidden_states: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor, RuntimeProfile)> {
        self.forward_profiled_with_state(hidden_states, attention_mask)
    }

    #[allow(dead_code)]
    fn forward(
        &mut self,
        hidden_states: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        self.forward_profiled(hidden_states, attention_mask)
            .map(|(output, _)| output)
    }

    fn clear_kv_cache(&mut self) {
        self.conv_state = None;
        self.recurrent_state = None;
    }
}

#[derive(Debug, Clone)]
enum LayerKind {
    Linear(GatedDeltaNet),
    Full(FullAttention),
}

#[derive(Debug, Clone)]
struct DecoderLayer {
    layer_type: String,
    token_mixer: LayerKind,
    mlp: Mlp,
    input_layernorm: Qwen35RmsNorm,
    post_attention_layernorm: Qwen35RmsNorm,
}

impl DecoderLayer {
    fn new(
        cfg: &TextConfig,
        layer_idx: usize,
        rotary_emb: Arc<RotaryEmbedding>,
        vb: VarBuilder,
    ) -> Result<Self> {
        let layer_type = cfg
            .layer_types
            .get(layer_idx)
            .cloned()
            .unwrap_or_else(|| "linear_attention".to_string());
        let token_mixer = match layer_type.as_str() {
            "linear_attention" => LayerKind::Linear(GatedDeltaNet::new(cfg, vb.pp("linear_attn"))?),
            "full_attention" => {
                LayerKind::Full(FullAttention::new(cfg, rotary_emb, vb.pp("self_attn"))?)
            }
            other => candle::bail!("unsupported qwen3.5 layer type {other:?}"),
        };
        Ok(Self {
            layer_type,
            token_mixer,
            mlp: Mlp::new(cfg, vb.pp("mlp"))?,
            input_layernorm: Qwen35RmsNorm::new(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                vb.pp("input_layernorm"),
            )?,
            post_attention_layernorm: Qwen35RmsNorm::new(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                vb.pp("post_attention_layernorm"),
            )?,
        })
    }

    fn forward_profiled(
        &mut self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
    ) -> Result<(Tensor, RuntimeProfile)> {
        let device = xs.device();
        let mut profile = RuntimeProfile::default();
        let residual = xs;
        let xs_norm = self.input_layernorm.forward(xs)?;
        let xs = match &mut self.token_mixer {
            LayerKind::Linear(linear_attn) => {
                let (xs, layer_profile) = linear_attn.forward_profiled(&xs_norm, attention_mask)?;
                profile.add_assign(&layer_profile);
                xs
            }
            LayerKind::Full(self_attn) => {
                let (xs, layer_profile) =
                    self_attn.forward_profiled(&xs_norm, attention_mask, seqlen_offset)?;
                profile.add_assign(&layer_profile);
                xs
            }
        };
        let xs = residual.broadcast_add(&xs)?;
        let residual = &xs;
        let xs = self.post_attention_layernorm.forward(&xs)?;
        let mlp_start = profile_start(device)?;
        let xs = self.mlp.forward(&xs)?;
        profile.mlp_millis += profile_elapsed(mlp_start, device)?;
        Ok((residual.broadcast_add(&xs)?, profile))
    }

    #[allow(dead_code)]
    fn forward(
        &mut self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        self.forward_profiled(xs, attention_mask, seqlen_offset)
            .map(|(output, _)| output)
    }

    fn clear_kv_cache(&mut self) {
        match &mut self.token_mixer {
            LayerKind::Linear(linear_attn) => linear_attn.clear_kv_cache(),
            LayerKind::Full(self_attn) => self_attn.clear_kv_cache(),
        }
    }

    pub fn layer_type(&self) -> &str {
        &self.layer_type
    }
}

#[derive(Debug, Clone)]
pub struct TextModel {
    embed_tokens: Embedding,
    layers: Vec<DecoderLayer>,
    norm: Qwen35RmsNorm,
    device: Device,
    dtype: DType,
}

impl TextModel {
    pub fn new(cfg: &TextConfig, vb: VarBuilder) -> Result<Self> {
        let cfg = cfg.clone().normalized();
        let vb_m = vb.pp("model").pp("language_model");
        let embed_tokens = embedding(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_tokens"))?;
        let rotary_emb = Arc::new(RotaryEmbedding::new(&cfg, vb.device(), vb.dtype())?);
        let vb_l = vb_m.pp("layers");
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for layer_idx in 0..cfg.num_hidden_layers {
            layers.push(DecoderLayer::new(
                &cfg,
                layer_idx,
                rotary_emb.clone(),
                vb_l.pp(layer_idx),
            )?);
        }
        Ok(Self {
            embed_tokens,
            layers,
            norm: Qwen35RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb_m.pp("norm"))?,
            device: vb.device().clone(),
            dtype: vb.dtype(),
        })
    }

    fn prepare_causal_attention_mask(
        &self,
        b_size: usize,
        tgt_len: usize,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        let lower = Tensor::tril2(tgt_len, DType::U8, &self.device)?;
        let on_true = lower.zeros_like()?.to_dtype(self.dtype)?;
        let on_false = Tensor::full(f32::NEG_INFINITY, (tgt_len, tgt_len), &self.device)?
            .to_dtype(self.dtype)?;
        let mask = lower.where_cond(&on_true, &on_false)?;
        let mask = if seqlen_offset > 0 {
            let prefix = Tensor::zeros((tgt_len, seqlen_offset), self.dtype, &self.device)?;
            Tensor::cat(&[&prefix, &mask], D::Minus1)?
        } else {
            mask
        };
        mask.expand((b_size, 1, tgt_len, tgt_len + seqlen_offset))
    }

    pub fn linear_attention_layer_ids(&self) -> Vec<usize> {
        self.layers
            .iter()
            .enumerate()
            .filter_map(|(layer_id, layer)| {
                (layer.layer_type() == "linear_attention").then_some(layer_id)
            })
            .collect()
    }

    pub fn bench_linear_attention_layer(
        &mut self,
        input_ids: &Tensor,
        target_layer: usize,
        seqlen_offset: usize,
        repeats: usize,
    ) -> Result<LinearAttentionBenchResult> {
        if repeats == 0 {
            candle::bail!("linear-attention bench requires repeats > 0");
        }
        if target_layer >= self.layers.len() {
            candle::bail!(
                "linear-attention bench target layer {} is out of range for {} layers",
                target_layer,
                self.layers.len()
            );
        }
        if self.layers[target_layer].layer_type() != "linear_attention" {
            candle::bail!(
                "linear-attention bench target layer {} is {:?}, expected \"linear_attention\"",
                target_layer,
                self.layers[target_layer].layer_type()
            );
        }

        self.clear_kv_cache();
        let (b_size, seq_len) = input_ids.dims2()?;
        let attention_mask = if seq_len > 1 {
            Some(self.prepare_causal_attention_mask(b_size, seq_len, seqlen_offset)?)
        } else {
            None
        };
        let mut xs = self.embed_tokens.forward(input_ids)?;
        for layer in self.layers.iter_mut().take(target_layer) {
            let mask = if layer.layer_type() == "full_attention" {
                attention_mask.as_ref()
            } else {
                None
            };
            let (next_xs, _) = layer.forward_profiled(&xs, mask, seqlen_offset)?;
            xs = next_xs;
        }
        self.clear_kv_cache();

        let mut total_profile = RuntimeProfile::default();
        let mut total_millis_acc = 0.0;
        let mut best_total_millis = f64::INFINITY;
        let mut best_profile = RuntimeProfile::default();
        let mut iteration_total_millis = Vec::with_capacity(repeats);
        let device = input_ids.device();
        for _ in 0..repeats {
            self.layers[target_layer].clear_kv_cache();
            let iteration_start = profile_start(device)?;
            let (_, profile) =
                self.layers[target_layer].forward_profiled(&xs, None, seqlen_offset)?;
            let total_millis = profile_elapsed(iteration_start, device)?;
            iteration_total_millis.push(total_millis);
            total_millis_acc += total_millis;
            total_profile.add_assign(&profile);
            if total_millis < best_total_millis {
                best_total_millis = total_millis;
                best_profile = profile.clone();
            }
        }
        self.clear_kv_cache();

        Ok(LinearAttentionBenchResult {
            layer_id: target_layer,
            sequence_length: seq_len,
            repeats,
            mean_total_millis: total_millis_acc / repeats as f64,
            best_total_millis,
            iteration_total_millis,
            mean_profile: total_profile.scaled(1.0 / repeats as f64),
            best_profile,
        })
    }

    pub fn trace_linear_attention_layer(
        &mut self,
        input_ids: &Tensor,
        target_layer: usize,
        seqlen_offset: usize,
    ) -> Result<LinearAttentionTrace> {
        if target_layer >= self.layers.len() {
            candle::bail!(
                "linear-attention trace target layer {} is out of range for {} layers",
                target_layer,
                self.layers.len()
            );
        }
        if self.layers[target_layer].layer_type() != "linear_attention" {
            candle::bail!(
                "linear-attention trace target layer {} is {:?}, expected \"linear_attention\"",
                target_layer,
                self.layers[target_layer].layer_type()
            );
        }

        self.clear_kv_cache();
        let (b_size, seq_len) = input_ids.dims2()?;
        let attention_mask = if seq_len > 1 {
            Some(self.prepare_causal_attention_mask(b_size, seq_len, seqlen_offset)?)
        } else {
            None
        };
        let mut xs = self.embed_tokens.forward(input_ids)?;
        for layer in self.layers.iter_mut().take(target_layer) {
            let mask = if layer.layer_type() == "full_attention" {
                attention_mask.as_ref()
            } else {
                None
            };
            let (next_xs, _) = layer.forward_profiled(&xs, mask, seqlen_offset)?;
            xs = next_xs;
        }

        let target = self
            .layers
            .get_mut(target_layer)
            .expect("target layer index already validated");
        let (layer_output, recurrent_state, profile) = match &mut target.token_mixer {
            LayerKind::Linear(linear_attn) => linear_attn.trace_profiled(&xs, None)?,
            LayerKind::Full(_) => unreachable!("target layer is validated as linear attention"),
        };
        self.clear_kv_cache();

        Ok(LinearAttentionTrace {
            layer_id: target_layer,
            sequence_length: seq_len,
            layer_output,
            recurrent_state,
            profile,
        })
    }

    pub fn forward_profiled_with_linear_traces(
        &mut self,
        input_ids: &Tensor,
        seqlen_offset: usize,
        target_layers: &[usize],
    ) -> Result<(Tensor, Vec<LinearAttentionTrace>, RuntimeProfile)> {
        let device = input_ids.device();
        let (b_size, seq_len) = input_ids.dims2()?;
        let mut profile = RuntimeProfile::default();
        let scheduler_start = profile_start(device)?;
        let attention_mask = if seq_len > 1 {
            Some(self.prepare_causal_attention_mask(b_size, seq_len, seqlen_offset)?)
        } else {
            None
        };
        profile.scheduler_planning_millis += profile_elapsed(scheduler_start, device)?;
        let mut xs = self.embed_tokens.forward(input_ids)?;
        let mut traces = Vec::new();
        for (layer_id, layer) in self.layers.iter_mut().enumerate() {
            let mask = if layer.layer_type() == "full_attention" {
                attention_mask.as_ref()
            } else {
                None
            };
            let should_trace = target_layers.contains(&layer_id);
            let (next_xs, maybe_trace, layer_profile) = match &mut layer.token_mixer {
                LayerKind::Linear(linear_attn) if should_trace => {
                    let (layer_output, recurrent_state, layer_profile) =
                        linear_attn.trace_profiled(&layer.input_layernorm.forward(&xs)?, None)?;
                    linear_attn.recurrent_state = Some(recurrent_state.clone());
                    let residual = &xs;
                    let attn_residual = residual.broadcast_add(&layer_output)?;
                    let post_norm = layer.post_attention_layernorm.forward(&attn_residual)?;
                    let mlp_start = profile_start(device)?;
                    let mlp_out = layer.mlp.forward(&post_norm)?;
                    let mut profile = layer_profile;
                    profile.mlp_millis += profile_elapsed(mlp_start, device)?;
                    let next_xs = attn_residual.broadcast_add(&mlp_out)?;
                    (
                        next_xs,
                        Some(LinearAttentionTrace {
                            layer_id,
                            sequence_length: seq_len,
                            layer_output,
                            recurrent_state,
                            profile: profile.clone(),
                        }),
                        profile,
                    )
                }
                _ => {
                    let (next_xs, layer_profile) =
                        layer.forward_profiled(&xs, mask, seqlen_offset)?;
                    (next_xs, None, layer_profile)
                }
            };
            if let Some(trace) = maybe_trace {
                traces.push(trace);
            }
            profile.add_assign(&layer_profile);
            xs = next_xs;
        }
        Ok((self.norm.forward(&xs)?, traces, profile))
    }

    pub fn forward_profiled(
        &mut self,
        input_ids: &Tensor,
        seqlen_offset: usize,
    ) -> Result<(Tensor, RuntimeProfile)> {
        let device = input_ids.device();
        let (b_size, seq_len) = input_ids.dims2()?;
        let mut profile = RuntimeProfile::default();
        let scheduler_start = profile_start(device)?;
        let attention_mask = if seq_len > 1 {
            Some(self.prepare_causal_attention_mask(b_size, seq_len, seqlen_offset)?)
        } else {
            None
        };
        profile.scheduler_planning_millis += profile_elapsed(scheduler_start, device)?;
        let mut xs = self.embed_tokens.forward(input_ids)?;
        for layer in self.layers.iter_mut() {
            let mask = if layer.layer_type() == "full_attention" {
                attention_mask.as_ref()
            } else {
                None
            };
            let (next_xs, layer_profile) = layer.forward_profiled(&xs, mask, seqlen_offset)?;
            profile.add_assign(&layer_profile);
            xs = next_xs;
        }
        Ok((self.norm.forward(&xs)?, profile))
    }

    pub fn forward(&mut self, input_ids: &Tensor, seqlen_offset: usize) -> Result<Tensor> {
        self.forward_profiled(input_ids, seqlen_offset)
            .map(|(output, _)| output)
    }

    pub fn clear_kv_cache(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.clear_kv_cache();
        }
    }
}

#[derive(Debug, Clone)]
pub struct ModelForCausalLM {
    language_model: TextModel,
    lm_head: Linear,
}

impl ModelForCausalLM {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let cfg = cfg.clone().normalized();
        let language_model = TextModel::new(&cfg.text_config, vb.clone())?;
        let lm_head = if vb.contains_tensor("lm_head.weight") {
            linear_no_bias(
                cfg.text_config.hidden_size,
                cfg.text_config.vocab_size,
                vb.pp("lm_head"),
            )?
        } else {
            Linear::from_weights(language_model.embed_tokens.embeddings().clone(), None)
        };
        Ok(Self {
            language_model,
            lm_head,
        })
    }

    pub fn forward_profiled(
        &mut self,
        input_ids: &Tensor,
        seqlen_offset: usize,
    ) -> Result<(Tensor, RuntimeProfile)> {
        let device = input_ids.device();
        let (_, seq_len) = input_ids.dims2()?;
        let (hidden_states, mut profile) = self
            .language_model
            .forward_profiled(input_ids, seqlen_offset)?;
        let output_start = profile_start(device)?;
        let logits = hidden_states
            .narrow(1, seq_len - 1, 1)?
            .apply(&self.lm_head)?;
        profile.output_projection_millis += profile_elapsed(output_start, device)?;
        Ok((logits, profile))
    }

    pub fn forward_profiled_with_linear_traces(
        &mut self,
        input_ids: &Tensor,
        seqlen_offset: usize,
        target_layers: &[usize],
    ) -> Result<(Tensor, Vec<LinearAttentionTrace>, RuntimeProfile)> {
        let device = input_ids.device();
        let (_, seq_len) = input_ids.dims2()?;
        let (hidden_states, traces, mut profile) = self
            .language_model
            .forward_profiled_with_linear_traces(input_ids, seqlen_offset, target_layers)?;
        let output_start = profile_start(device)?;
        let logits = hidden_states
            .narrow(1, seq_len - 1, 1)?
            .apply(&self.lm_head)?;
        profile.output_projection_millis += profile_elapsed(output_start, device)?;
        Ok((logits, traces, profile))
    }

    pub fn forward(&mut self, input_ids: &Tensor, seqlen_offset: usize) -> Result<Tensor> {
        self.forward_profiled(input_ids, seqlen_offset)
            .map(|(output, _)| output)
    }

    pub fn linear_attention_layer_ids(&self) -> Vec<usize> {
        self.language_model.linear_attention_layer_ids()
    }

    pub fn bench_linear_attention_layer(
        &mut self,
        input_ids: &Tensor,
        target_layer: usize,
        seqlen_offset: usize,
        repeats: usize,
    ) -> Result<LinearAttentionBenchResult> {
        self.language_model.bench_linear_attention_layer(
            input_ids,
            target_layer,
            seqlen_offset,
            repeats,
        )
    }

    pub fn trace_linear_attention_layer(
        &mut self,
        input_ids: &Tensor,
        target_layer: usize,
        seqlen_offset: usize,
    ) -> Result<LinearAttentionTrace> {
        self.language_model
            .trace_linear_attention_layer(input_ids, target_layer, seqlen_offset)
    }

    pub fn clear_kv_cache(&mut self) {
        self.language_model.clear_kv_cache()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "hip")]
    fn assert_close(lhs: &[f32], rhs: &[f32], tol: f32) {
        assert_eq!(lhs.len(), rhs.len());
        for (idx, (lhs, rhs)) in lhs.iter().zip(rhs.iter()).enumerate() {
            let delta = (lhs - rhs).abs();
            assert!(
                delta <= tol,
                "mismatch at {idx}: lhs={lhs} rhs={rhs} delta={delta} tol={tol}"
            );
        }
    }

    #[cfg(feature = "hip")]
    #[test]
    fn hip_linear_prefill_conv_pack_matches_reference() -> Result<()> {
        let device = Device::new_hip(0)?;
        let batch_size = 1usize;
        let conv_dim = 2usize;
        let total_len = 6usize;
        let seq_len = 4usize;
        let kernel_size = 3usize;

        let mixed_qkv_data = vec![
            0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6, // channel 0
            -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, // channel 1
        ];
        let weight_data = vec![
            0.5f32, -0.25, 0.75, // channel 0
            -0.4, 0.3, 0.2, // channel 1
        ];
        let mixed_qkv = Tensor::from_vec(
            mixed_qkv_data.clone(),
            (batch_size, conv_dim, total_len),
            &device,
        )?;
        let weights = Tensor::from_vec(weight_data.clone(), (conv_dim, kernel_size), &device)?;
        let output = linear_prefill_conv_pack(&mixed_qkv, &weights, seq_len, kernel_size)?;
        let output = output.flatten_all()?.to_vec1::<f32>()?;

        let mut expected = Vec::with_capacity(batch_size * seq_len * conv_dim);
        for b in 0..batch_size {
            for t in 0..seq_len {
                for c in 0..conv_dim {
                    let input_base = b * conv_dim * total_len + c * total_len;
                    let weight_base = c * kernel_size;
                    let mut acc = 0.0f32;
                    for tap in 0..kernel_size {
                        acc +=
                            mixed_qkv_data[input_base + t + tap] * weight_data[weight_base + tap];
                    }
                    expected.push(acc / (1.0 + (-acc).exp()));
                }
            }
        }

        assert_close(&output, &expected, 1e-5);
        Ok(())
    }

    #[cfg(feature = "hip")]
    #[test]
    fn hip_full_attention_prefill_matches_reference() -> Result<()> {
        let device = Device::new_hip(0)?;
        let batch_size = 1usize;
        let q_heads = 2usize;
        let kv_heads = 1usize;
        let q_len = 3usize;
        let kv_len = 5usize;
        let head_dim = 4usize;
        let num_kv_groups = 2usize;
        let scale = 0.5f32;
        let seqlen_offset = 2usize;

        let query_data = vec![
            0.2f32, 0.0, 0.1, -0.1, 0.1, 0.3, -0.2, 0.0, 0.4, -0.1, 0.0, 0.2, // head 0
            -0.2, 0.1, 0.0, 0.3, 0.2, -0.3, 0.1, 0.0, 0.0, 0.2, 0.2, -0.1, // head 1
        ];
        let key_data = vec![
            0.1f32, 0.0, 0.2, -0.1, 0.0, 0.3, -0.2, 0.1, 0.2, -0.1, 0.0, 0.4, -0.3, 0.2, 0.1, 0.0,
            0.1, 0.1, -0.1, 0.2,
        ];
        let value_data = vec![
            0.0f32, 0.2, -0.1, 0.3, 0.1, -0.2, 0.0, 0.2, 0.4, 0.1, -0.3, 0.0, -0.1, 0.3, 0.2, -0.2,
            0.2, 0.0, 0.1, 0.4,
        ];

        let query = Tensor::from_vec(
            query_data.clone(),
            (batch_size, q_heads, q_len, head_dim),
            &device,
        )?;
        let key = Tensor::from_vec(
            key_data.clone(),
            (batch_size, kv_heads, kv_len, head_dim),
            &device,
        )?;
        let value = Tensor::from_vec(
            value_data.clone(),
            (batch_size, kv_heads, kv_len, head_dim),
            &device,
        )?;
        let output = full_attention_prefill_megakernel(
            &query,
            &key,
            &value,
            num_kv_groups,
            scale,
            seqlen_offset,
        )?;
        let output = output.flatten_all()?.to_vec1::<f32>()?;

        let mut expected = Vec::with_capacity(batch_size * q_heads * q_len * head_dim);
        for b in 0..batch_size {
            for q_head in 0..q_heads {
                let kv_head = q_head / num_kv_groups;
                for q_pos in 0..q_len {
                    let causal_limit = kv_len.min(seqlen_offset + q_pos + 1);
                    let query_offset = ((b * q_heads + q_head) * q_len + q_pos) * head_dim;
                    let q_row = &query_data[query_offset..query_offset + head_dim];
                    let key_head_offset = (b * kv_heads + kv_head) * kv_len * head_dim;
                    let value_head_offset = key_head_offset;

                    let mut max_score = f32::NEG_INFINITY;
                    let mut denom = 0.0f32;
                    let mut out_row = vec![0.0f32; head_dim];
                    for k_pos in 0..causal_limit {
                        let key_offset = key_head_offset + k_pos * head_dim;
                        let value_offset = value_head_offset + k_pos * head_dim;
                        let mut score = 0.0f32;
                        for d in 0..head_dim {
                            score += q_row[d] * key_data[key_offset + d];
                        }
                        score *= scale;

                        if !max_score.is_finite() {
                            max_score = score;
                            denom = 1.0;
                            out_row.copy_from_slice(
                                &value_data[value_offset..value_offset + head_dim],
                            );
                            continue;
                        }

                        let new_max = max_score.max(score);
                        let prev_scale = (max_score - new_max).exp();
                        let curr_scale = (score - new_max).exp();
                        denom = denom * prev_scale + curr_scale;
                        for d in 0..head_dim {
                            out_row[d] =
                                out_row[d] * prev_scale + curr_scale * value_data[value_offset + d];
                        }
                        max_score = new_max;
                    }

                    let inv_denom = if denom > 0.0 { 1.0 / denom } else { 0.0 };
                    for value in out_row {
                        expected.push(value * inv_denom);
                    }
                }
            }
        }

        assert_close(&output, &expected, 1e-5);
        Ok(())
    }

    #[cfg(feature = "hip")]
    #[test]
    fn hip_delta_recurrent_prefill_matches_reference() -> Result<()> {
        let device = Device::new_hip(0)?;
        let batch_heads = 1usize;
        let seq_len = 3usize;
        let k_head_dim = 2usize;
        let v_head_dim = 2usize;

        let initial_state_data = vec![0.1f32, -0.2, 0.05, 0.3];
        let query_data = vec![0.2f32, -0.1, 0.0, 0.3, -0.2, 0.4];
        let key_data = vec![0.1f32, 0.2, -0.3, 0.5, 0.4, -0.2];
        let value_data = vec![0.3f32, -0.1, 0.2, 0.4, -0.2, 0.1];
        let beta_data = vec![0.5f32, 0.25, 0.75];
        let g_data = vec![0.0f32, -0.2, 0.1];

        let initial_state = Tensor::from_vec(
            initial_state_data.clone(),
            (batch_heads, k_head_dim, v_head_dim),
            &device,
        )?;
        let query = Tensor::from_vec(
            query_data.clone(),
            (batch_heads, seq_len, k_head_dim),
            &device,
        )?;
        let key = Tensor::from_vec(
            key_data.clone(),
            (batch_heads, seq_len, k_head_dim),
            &device,
        )?;
        let value = Tensor::from_vec(
            value_data.clone(),
            (batch_heads, seq_len, v_head_dim),
            &device,
        )?;
        let beta = Tensor::from_vec(beta_data.clone(), (batch_heads, seq_len), &device)?;
        let g = Tensor::from_vec(g_data.clone(), (batch_heads, seq_len), &device)?;

        let output = delta_recurrent_prefill(&initial_state, &query, &key, &value, &beta, &g)?;
        let output = output.flatten_all()?.to_vec1::<f32>()?;

        let mut expected = vec![0.0f32; batch_heads * (seq_len + k_head_dim) * v_head_dim];
        for bh in 0..batch_heads {
            for v_idx in 0..v_head_dim {
                let mut state = vec![0.0f32; k_head_dim];
                for k_idx in 0..k_head_dim {
                    state[k_idx] = initial_state_data
                        [bh * k_head_dim * v_head_dim + k_idx * v_head_dim + v_idx];
                }
                let out_base = bh * (seq_len + k_head_dim) * v_head_dim;
                for t in 0..seq_len {
                    let g_t = g_data[bh * seq_len + t].exp();
                    let key_row = bh * seq_len * k_head_dim + t * k_head_dim;
                    let value_row = bh * seq_len * v_head_dim + t * v_head_dim;
                    for entry in &mut state {
                        *entry *= g_t;
                    }
                    let mut kv_mem = 0.0f32;
                    for k_idx in 0..k_head_dim {
                        kv_mem += state[k_idx] * key_data[key_row + k_idx];
                    }
                    let delta =
                        (value_data[value_row + v_idx] - kv_mem) * beta_data[bh * seq_len + t];
                    for k_idx in 0..k_head_dim {
                        state[k_idx] += key_data[key_row + k_idx] * delta;
                    }
                    let mut out_t = 0.0f32;
                    for k_idx in 0..k_head_dim {
                        out_t += state[k_idx] * query_data[key_row + k_idx];
                    }
                    expected[out_base + t * v_head_dim + v_idx] = out_t;
                }
                let state_out = out_base + seq_len * v_head_dim;
                for k_idx in 0..k_head_dim {
                    expected[state_out + k_idx * v_head_dim + v_idx] = state[k_idx];
                }
            }
        }

        assert_close(&output, &expected, 1e-5);
        Ok(())
    }

    #[cfg(feature = "hip")]
    #[test]
    fn hip_delta_chunk_step_matches_reference() -> Result<()> {
        let device = Device::new_hip(0)?;
        let batch_heads = 1usize;
        let chunk_size = 3usize;
        let k_head_dim = 2usize;
        let v_head_dim = 2usize;

        let prev_state_data = vec![0.15f32, -0.05, 0.2, 0.1];
        let query_data = vec![0.1f32, 0.3, -0.2, 0.4, 0.5, -0.1];
        let key_data = vec![0.2f32, -0.1, 0.0, 0.25, -0.3, 0.15];
        let value_data = vec![0.4f32, 0.2, -0.1, 0.3, 0.05, -0.2];
        let beta_data = vec![0.6f32, 0.5, 0.4];
        let g_data = vec![-0.1f32, 0.0, 0.2];

        let prev_state = Tensor::from_vec(
            prev_state_data.clone(),
            (batch_heads, k_head_dim, v_head_dim),
            &device,
        )?;
        let query = Tensor::from_vec(
            query_data.clone(),
            (batch_heads, chunk_size, k_head_dim),
            &device,
        )?;
        let key = Tensor::from_vec(
            key_data.clone(),
            (batch_heads, chunk_size, k_head_dim),
            &device,
        )?;
        let value = Tensor::from_vec(
            value_data.clone(),
            (batch_heads, chunk_size, v_head_dim),
            &device,
        )?;
        let beta = Tensor::from_vec(beta_data.clone(), (batch_heads, chunk_size), &device)?;
        let g = Tensor::from_vec(g_data.clone(), (batch_heads, chunk_size), &device)?;

        let output = delta_chunk_step_raw(&prev_state, &query, &key, &value, &beta, &g)?;
        let output = output.flatten_all()?.to_vec1::<f32>()?;

        let mut expected = vec![0.0f32; batch_heads * (chunk_size + k_head_dim) * v_head_dim];
        for bh in 0..batch_heads {
            for v_idx in 0..v_head_dim {
                let mut state = vec![0.0f32; k_head_dim];
                for k_idx in 0..k_head_dim {
                    state[k_idx] =
                        prev_state_data[bh * k_head_dim * v_head_dim + k_idx * v_head_dim + v_idx];
                }
                let out_base = bh * (chunk_size + k_head_dim) * v_head_dim;
                for t in 0..chunk_size {
                    let g_t = g_data[bh * chunk_size + t].exp();
                    let key_row = bh * chunk_size * k_head_dim + t * k_head_dim;
                    let value_row = bh * chunk_size * v_head_dim + t * v_head_dim;
                    for entry in &mut state {
                        *entry *= g_t;
                    }
                    let mut kv_mem = 0.0f32;
                    for k_idx in 0..k_head_dim {
                        kv_mem += state[k_idx] * key_data[key_row + k_idx];
                    }
                    let delta =
                        (value_data[value_row + v_idx] - kv_mem) * beta_data[bh * chunk_size + t];
                    for k_idx in 0..k_head_dim {
                        state[k_idx] += key_data[key_row + k_idx] * delta;
                    }
                    let mut out_t = 0.0f32;
                    for k_idx in 0..k_head_dim {
                        out_t += state[k_idx] * query_data[key_row + k_idx];
                    }
                    expected[out_base + t * v_head_dim + v_idx] = out_t;
                }
                let state_out = out_base + chunk_size * v_head_dim;
                for k_idx in 0..k_head_dim {
                    expected[state_out + k_idx * v_head_dim + v_idx] = state[k_idx];
                }
            }
        }

        assert_close(&output, &expected, 1e-5);
        Ok(())
    }

    #[cfg(feature = "hip")]
    #[test]
    fn hip_delta_chunk_step_windowed_matches_reference() -> Result<()> {
        let device = Device::new_hip(0)?;
        let batch_heads = 1usize;
        let num_chunks = 2usize;
        let chunk_size = 2usize;
        let k_head_dim = 2usize;
        let v_head_dim = 2usize;

        let prev_state_data = vec![0.1f32, -0.1, 0.2, 0.05];
        let query_data = vec![0.2f32, 0.1, -0.3, 0.4, 0.0, 0.25, 0.5, -0.2];
        let key_data = vec![0.05f32, 0.2, -0.1, 0.3, 0.4, -0.2, 0.15, 0.1];
        let value_data = vec![0.3f32, -0.2, 0.1, 0.5, -0.1, 0.2, 0.4, 0.0];
        let beta_data = vec![0.5f32, 0.25, 0.75, 0.4];
        let g_data = vec![0.0f32, -0.3, 0.2, 0.1];

        let prev_state = Tensor::from_vec(
            prev_state_data.clone(),
            (batch_heads, k_head_dim, v_head_dim),
            &device,
        )?;
        let query = Tensor::from_vec(
            query_data.clone(),
            (batch_heads, num_chunks, chunk_size, k_head_dim),
            &device,
        )?;
        let key = Tensor::from_vec(
            key_data.clone(),
            (batch_heads, num_chunks, chunk_size, k_head_dim),
            &device,
        )?;
        let value = Tensor::from_vec(
            value_data.clone(),
            (batch_heads, num_chunks, chunk_size, v_head_dim),
            &device,
        )?;
        let beta = Tensor::from_vec(
            beta_data.clone(),
            (batch_heads, num_chunks, chunk_size),
            &device,
        )?;
        let g = Tensor::from_vec(
            g_data.clone(),
            (batch_heads, num_chunks, chunk_size),
            &device,
        )?;

        let output = delta_chunk_step_windowed_raw(&prev_state, &query, &key, &value, &beta, &g)?;
        let output = output.flatten_all()?.to_vec1::<f32>()?;

        let total_tokens = num_chunks * chunk_size;
        let mut expected = vec![0.0f32; batch_heads * (total_tokens + k_head_dim) * v_head_dim];
        for bh in 0..batch_heads {
            for v_idx in 0..v_head_dim {
                let mut state = vec![0.0f32; k_head_dim];
                for k_idx in 0..k_head_dim {
                    state[k_idx] =
                        prev_state_data[bh * k_head_dim * v_head_dim + k_idx * v_head_dim + v_idx];
                }
                let out_base = bh * (total_tokens + k_head_dim) * v_head_dim;
                for t in 0..total_tokens {
                    let key_row = bh * total_tokens * k_head_dim + t * k_head_dim;
                    let value_row = bh * total_tokens * v_head_dim + t * v_head_dim;
                    let g_t = g_data[bh * total_tokens + t].exp();
                    for entry in &mut state {
                        *entry *= g_t;
                    }
                    let mut kv_mem = 0.0f32;
                    for k_idx in 0..k_head_dim {
                        kv_mem += state[k_idx] * key_data[key_row + k_idx];
                    }
                    let delta =
                        (value_data[value_row + v_idx] - kv_mem) * beta_data[bh * total_tokens + t];
                    for k_idx in 0..k_head_dim {
                        state[k_idx] += key_data[key_row + k_idx] * delta;
                    }
                    let mut out_t = 0.0f32;
                    for k_idx in 0..k_head_dim {
                        out_t += state[k_idx] * query_data[key_row + k_idx];
                    }
                    expected[out_base + t * v_head_dim + v_idx] = out_t;
                }
                let state_out = out_base + total_tokens * v_head_dim;
                for k_idx in 0..k_head_dim {
                    expected[state_out + k_idx * v_head_dim + v_idx] = state[k_idx];
                }
            }
        }

        assert_close(&output, &expected, 1e-5);
        Ok(())
    }

    #[cfg(feature = "hip")]
    #[test]
    fn hip_delta_chunk_scan_raw_matches_reference() -> Result<()> {
        let device = Device::new_hip(0)?;
        let batch_heads = 1usize;
        let num_chunks = 2usize;
        let chunk_size = 2usize;
        let k_head_dim = 2usize;
        let v_head_dim = 2usize;

        let initial_state_data = vec![0.05f32, 0.1, -0.2, 0.15];
        let query_data = vec![0.1f32, -0.2, 0.3, 0.4, -0.1, 0.2, 0.5, -0.3];
        let key_data = vec![0.2f32, 0.0, -0.15, 0.35, 0.25, -0.2, 0.1, 0.05];
        let value_data = vec![0.3f32, 0.1, -0.2, 0.4, 0.05, -0.1, 0.2, 0.3];
        let beta_data = vec![0.4f32, 0.7, 0.5, 0.6];
        let g_data = vec![-0.2f32, 0.0, 0.1, -0.1];

        let initial_state = Tensor::from_vec(
            initial_state_data.clone(),
            (batch_heads, k_head_dim, v_head_dim),
            &device,
        )?;
        let query = Tensor::from_vec(
            query_data.clone(),
            (batch_heads, num_chunks, chunk_size, k_head_dim),
            &device,
        )?;
        let key = Tensor::from_vec(
            key_data.clone(),
            (batch_heads, num_chunks, chunk_size, k_head_dim),
            &device,
        )?;
        let value = Tensor::from_vec(
            value_data.clone(),
            (batch_heads, num_chunks, chunk_size, v_head_dim),
            &device,
        )?;
        let beta = Tensor::from_vec(
            beta_data.clone(),
            (batch_heads, num_chunks, chunk_size),
            &device,
        )?;
        let g = Tensor::from_vec(
            g_data.clone(),
            (batch_heads, num_chunks, chunk_size),
            &device,
        )?;

        let output = delta_chunk_scan_raw(&initial_state, &query, &key, &value, &beta, &g)?;
        let output = output.flatten_all()?.to_vec1::<f32>()?;

        let total_tokens = num_chunks * chunk_size;
        let mut expected = vec![0.0f32; batch_heads * (total_tokens + k_head_dim) * v_head_dim];
        for bh in 0..batch_heads {
            for v_idx in 0..v_head_dim {
                let mut state = vec![0.0f32; k_head_dim];
                for k_idx in 0..k_head_dim {
                    state[k_idx] = initial_state_data
                        [bh * k_head_dim * v_head_dim + k_idx * v_head_dim + v_idx];
                }
                let out_base = bh * (total_tokens + k_head_dim) * v_head_dim;
                for t in 0..total_tokens {
                    let key_row = bh * total_tokens * k_head_dim + t * k_head_dim;
                    let value_row = bh * total_tokens * v_head_dim + t * v_head_dim;
                    let g_t = g_data[bh * total_tokens + t].exp();
                    for entry in &mut state {
                        *entry *= g_t;
                    }
                    let mut kv_mem = 0.0f32;
                    for k_idx in 0..k_head_dim {
                        kv_mem += state[k_idx] * key_data[key_row + k_idx];
                    }
                    let delta =
                        (value_data[value_row + v_idx] - kv_mem) * beta_data[bh * total_tokens + t];
                    for k_idx in 0..k_head_dim {
                        state[k_idx] += key_data[key_row + k_idx] * delta;
                    }
                    let mut out_t = 0.0f32;
                    for k_idx in 0..k_head_dim {
                        out_t += state[k_idx] * query_data[key_row + k_idx];
                    }
                    expected[out_base + t * v_head_dim + v_idx] = out_t;
                }
                let state_out = out_base + total_tokens * v_head_dim;
                for k_idx in 0..k_head_dim {
                    expected[state_out + k_idx * v_head_dim + v_idx] = state[k_idx];
                }
            }
        }

        assert_close(&output, &expected, 1e-5);
        Ok(())
    }

    #[cfg(feature = "hip")]
    #[test]
    fn hip_delta_state_scan_matches_reference() -> Result<()> {
        let device = Device::new_hip(0)?;
        let batch_heads = 1usize;
        let num_chunks = 2usize;
        let chunk_size = 2usize;
        let k_head_dim = 2usize;
        let v_head_dim = 2usize;
        let packed_width = 2 * k_head_dim + 1;

        let initial_state_data = vec![0.1f32, -0.2, 0.05, 0.15];
        let packed_scan_data = vec![
            0.2f32, -0.1, 0.05, 0.3, 0.9, -0.2, 0.4, 0.1, -0.05, 0.9, 0.3, 0.1, -0.15, 0.2, 0.8,
            0.05, -0.25, 0.2, 0.1, 0.8,
        ];
        let value_data = vec![0.4f32, 0.1, -0.2, 0.3, 0.05, -0.1, 0.2, 0.25];

        let initial_state = Tensor::from_vec(
            initial_state_data.clone(),
            (batch_heads, k_head_dim, v_head_dim),
            &device,
        )?;
        let packed_scan = Tensor::from_vec(
            packed_scan_data.clone(),
            (batch_heads, num_chunks, chunk_size, packed_width),
            &device,
        )?;
        let value = Tensor::from_vec(
            value_data.clone(),
            (batch_heads, num_chunks, chunk_size, v_head_dim),
            &device,
        )?;

        let output = delta_state_scan(&initial_state, &packed_scan, &value)?;
        let output = output.flatten_all()?.to_vec1::<f32>()?;

        let mut expected = vec![0.0f32; batch_heads * (num_chunks + 1) * k_head_dim * v_head_dim];
        for bh in 0..batch_heads {
            for v_idx in 0..v_head_dim {
                let mut state = vec![0.0f32; k_head_dim];
                for k_idx in 0..k_head_dim {
                    let idx = bh * k_head_dim * v_head_dim + k_idx * v_head_dim + v_idx;
                    state[k_idx] = initial_state_data[idx];
                    expected[idx] = state[k_idx];
                }
                for chunk in 0..num_chunks {
                    let packed_chunk_base = ((bh * num_chunks) + chunk) * chunk_size * packed_width;
                    let value_chunk_base = ((bh * num_chunks) + chunk) * chunk_size * v_head_dim;
                    let state_decay = packed_scan_data[packed_chunk_base + 2 * k_head_dim];
                    let mut update = vec![0.0f32; k_head_dim];
                    for t in 0..chunk_size {
                        let packed_row = packed_chunk_base + t * packed_width;
                        let value_row = value_chunk_base + t * v_head_dim;
                        let mut v_prime = 0.0f32;
                        for k_idx in 0..k_head_dim {
                            v_prime +=
                                packed_scan_data[packed_row + k_head_dim + k_idx] * state[k_idx];
                        }
                        let v_new = value_data[value_row + v_idx] - v_prime;
                        for k_idx in 0..k_head_dim {
                            update[k_idx] += packed_scan_data[packed_row + k_idx] * v_new;
                        }
                    }
                    let out_chunk_base =
                        ((bh * (num_chunks + 1)) + (chunk + 1)) * k_head_dim * v_head_dim;
                    for k_idx in 0..k_head_dim {
                        state[k_idx] = state_decay * state[k_idx] + update[k_idx];
                        expected[out_chunk_base + k_idx * v_head_dim + v_idx] = state[k_idx];
                    }
                }
            }
        }

        assert_close(&output, &expected, 1e-5);
        Ok(())
    }

    #[cfg(feature = "hip")]
    #[test]
    fn hip_delta_chunk_fused_matches_reference() -> Result<()> {
        let device = Device::new_hip(0)?;
        let batch_heads = 1usize;
        let chunk_size = 2usize;
        let k_head_dim = 2usize;
        let v_head_dim = 2usize;
        let packed_width = 3 * k_head_dim + 1;

        let prev_state_data = vec![0.1f32, 0.2, -0.05, 0.15];
        let packed_chunk_data = vec![
            0.2f32, -0.1, 0.05, 0.3, 0.1, -0.2, 0.85, -0.15, 0.25, 0.2, -0.05, -0.1, 0.15, 0.85,
        ];
        let value_data = vec![0.35f32, -0.1, 0.05, 0.4];

        let prev_state = Tensor::from_vec(
            prev_state_data.clone(),
            (batch_heads, k_head_dim, v_head_dim),
            &device,
        )?;
        let packed_chunk = Tensor::from_vec(
            packed_chunk_data.clone(),
            (batch_heads, chunk_size, packed_width),
            &device,
        )?;
        let value = Tensor::from_vec(
            value_data.clone(),
            (batch_heads, chunk_size, v_head_dim),
            &device,
        )?;

        let output = delta_chunk_fused(&prev_state, &packed_chunk, &value)?;
        let output = output.flatten_all()?.to_vec1::<f32>()?;

        let mut expected = vec![0.0f32; batch_heads * (2 * chunk_size + k_head_dim) * v_head_dim];
        for bh in 0..batch_heads {
            for v_idx in 0..v_head_dim {
                let mut state = vec![0.0f32; k_head_dim];
                for k_idx in 0..k_head_dim {
                    state[k_idx] =
                        prev_state_data[bh * k_head_dim * v_head_dim + k_idx * v_head_dim + v_idx];
                }
                let packed_base = bh * chunk_size * packed_width;
                let value_base = bh * chunk_size * v_head_dim;
                let out_base = bh * (2 * chunk_size + k_head_dim) * v_head_dim;
                let mut v_new = vec![0.0f32; chunk_size];
                let mut attn_inter = vec![0.0f32; chunk_size];
                for t in 0..chunk_size {
                    let packed_row = packed_base + t * packed_width;
                    let mut v_prime = 0.0f32;
                    let mut attn = 0.0f32;
                    for k_idx in 0..k_head_dim {
                        v_prime +=
                            packed_chunk_data[packed_row + k_head_dim + k_idx] * state[k_idx];
                        attn +=
                            packed_chunk_data[packed_row + 2 * k_head_dim + k_idx] * state[k_idx];
                    }
                    v_new[t] = value_data[value_base + t * v_head_dim + v_idx] - v_prime;
                    attn_inter[t] = attn;
                    expected[out_base + t * v_head_dim + v_idx] = v_new[t];
                    expected[out_base + (chunk_size + t) * v_head_dim + v_idx] = attn_inter[t];
                }
                let state_decay = packed_chunk_data[packed_base + 3 * k_head_dim];
                for k_idx in 0..k_head_dim {
                    let mut update = 0.0f32;
                    for t in 0..chunk_size {
                        let packed_row = packed_base + t * packed_width;
                        update += packed_chunk_data[packed_row + k_idx] * v_new[t];
                    }
                    expected[out_base + (2 * chunk_size + k_idx) * v_head_dim + v_idx] =
                        state_decay * state[k_idx] + update;
                }
            }
        }

        assert_close(&output, &expected, 1e-5);
        Ok(())
    }

    #[cfg(feature = "hip")]
    #[test]
    fn hip_delta_full_scan_matches_reference() -> Result<()> {
        let device = Device::new_hip(0)?;
        let batch_heads = 1usize;
        let num_chunks = 2usize;
        let chunk_size = 2usize;
        let k_head_dim = 2usize;
        let v_head_dim = 2usize;

        let initial_state_data = vec![0.15f32, -0.05, 0.2, 0.1];
        let weighted_key_data = vec![0.2f32, -0.1, 0.05, 0.3, -0.2, 0.15, 0.25, -0.05];
        let k_cumdecay_data = vec![0.1f32, 0.25, -0.2, 0.05, 0.15, -0.1, 0.05, 0.2];
        let q_state_data = vec![0.05f32, -0.15, 0.2, 0.1, -0.1, 0.3, 0.15, -0.05];
        let local_attn_data = vec![0.2f32, 0.1, -0.1, 0.3, 0.05, -0.2, 0.25, 0.15];
        let state_decay_data = vec![0.85f32, 0.9];
        let value_data = vec![0.3f32, 0.1, -0.2, 0.4, 0.05, -0.1, 0.2, 0.35];

        let initial_state = Tensor::from_vec(
            initial_state_data.clone(),
            (batch_heads, k_head_dim, v_head_dim),
            &device,
        )?;
        let weighted_key_scan = Tensor::from_vec(
            weighted_key_data.clone(),
            (batch_heads, num_chunks, chunk_size, k_head_dim),
            &device,
        )?;
        let k_cumdecay_scan = Tensor::from_vec(
            k_cumdecay_data.clone(),
            (batch_heads, num_chunks, chunk_size, k_head_dim),
            &device,
        )?;
        let q_state_scan = Tensor::from_vec(
            q_state_data.clone(),
            (batch_heads, num_chunks, chunk_size, k_head_dim),
            &device,
        )?;
        let local_attn_scan = Tensor::from_vec(
            local_attn_data.clone(),
            (batch_heads, num_chunks, chunk_size, chunk_size),
            &device,
        )?;
        let state_decay_scan =
            Tensor::from_vec(state_decay_data.clone(), (batch_heads, num_chunks), &device)?;
        let value = Tensor::from_vec(
            value_data.clone(),
            (batch_heads, num_chunks, chunk_size, v_head_dim),
            &device,
        )?;

        let output = delta_full_scan(
            &initial_state,
            &weighted_key_scan,
            &k_cumdecay_scan,
            &q_state_scan,
            &local_attn_scan,
            &state_decay_scan,
            &value,
        )?;
        let output = output.flatten_all()?.to_vec1::<f32>()?;

        let token_count = num_chunks * chunk_size;
        let mut expected = vec![0.0f32; batch_heads * (token_count + k_head_dim) * v_head_dim];
        for bh in 0..batch_heads {
            for v_idx in 0..v_head_dim {
                let mut state = vec![0.0f32; k_head_dim];
                for k_idx in 0..k_head_dim {
                    state[k_idx] = initial_state_data
                        [bh * k_head_dim * v_head_dim + k_idx * v_head_dim + v_idx];
                }
                let scan_base = bh * num_chunks * chunk_size * k_head_dim;
                let local_base = bh * num_chunks * chunk_size * chunk_size;
                let decay_base = bh * num_chunks;
                let value_base = bh * token_count * v_head_dim;
                let out_base = bh * (token_count + k_head_dim) * v_head_dim;
                let mut v_new = vec![0.0f32; chunk_size];
                let mut attn_inter = vec![0.0f32; chunk_size];
                for chunk in 0..num_chunks {
                    let chunk_scan = scan_base + chunk * chunk_size * k_head_dim;
                    let chunk_local = local_base + chunk * chunk_size * chunk_size;
                    let chunk_value = value_base + chunk * chunk_size * v_head_dim;
                    for t in 0..chunk_size {
                        let row = chunk_scan + t * k_head_dim;
                        let mut v_prime = 0.0f32;
                        let mut attn = 0.0f32;
                        for k_idx in 0..k_head_dim {
                            v_prime += k_cumdecay_data[row + k_idx] * state[k_idx];
                            attn += q_state_data[row + k_idx] * state[k_idx];
                        }
                        v_new[t] = value_data[chunk_value + t * v_head_dim + v_idx] - v_prime;
                        attn_inter[t] = attn;
                    }
                    for t in 0..chunk_size {
                        let row = chunk_local + t * chunk_size;
                        let mut local = 0.0f32;
                        for s in 0..chunk_size {
                            local += local_attn_data[row + s] * v_new[s];
                        }
                        expected[out_base + (chunk * chunk_size + t) * v_head_dim + v_idx] =
                            attn_inter[t] + local;
                    }
                    let state_decay = state_decay_data[decay_base + chunk];
                    for k_idx in 0..k_head_dim {
                        let mut update = 0.0f32;
                        for t in 0..chunk_size {
                            let row = chunk_scan + t * k_head_dim;
                            update += weighted_key_data[row + k_idx] * v_new[t];
                        }
                        state[k_idx] = state_decay * state[k_idx] + update;
                    }
                }
                let state_out = out_base + token_count * v_head_dim;
                for k_idx in 0..k_head_dim {
                    expected[state_out + k_idx * v_head_dim + v_idx] = state[k_idx];
                }
            }
        }

        assert_close(&output, &expected, 1e-5);
        Ok(())
    }

    #[test]
    fn parses_nested_text_config() {
        let config: Config = serde_json::from_str(
            r#"{
                "text_config": {
                    "vocab_size": 16,
                    "hidden_size": 32,
                    "intermediate_size": 64,
                    "num_hidden_layers": 4,
                    "num_attention_heads": 4,
                    "num_key_value_heads": 2,
                    "hidden_act": "silu",
                    "max_position_embeddings": 128,
                    "rms_norm_eps": 1e-6,
                    "head_dim": 8,
                    "linear_conv_kernel_dim": 4,
                    "linear_key_head_dim": 4,
                    "linear_value_head_dim": 4,
                    "linear_num_key_heads": 2,
                    "linear_num_value_heads": 4,
                    "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"],
                    "rope_parameters": {
                        "rope_type": "default",
                        "rope_theta": 1000000.0,
                        "partial_rotary_factor": 0.25
                    }
                }
            }"#,
        )
        .unwrap();
        let config = config.normalized();
        assert_eq!(config.text_config.layer_types.len(), 4);
        assert_eq!(config.text_config.layer_types[3], "full_attention");
        assert_eq!(config.text_config.rope_theta(), 1_000_000.0);
        assert_eq!(config.text_config.partial_rotary_factor(), 0.25);
    }

    #[test]
    fn normalized_config_supplies_hybrid_layer_pattern() {
        let cfg = Config {
            text_config: TextConfig {
                vocab_size: 16,
                hidden_size: 32,
                intermediate_size: 64,
                num_hidden_layers: 8,
                num_attention_heads: 4,
                num_key_value_heads: 2,
                hidden_act: candle_nn::Activation::Silu,
                max_position_embeddings: 128,
                rms_norm_eps: 1e-6,
                tie_word_embeddings: false,
                attention_bias: false,
                attention_dropout: 0.0,
                head_dim: 8,
                linear_conv_kernel_dim: 4,
                linear_key_head_dim: 8,
                linear_value_head_dim: 8,
                linear_num_key_heads: 2,
                linear_num_value_heads: 4,
                layer_types: Vec::new(),
                rope_parameters: None,
            },
        }
        .normalized();

        assert_eq!(
            cfg.text_config.layer_types,
            vec![
                "linear_attention",
                "linear_attention",
                "linear_attention",
                "full_attention",
                "linear_attention",
                "linear_attention",
                "linear_attention",
                "full_attention",
            ]
        );
    }

    #[test]
    fn metal_chunk_size_scales_with_sequence_length() {
        assert_eq!(recommended_metal_linear_chunk_size(512), 16);
        assert_eq!(recommended_metal_linear_chunk_size(2048), 24);
        assert_eq!(recommended_metal_linear_chunk_size(8192), 24);
    }

    #[test]
    fn parses_delta_scan_modes() {
        assert_eq!(
            parse_delta_net_scan_mode("flat3d"),
            Some(DeltaNetScanMode::Flat3d)
        );
        assert_eq!(
            parse_delta_net_scan_mode("hoisted-decays"),
            Some(DeltaNetScanMode::HoistedDecays)
        );
        assert_eq!(
            parse_delta_net_scan_mode("prebatched-local"),
            Some(DeltaNetScanMode::PrebatchedLocal)
        );
        assert_eq!(
            parse_delta_net_scan_mode("torch-like"),
            Some(DeltaNetScanMode::TorchLike)
        );
        assert_eq!(parse_delta_net_scan_mode("unknown"), None);
    }

    #[test]
    fn recommended_delta_scan_mode_uses_prebatched_local_for_long_metal_contexts() {
        let device = Device::new_metal(0).unwrap_or_else(|_| Device::Cpu);
        let short = recommended_delta_net_execution_policy(&device, 512, 32);
        let long = recommended_delta_net_execution_policy(&device, 2048, 86);
        match device.location() {
            DeviceLocation::Metal { .. } => {
                assert_eq!(short.scan_mode, DeltaNetScanMode::Flat3d);
                assert!(!short.use_flattened_solve);
                assert_eq!(long.scan_mode, DeltaNetScanMode::PrebatchedLocal);
                assert!(long.use_flattened_solve);
            }
            _ => {
                assert_eq!(short.scan_mode, DeltaNetScanMode::Flat3d);
                assert!(!short.use_flattened_solve);
                assert_eq!(long.scan_mode, DeltaNetScanMode::Flat3d);
                assert!(!long.use_flattened_solve);
            }
        }
    }
}
