use crate::utils::{BufferOffset, EncoderProvider};
use crate::{linear_split, set_params, Buffer, ComputeCommandEncoder, Device, Kernels};
use crate::{DType, MetalKernelError, Source};
use objc2_metal::MTLResourceUsage;
use objc2_metal::MTLSize;

#[allow(clippy::too_many_arguments)]
pub fn call_linear_prefill_conv_pack(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    dtype: DType,
    batch_size: usize,
    conv_dim: usize,
    total_len: usize,
    seq_len: usize,
    kernel_size: usize,
    mixed_qkv: BufferOffset,
    weights: BufferOffset,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    let kernel_name = match dtype {
        DType::F16 => "linear_prefill_conv_pack_f16",
        DType::F32 => "linear_prefill_conv_pack_f32",
        DType::BF16 => "linear_prefill_conv_pack_bf16",
        other => {
            return Err(MetalKernelError::UnsupportedDTypeForOp(
                match other {
                    DType::U8 => "u8",
                    DType::U32 => "u32",
                    DType::I64 => "i64",
                    DType::BF16 => "bf16",
                    DType::F16 => "f16",
                    DType::F32 => "f32",
                },
                "linear_prefill_conv_pack",
            ));
        }
    };
    let pipeline = kernels.load_pipeline(device, Source::Delta, kernel_name)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    let threads = batch_size * seq_len * conv_dim;
    set_params!(
        encoder,
        (
            batch_size,
            conv_dim,
            total_len,
            seq_len,
            kernel_size,
            &mixed_qkv,
            &weights,
            output
        )
    );
    let (thread_group_count, thread_group_size) = linear_split(&pipeline, threads);
    encoder.use_resource(mixed_qkv.buffer, MTLResourceUsage::Read);
    encoder.use_resource(weights.buffer, MTLResourceUsage::Read);
    encoder.use_resource(output, MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_full_attention_prefill(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    dtype: DType,
    batch_size: usize,
    q_heads: usize,
    kv_heads: usize,
    q_len: usize,
    kv_len: usize,
    head_dim: usize,
    num_kv_groups: usize,
    scale: f32,
    seqlen_offset: usize,
    query: BufferOffset,
    key: BufferOffset,
    value: BufferOffset,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    let kernel_name = match dtype {
        DType::F16 => "full_attention_prefill_f16",
        DType::F32 => "full_attention_prefill_f32",
        DType::BF16 => "full_attention_prefill_bf16",
        other => {
            return Err(MetalKernelError::UnsupportedDTypeForOp(
                match other {
                    DType::U8 => "u8",
                    DType::U32 => "u32",
                    DType::I64 => "i64",
                    DType::BF16 => "bf16",
                    DType::F16 => "f16",
                    DType::F32 => "f32",
                },
                "full_attention_prefill",
            ));
        }
    };
    let pipeline = kernels.load_pipeline(device, Source::Delta, kernel_name)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    let threads = batch_size * q_heads * q_len;
    set_params!(
        encoder,
        (
            batch_size,
            q_heads,
            kv_heads,
            q_len,
            kv_len,
            head_dim,
            num_kv_groups,
            scale,
            seqlen_offset,
            &query,
            &key,
            &value,
            output
        )
    );
    let (thread_group_count, thread_group_size) = linear_split(&pipeline, threads);
    encoder.use_resource(query.buffer, MTLResourceUsage::Read);
    encoder.use_resource(key.buffer, MTLResourceUsage::Read);
    encoder.use_resource(value.buffer, MTLResourceUsage::Read);
    encoder.use_resource(output, MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_delta_recurrent_prefill(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    dtype: DType,
    batch_heads: usize,
    seq_len: usize,
    k_head_dim: usize,
    v_head_dim: usize,
    initial_state: BufferOffset,
    query: BufferOffset,
    key: BufferOffset,
    value: BufferOffset,
    beta: BufferOffset,
    g: BufferOffset,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    let kernel_name = match dtype {
        DType::F16 => "delta_recurrent_prefill_f16",
        DType::F32 => "delta_recurrent_prefill_f32",
        DType::BF16 => "delta_recurrent_prefill_bf16",
        other => {
            return Err(MetalKernelError::UnsupportedDTypeForOp(
                match other {
                    DType::U8 => "u8",
                    DType::U32 => "u32",
                    DType::I64 => "i64",
                    DType::BF16 => "bf16",
                    DType::F16 => "f16",
                    DType::F32 => "f32",
                },
                "delta_recurrent_prefill",
            ));
        }
    };
    let pipeline = kernels.load_pipeline(device, Source::Delta, kernel_name)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    let threads = batch_heads * v_head_dim;
    set_params!(
        encoder,
        (
            batch_heads,
            seq_len,
            k_head_dim,
            v_head_dim,
            &initial_state,
            &query,
            &key,
            &value,
            &beta,
            &g,
            output
        )
    );
    let (thread_group_count, thread_group_size) = linear_split(&pipeline, threads);
    encoder.use_resource(initial_state.buffer, MTLResourceUsage::Read);
    encoder.use_resource(query.buffer, MTLResourceUsage::Read);
    encoder.use_resource(key.buffer, MTLResourceUsage::Read);
    encoder.use_resource(value.buffer, MTLResourceUsage::Read);
    encoder.use_resource(beta.buffer, MTLResourceUsage::Read);
    encoder.use_resource(g.buffer, MTLResourceUsage::Read);
    encoder.use_resource(output, MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_delta_chunk_step(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    dtype: DType,
    batch_heads: usize,
    chunk_size: usize,
    k_head_dim: usize,
    v_head_dim: usize,
    prev_state: BufferOffset,
    query: BufferOffset,
    key: BufferOffset,
    value: BufferOffset,
    beta: BufferOffset,
    g: BufferOffset,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    let kernel_name = match dtype {
        DType::F16 => "delta_chunk_step_f16",
        DType::F32 => "delta_chunk_step_f32",
        DType::BF16 => "delta_chunk_step_bf16",
        other => {
            return Err(MetalKernelError::UnsupportedDTypeForOp(
                match other {
                    DType::U8 => "u8",
                    DType::U32 => "u32",
                    DType::I64 => "i64",
                    DType::BF16 => "bf16",
                    DType::F16 => "f16",
                    DType::F32 => "f32",
                },
                "delta_chunk_step",
            ));
        }
    };
    let pipeline = kernels.load_pipeline(device, Source::Delta, kernel_name)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    set_params!(
        encoder,
        (
            batch_heads,
            chunk_size,
            k_head_dim,
            v_head_dim,
            &prev_state,
            &query,
            &key,
            &value,
            &beta,
            &g,
            output
        )
    );

    let group_width = std::cmp::min(
        pipeline.max_total_threads_per_threadgroup(),
        v_head_dim.next_power_of_two(),
    );
    let thread_group_count = MTLSize {
        width: batch_heads,
        height: 1,
        depth: 1,
    };
    let thread_group_size = MTLSize {
        width: group_width,
        height: 1,
        depth: 1,
    };

    encoder.use_resource(prev_state.buffer, MTLResourceUsage::Read);
    encoder.use_resource(query.buffer, MTLResourceUsage::Read);
    encoder.use_resource(key.buffer, MTLResourceUsage::Read);
    encoder.use_resource(value.buffer, MTLResourceUsage::Read);
    encoder.use_resource(beta.buffer, MTLResourceUsage::Read);
    encoder.use_resource(g.buffer, MTLResourceUsage::Read);
    encoder.use_resource(output, MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_delta_chunk_step_windowed(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    dtype: DType,
    batch_heads: usize,
    chunk_size: usize,
    k_head_dim: usize,
    v_head_dim: usize,
    prev_state_bh_stride: usize,
    query_bh_stride: usize,
    value_bh_stride: usize,
    scalar_bh_stride: usize,
    output_total_rows: usize,
    output_token_row_offset: usize,
    output_state_row_offset: usize,
    prev_state: BufferOffset,
    query: BufferOffset,
    key: BufferOffset,
    value: BufferOffset,
    beta: BufferOffset,
    g: BufferOffset,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    let kernel_name = match dtype {
        DType::F16 => "delta_chunk_step_windowed_f16",
        DType::F32 => "delta_chunk_step_windowed_f32",
        DType::BF16 => "delta_chunk_step_windowed_bf16",
        other => {
            return Err(MetalKernelError::UnsupportedDTypeForOp(
                match other {
                    DType::U8 => "u8",
                    DType::U32 => "u32",
                    DType::I64 => "i64",
                    DType::BF16 => "bf16",
                    DType::F16 => "f16",
                    DType::F32 => "f32",
                },
                "delta_chunk_step_windowed",
            ));
        }
    };
    let pipeline = kernels.load_pipeline(device, Source::Delta, kernel_name)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    set_params!(
        encoder,
        (
            batch_heads,
            chunk_size,
            k_head_dim,
            v_head_dim,
            prev_state_bh_stride,
            query_bh_stride,
            value_bh_stride,
            scalar_bh_stride,
            output_total_rows,
            output_token_row_offset,
            output_state_row_offset,
            &prev_state,
            &query,
            &key,
            &value,
            &beta,
            &g,
            output
        )
    );

    let group_width = std::cmp::min(
        pipeline.max_total_threads_per_threadgroup(),
        v_head_dim.next_power_of_two(),
    );
    let thread_group_count = MTLSize {
        width: batch_heads,
        height: 1,
        depth: 1,
    };
    let thread_group_size = MTLSize {
        width: group_width,
        height: 1,
        depth: 1,
    };

    encoder.use_resource(prev_state.buffer, MTLResourceUsage::Read);
    encoder.use_resource(query.buffer, MTLResourceUsage::Read);
    encoder.use_resource(key.buffer, MTLResourceUsage::Read);
    encoder.use_resource(value.buffer, MTLResourceUsage::Read);
    encoder.use_resource(beta.buffer, MTLResourceUsage::Read);
    encoder.use_resource(g.buffer, MTLResourceUsage::Read);
    encoder.use_resource(output, MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_delta_chunk_step_windowed_2d(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    dtype: DType,
    batch_heads: usize,
    chunk_size: usize,
    k_head_dim: usize,
    v_head_dim: usize,
    prev_state_bh_stride: usize,
    query_bh_stride: usize,
    value_bh_stride: usize,
    scalar_bh_stride: usize,
    output_total_rows: usize,
    output_token_row_offset: usize,
    output_state_row_offset: usize,
    prev_state: BufferOffset,
    query: BufferOffset,
    key: BufferOffset,
    value: BufferOffset,
    beta: BufferOffset,
    g: BufferOffset,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    let kernel_name = match dtype {
        DType::F16 => "delta_chunk_step_windowed_2d_f16",
        DType::F32 => "delta_chunk_step_windowed_2d_f32",
        DType::BF16 => "delta_chunk_step_windowed_2d_bf16",
        other => {
            return Err(MetalKernelError::UnsupportedDTypeForOp(
                match other {
                    DType::U8 => "u8",
                    DType::U32 => "u32",
                    DType::I64 => "i64",
                    DType::BF16 => "bf16",
                    DType::F16 => "f16",
                    DType::F32 => "f32",
                },
                "delta_chunk_step_windowed_2d",
            ));
        }
    };
    let pipeline = kernels.load_pipeline(device, Source::Delta, kernel_name)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    set_params!(
        encoder,
        (
            batch_heads,
            chunk_size,
            k_head_dim,
            v_head_dim,
            prev_state_bh_stride,
            query_bh_stride,
            value_bh_stride,
            scalar_bh_stride,
            output_total_rows,
            output_token_row_offset,
            output_state_row_offset,
            &prev_state,
            &query,
            &key,
            &value,
            &beta,
            &g,
            output
        )
    );

    let group_width = std::cmp::min(24usize, v_head_dim).max(1);
    let group_height = std::cmp::min(
        2usize,
        std::cmp::max(
            1,
            pipeline.max_total_threads_per_threadgroup() / group_width,
        ),
    );
    let v_tiles = std::cmp::max(1, v_head_dim.div_ceil(group_width));
    let thread_group_count = MTLSize {
        width: batch_heads,
        height: v_tiles,
        depth: 1,
    };
    let thread_group_size = MTLSize {
        width: group_width,
        height: group_height,
        depth: 1,
    };

    encoder.use_resource(prev_state.buffer, MTLResourceUsage::Read);
    encoder.use_resource(query.buffer, MTLResourceUsage::Read);
    encoder.use_resource(key.buffer, MTLResourceUsage::Read);
    encoder.use_resource(value.buffer, MTLResourceUsage::Read);
    encoder.use_resource(beta.buffer, MTLResourceUsage::Read);
    encoder.use_resource(g.buffer, MTLResourceUsage::Read);
    encoder.use_resource(output, MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_delta_chunk_readout(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    dtype: DType,
    batch_heads: usize,
    chunk_size: usize,
    k_head_dim: usize,
    v_head_dim: usize,
    prev_state: BufferOffset,
    query: BufferOffset,
    key: BufferOffset,
    value: BufferOffset,
    beta: BufferOffset,
    g: BufferOffset,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    let kernel_name = match dtype {
        DType::F16 => "delta_chunk_readout_f16",
        DType::F32 => "delta_chunk_readout_f32",
        DType::BF16 => "delta_chunk_readout_bf16",
        other => {
            return Err(MetalKernelError::UnsupportedDTypeForOp(
                match other {
                    DType::U8 => "u8",
                    DType::U32 => "u32",
                    DType::I64 => "i64",
                    DType::BF16 => "bf16",
                    DType::F16 => "f16",
                    DType::F32 => "f32",
                },
                "delta_chunk_readout",
            ));
        }
    };
    let pipeline = kernels.load_pipeline(device, Source::Delta, kernel_name)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    set_params!(
        encoder,
        (
            batch_heads,
            chunk_size,
            k_head_dim,
            v_head_dim,
            &prev_state,
            &query,
            &key,
            &value,
            &beta,
            &g,
            output
        )
    );

    let group_width = std::cmp::min(
        pipeline.max_total_threads_per_threadgroup(),
        v_head_dim.next_power_of_two(),
    );
    let thread_group_count = MTLSize {
        width: batch_heads,
        height: 1,
        depth: 1,
    };
    let thread_group_size = MTLSize {
        width: group_width,
        height: 1,
        depth: 1,
    };

    encoder.use_resource(prev_state.buffer, MTLResourceUsage::Read);
    encoder.use_resource(query.buffer, MTLResourceUsage::Read);
    encoder.use_resource(key.buffer, MTLResourceUsage::Read);
    encoder.use_resource(value.buffer, MTLResourceUsage::Read);
    encoder.use_resource(beta.buffer, MTLResourceUsage::Read);
    encoder.use_resource(g.buffer, MTLResourceUsage::Read);
    encoder.use_resource(output, MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_delta_chunk_readout_split(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    dtype: DType,
    batch_heads: usize,
    chunk_size: usize,
    k_head_dim: usize,
    v_head_dim: usize,
    prev_state: BufferOffset,
    query: BufferOffset,
    key: BufferOffset,
    value: BufferOffset,
    beta: BufferOffset,
    g: BufferOffset,
    output: &Buffer,
    v_new_output: &Buffer,
) -> Result<(), MetalKernelError> {
    let kernel_name = match dtype {
        DType::F16 => "delta_chunk_readout_split_f16",
        DType::F32 => "delta_chunk_readout_split_f32",
        DType::BF16 => "delta_chunk_readout_split_bf16",
        other => {
            return Err(MetalKernelError::UnsupportedDTypeForOp(
                match other {
                    DType::U8 => "u8",
                    DType::U32 => "u32",
                    DType::I64 => "i64",
                    DType::BF16 => "bf16",
                    DType::F16 => "f16",
                    DType::F32 => "f32",
                },
                "delta_chunk_readout_split",
            ));
        }
    };
    let pipeline = kernels.load_pipeline(device, Source::Delta, kernel_name)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    set_params!(
        encoder,
        (
            batch_heads,
            chunk_size,
            k_head_dim,
            v_head_dim,
            &prev_state,
            &query,
            &key,
            &value,
            &beta,
            &g,
            output,
            v_new_output
        )
    );

    let group_width = std::cmp::min(
        pipeline.max_total_threads_per_threadgroup(),
        v_head_dim.next_power_of_two(),
    );
    let thread_group_count = MTLSize {
        width: batch_heads,
        height: 1,
        depth: 1,
    };
    let thread_group_size = MTLSize {
        width: group_width,
        height: 1,
        depth: 1,
    };

    encoder.use_resource(prev_state.buffer, MTLResourceUsage::Read);
    encoder.use_resource(query.buffer, MTLResourceUsage::Read);
    encoder.use_resource(key.buffer, MTLResourceUsage::Read);
    encoder.use_resource(value.buffer, MTLResourceUsage::Read);
    encoder.use_resource(beta.buffer, MTLResourceUsage::Read);
    encoder.use_resource(g.buffer, MTLResourceUsage::Read);
    encoder.use_resource(output, MTLResourceUsage::Write);
    encoder.use_resource(v_new_output, MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_delta_chunk_state_update_raw(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    dtype: DType,
    batch_heads: usize,
    chunk_size: usize,
    k_head_dim: usize,
    v_head_dim: usize,
    prev_state: BufferOffset,
    key: BufferOffset,
    v_new: BufferOffset,
    g: BufferOffset,
    output_row_offset: usize,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    let kernel_name = match dtype {
        DType::F16 => "delta_chunk_state_update_raw_f16",
        DType::F32 => "delta_chunk_state_update_raw_f32",
        DType::BF16 => "delta_chunk_state_update_raw_bf16",
        other => {
            return Err(MetalKernelError::UnsupportedDTypeForOp(
                match other {
                    DType::U8 => "u8",
                    DType::U32 => "u32",
                    DType::I64 => "i64",
                    DType::BF16 => "bf16",
                    DType::F16 => "f16",
                    DType::F32 => "f32",
                },
                "delta_chunk_state_update_raw",
            ));
        }
    };
    let pipeline = kernels.load_pipeline(device, Source::Delta, kernel_name)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    let threads = batch_heads * k_head_dim * v_head_dim;
    set_params!(
        encoder,
        (
            batch_heads,
            chunk_size,
            k_head_dim,
            v_head_dim,
            &prev_state,
            &key,
            &v_new,
            &g,
            output_row_offset,
            output
        )
    );
    let (thread_group_count, thread_group_size) = linear_split(&pipeline, threads);
    encoder.use_resource(prev_state.buffer, MTLResourceUsage::Read);
    encoder.use_resource(key.buffer, MTLResourceUsage::Read);
    encoder.use_resource(v_new.buffer, MTLResourceUsage::Read);
    encoder.use_resource(g.buffer, MTLResourceUsage::Read);
    encoder.use_resource(output, MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_delta_chunk_step_2d(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    dtype: DType,
    batch_heads: usize,
    chunk_size: usize,
    k_head_dim: usize,
    v_head_dim: usize,
    prev_state: BufferOffset,
    query: BufferOffset,
    key: BufferOffset,
    value: BufferOffset,
    beta: BufferOffset,
    g: BufferOffset,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    let kernel_name = match dtype {
        DType::F16 => "delta_chunk_step_2d_f16",
        DType::F32 => "delta_chunk_step_2d_f32",
        DType::BF16 => "delta_chunk_step_2d_bf16",
        other => {
            return Err(MetalKernelError::UnsupportedDTypeForOp(
                match other {
                    DType::U8 => "u8",
                    DType::U32 => "u32",
                    DType::I64 => "i64",
                    DType::BF16 => "bf16",
                    DType::F16 => "f16",
                    DType::F32 => "f32",
                },
                "delta_chunk_step_2d",
            ));
        }
    };
    let pipeline = kernels.load_pipeline(device, Source::Delta, kernel_name)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    set_params!(
        encoder,
        (
            batch_heads,
            chunk_size,
            k_head_dim,
            v_head_dim,
            &prev_state,
            &query,
            &key,
            &value,
            &beta,
            &g,
            output
        )
    );

    let group_width = std::cmp::min(
        pipeline.max_total_threads_per_threadgroup() / 4,
        v_head_dim.next_power_of_two(),
    )
    .max(1);
    let group_height = std::cmp::min(
        4usize,
        std::cmp::max(
            1,
            pipeline.max_total_threads_per_threadgroup() / group_width,
        ),
    );
    let thread_group_count = MTLSize {
        width: batch_heads,
        height: 1,
        depth: 1,
    };
    let thread_group_size = MTLSize {
        width: group_width,
        height: group_height,
        depth: 1,
    };

    encoder.use_resource(prev_state.buffer, MTLResourceUsage::Read);
    encoder.use_resource(query.buffer, MTLResourceUsage::Read);
    encoder.use_resource(key.buffer, MTLResourceUsage::Read);
    encoder.use_resource(value.buffer, MTLResourceUsage::Read);
    encoder.use_resource(beta.buffer, MTLResourceUsage::Read);
    encoder.use_resource(g.buffer, MTLResourceUsage::Read);
    encoder.use_resource(output, MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_delta_chunk_scan_raw(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    dtype: DType,
    batch_heads: usize,
    num_chunks: usize,
    chunk_size: usize,
    k_head_dim: usize,
    v_head_dim: usize,
    initial_state: BufferOffset,
    query: BufferOffset,
    key: BufferOffset,
    value: BufferOffset,
    beta: BufferOffset,
    g: BufferOffset,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    let kernel_name = match dtype {
        DType::F16 => "delta_chunk_scan_raw_f16",
        DType::F32 => "delta_chunk_scan_raw_f32",
        DType::BF16 => "delta_chunk_scan_raw_bf16",
        other => {
            return Err(MetalKernelError::UnsupportedDTypeForOp(
                match other {
                    DType::U8 => "u8",
                    DType::U32 => "u32",
                    DType::I64 => "i64",
                    DType::BF16 => "bf16",
                    DType::F16 => "f16",
                    DType::F32 => "f32",
                },
                "delta_chunk_scan_raw",
            ));
        }
    };
    let pipeline = kernels.load_pipeline(device, Source::Delta, kernel_name)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    set_params!(
        encoder,
        (
            batch_heads,
            num_chunks,
            chunk_size,
            k_head_dim,
            v_head_dim,
            &initial_state,
            &query,
            &key,
            &value,
            &beta,
            &g,
            output
        )
    );

    let group_width = std::cmp::min(
        pipeline.max_total_threads_per_threadgroup(),
        v_head_dim.next_power_of_two(),
    );
    let thread_group_count = MTLSize {
        width: batch_heads,
        height: 1,
        depth: 1,
    };
    let thread_group_size = MTLSize {
        width: group_width,
        height: 1,
        depth: 1,
    };

    encoder.use_resource(initial_state.buffer, MTLResourceUsage::Read);
    encoder.use_resource(query.buffer, MTLResourceUsage::Read);
    encoder.use_resource(key.buffer, MTLResourceUsage::Read);
    encoder.use_resource(value.buffer, MTLResourceUsage::Read);
    encoder.use_resource(beta.buffer, MTLResourceUsage::Read);
    encoder.use_resource(g.buffer, MTLResourceUsage::Read);
    encoder.use_resource(output, MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_delta_state_update(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    dtype: DType,
    batch_heads: usize,
    chunk_size: usize,
    k_head_dim: usize,
    v_head_dim: usize,
    prev_state: BufferOffset,
    weighted_key: BufferOffset,
    value: BufferOffset,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    let kernel_name = match dtype {
        DType::F16 => "delta_state_update_f16",
        DType::F32 => "delta_state_update_f32",
        DType::BF16 => "delta_state_update_bf16",
        other => {
            return Err(MetalKernelError::UnsupportedDTypeForOp(
                match other {
                    DType::U8 => "u8",
                    DType::U32 => "u32",
                    DType::I64 => "i64",
                    DType::BF16 => "bf16",
                    DType::F16 => "f16",
                    DType::F32 => "f32",
                },
                "delta_state_update",
            ));
        }
    };
    let pipeline = kernels.load_pipeline(device, Source::Delta, kernel_name)?;

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    let output_elems = batch_heads * k_head_dim * v_head_dim;
    set_params!(
        encoder,
        (
            batch_heads,
            chunk_size,
            k_head_dim,
            v_head_dim,
            &prev_state,
            &weighted_key,
            &value,
            output
        )
    );

    let (thread_group_count, thread_group_size) = linear_split(&pipeline, output_elems);
    encoder.use_resource(prev_state.buffer, MTLResourceUsage::Read);
    encoder.use_resource(weighted_key.buffer, MTLResourceUsage::Read);
    encoder.use_resource(value.buffer, MTLResourceUsage::Read);
    encoder.use_resource(output, MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_delta_state_scan(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    dtype: DType,
    batch_heads: usize,
    num_chunks: usize,
    chunk_size: usize,
    k_head_dim: usize,
    v_head_dim: usize,
    initial_state: BufferOffset,
    packed_scan: BufferOffset,
    value: BufferOffset,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    let kernel_name = match dtype {
        DType::F16 => "delta_state_scan_f16",
        DType::F32 => "delta_state_scan_f32",
        DType::BF16 => "delta_state_scan_bf16",
        other => {
            return Err(MetalKernelError::UnsupportedDTypeForOp(
                match other {
                    DType::U8 => "u8",
                    DType::U32 => "u32",
                    DType::I64 => "i64",
                    DType::BF16 => "bf16",
                    DType::F16 => "f16",
                    DType::F32 => "f32",
                },
                "delta_state_scan",
            ));
        }
    };
    let pipeline = kernels.load_pipeline(device, Source::Delta, kernel_name)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    let threads = batch_heads * v_head_dim;
    set_params!(
        encoder,
        (
            batch_heads,
            num_chunks,
            chunk_size,
            k_head_dim,
            v_head_dim,
            &initial_state,
            &packed_scan,
            &value,
            output
        )
    );
    let (thread_group_count, thread_group_size) = linear_split(&pipeline, threads);
    encoder.use_resource(initial_state.buffer, MTLResourceUsage::Read);
    encoder.use_resource(packed_scan.buffer, MTLResourceUsage::Read);
    encoder.use_resource(value.buffer, MTLResourceUsage::Read);
    encoder.use_resource(output, MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_delta_chunk_fused(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    dtype: DType,
    batch_heads: usize,
    chunk_size: usize,
    k_head_dim: usize,
    v_head_dim: usize,
    prev_state: BufferOffset,
    packed_chunk: BufferOffset,
    value: BufferOffset,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    let kernel_name = match dtype {
        DType::F16 => "delta_chunk_fused_f16",
        DType::F32 => "delta_chunk_fused_f32",
        DType::BF16 => "delta_chunk_fused_bf16",
        other => {
            return Err(MetalKernelError::UnsupportedDTypeForOp(
                match other {
                    DType::U8 => "u8",
                    DType::U32 => "u32",
                    DType::I64 => "i64",
                    DType::BF16 => "bf16",
                    DType::F16 => "f16",
                    DType::F32 => "f32",
                },
                "delta_chunk_fused",
            ));
        }
    };
    let pipeline = kernels.load_pipeline(device, Source::Delta, kernel_name)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    let threads = batch_heads * v_head_dim;
    set_params!(
        encoder,
        (
            batch_heads,
            chunk_size,
            k_head_dim,
            v_head_dim,
            &prev_state,
            &packed_chunk,
            &value,
            output
        )
    );
    let (thread_group_count, thread_group_size) = linear_split(&pipeline, threads);
    encoder.use_resource(prev_state.buffer, MTLResourceUsage::Read);
    encoder.use_resource(packed_chunk.buffer, MTLResourceUsage::Read);
    encoder.use_resource(value.buffer, MTLResourceUsage::Read);
    encoder.use_resource(output, MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_delta_full_scan(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    dtype: DType,
    batch_heads: usize,
    num_chunks: usize,
    chunk_size: usize,
    k_head_dim: usize,
    v_head_dim: usize,
    initial_state: BufferOffset,
    weighted_key_scan: BufferOffset,
    k_cumdecay_scan: BufferOffset,
    q_state_scan: BufferOffset,
    local_attn_scan: BufferOffset,
    state_decay_scan: BufferOffset,
    value: BufferOffset,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    let kernel_name = match dtype {
        DType::F16 => "delta_full_scan_f16",
        DType::F32 => "delta_full_scan_f32",
        DType::BF16 => "delta_full_scan_bf16",
        other => {
            return Err(MetalKernelError::UnsupportedDTypeForOp(
                match other {
                    DType::U8 => "u8",
                    DType::U32 => "u32",
                    DType::I64 => "i64",
                    DType::BF16 => "bf16",
                    DType::F16 => "f16",
                    DType::F32 => "f32",
                },
                "delta_full_scan",
            ));
        }
    };
    let pipeline = kernels.load_pipeline(device, Source::Delta, kernel_name)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    let threads = batch_heads * v_head_dim;
    set_params!(
        encoder,
        (
            batch_heads,
            num_chunks,
            chunk_size,
            k_head_dim,
            v_head_dim,
            &initial_state,
            &weighted_key_scan,
            &k_cumdecay_scan,
            &q_state_scan,
            &local_attn_scan,
            &state_decay_scan,
            &value,
            output
        )
    );
    let (thread_group_count, thread_group_size) = linear_split(&pipeline, threads);
    encoder.use_resource(initial_state.buffer, MTLResourceUsage::Read);
    encoder.use_resource(weighted_key_scan.buffer, MTLResourceUsage::Read);
    encoder.use_resource(k_cumdecay_scan.buffer, MTLResourceUsage::Read);
    encoder.use_resource(q_state_scan.buffer, MTLResourceUsage::Read);
    encoder.use_resource(local_attn_scan.buffer, MTLResourceUsage::Read);
    encoder.use_resource(state_decay_scan.buffer, MTLResourceUsage::Read);
    encoder.use_resource(value.buffer, MTLResourceUsage::Read);
    encoder.use_resource(output, MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}
