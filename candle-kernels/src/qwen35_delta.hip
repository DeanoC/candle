#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <math.h>
#include <stdint.h>

// This is the ROCm/HIP companion for qwen35_delta.cu.
// The first milestone is to keep the entry-point surface stable and make the
// two dense-control prefill kernels compile locally under hipcc:
// - packed linear prefill conv
// - full-attention prefill
// The DeltaNet kernels are declared with the same signatures but remain
// placeholders until Candle grows a real HIP backend/runtime path.

template <typename T>
__device__ inline float qwen35_to_float(T value);

template <>
__device__ inline float qwen35_to_float<float>(float value) {
    return value;
}

template <>
__device__ inline float qwen35_to_float<half>(half value) {
    return __half2float(value);
}

template <>
__device__ inline float qwen35_to_float<hip_bfloat16>(hip_bfloat16 value) {
    return static_cast<float>(value);
}

template <typename T>
__device__ inline T qwen35_from_float(float value);

template <>
__device__ inline float qwen35_from_float<float>(float value) {
    return value;
}

template <>
__device__ inline half qwen35_from_float<half>(float value) {
    return __float2half(value);
}

template <>
__device__ inline hip_bfloat16 qwen35_from_float<hip_bfloat16>(float value) {
    return hip_bfloat16(value);
}

template <typename T>
__device__ inline void linear_prefill_conv_pack_impl(
    size_t batch_size,
    size_t conv_dim,
    size_t total_len,
    size_t seq_len,
    size_t kernel_size,
    const T *mixed_qkv,
    const T *weights,
    T *out,
    size_t tid
) {
    const size_t output_elems = batch_size * seq_len * conv_dim;
    if (tid >= output_elems) {
        return;
    }

    const size_t b = tid / (seq_len * conv_dim);
    const size_t rem = tid - b * seq_len * conv_dim;
    const size_t t = rem / conv_dim;
    const size_t c = rem - t * conv_dim;

    const size_t input_b_offset = b * conv_dim * total_len;
    const size_t input_c_offset = input_b_offset + c * total_len;
    const size_t weight_offset = c * kernel_size;

    float acc = 0.0f;
    for (size_t tap = 0; tap < kernel_size; ++tap) {
        acc += qwen35_to_float(mixed_qkv[input_c_offset + t + tap])
            * qwen35_to_float(weights[weight_offset + tap]);
    }

    const float silu = acc / (1.0f + expf(-acc));
    out[tid] = qwen35_from_float<T>(silu);
}

template <typename T, int MAX_HEAD_DIM = 128>
__device__ inline void full_attention_prefill_impl(
    size_t batch_size,
    size_t q_heads,
    size_t kv_heads,
    size_t q_len,
    size_t kv_len,
    size_t head_dim,
    size_t num_kv_groups,
    float scale,
    size_t seqlen_offset,
    const T *query,
    const T *key,
    const T *value,
    T *out,
    size_t tid
) {
    const size_t total_rows = batch_size * q_heads * q_len;
    if (tid >= total_rows || head_dim > static_cast<size_t>(MAX_HEAD_DIM) || kv_heads == 0 ||
        num_kv_groups == 0) {
        return;
    }

    const size_t rows_per_batch = q_heads * q_len;
    const size_t b = tid / rows_per_batch;
    const size_t row = tid - b * rows_per_batch;
    const size_t q_head = row / q_len;
    const size_t q_pos = row - q_head * q_len;
    size_t kv_head = q_head / num_kv_groups;
    if (kv_head >= kv_heads) {
        kv_head = kv_heads - 1;
    }

    const size_t query_row_offset = ((b * q_heads + q_head) * q_len + q_pos) * head_dim;
    const size_t key_head_offset = (b * kv_heads + kv_head) * kv_len * head_dim;
    const size_t value_head_offset = key_head_offset;
    const size_t out_row_offset = ((b * q_heads + q_head) * q_len + q_pos) * head_dim;
    const size_t causal_limit_candidate = seqlen_offset + q_pos + 1;
    const size_t causal_limit = kv_len < causal_limit_candidate ? kv_len : causal_limit_candidate;

    float q_row[MAX_HEAD_DIM];
    float out_row[MAX_HEAD_DIM];
    for (size_t d = 0; d < head_dim; ++d) {
        q_row[d] = qwen35_to_float(query[query_row_offset + d]);
        out_row[d] = 0.0f;
    }

    if (causal_limit == 0) {
        for (size_t d = 0; d < head_dim; ++d) {
            out[out_row_offset + d] = qwen35_from_float<T>(0.0f);
        }
        return;
    }

    float max_score = -INFINITY;
    float denom = 0.0f;

    for (size_t k_pos = 0; k_pos < causal_limit; ++k_pos) {
        const size_t key_row_offset = key_head_offset + k_pos * head_dim;
        const size_t value_row_offset = value_head_offset + k_pos * head_dim;
        float score = 0.0f;
        for (size_t d = 0; d < head_dim; ++d) {
            score += q_row[d] * qwen35_to_float(key[key_row_offset + d]);
        }
        score *= scale;

        if (!isfinite(max_score)) {
            max_score = score;
            denom = 1.0f;
            for (size_t d = 0; d < head_dim; ++d) {
                out_row[d] = qwen35_to_float(value[value_row_offset + d]);
            }
            continue;
        }

        const float new_max = max_score > score ? max_score : score;
        const float prev_scale = expf(max_score - new_max);
        const float curr_scale = expf(score - new_max);
        denom = denom * prev_scale + curr_scale;
        for (size_t d = 0; d < head_dim; ++d) {
            out_row[d] = out_row[d] * prev_scale
                + curr_scale * qwen35_to_float(value[value_row_offset + d]);
        }
        max_score = new_max;
    }

    const float inv_denom = denom > 0.0f ? 1.0f / denom : 0.0f;
    for (size_t d = 0; d < head_dim; ++d) {
        out[out_row_offset + d] = qwen35_from_float<T>(out_row[d] * inv_denom);
    }
}

template <typename T, int MAX_K = 256>
__device__ inline void delta_recurrent_prefill_impl(
    size_t batch_heads,
    size_t seq_len,
    size_t k_head_dim,
    size_t v_head_dim,
    const T *initial_state,
    const T *query,
    const T *key,
    const T *value,
    const T *beta,
    const T *g,
    T *out,
    size_t tid
) {
    const size_t total_threads = batch_heads * v_head_dim;
    if (tid >= total_threads || k_head_dim > static_cast<size_t>(MAX_K)) {
        return;
    }

    const size_t bh = tid / v_head_dim;
    const size_t v_idx = tid - bh * v_head_dim;
    const size_t state_stride = k_head_dim * v_head_dim;
    const size_t token_stride_k = seq_len * k_head_dim;
    const size_t token_stride_v = seq_len * v_head_dim;
    const size_t token_stride_s = seq_len;
    const size_t out_base = bh * (seq_len + k_head_dim) * v_head_dim;

    float state[MAX_K];
    for (size_t k_idx = 0; k_idx < k_head_dim; ++k_idx) {
        state[k_idx] =
            qwen35_to_float(initial_state[bh * state_stride + k_idx * v_head_dim + v_idx]);
    }

    for (size_t t = 0; t < seq_len; ++t) {
        const float g_t = expf(qwen35_to_float(g[bh * token_stride_s + t]));
        const size_t key_row = bh * token_stride_k + t * k_head_dim;
        const size_t value_row = bh * token_stride_v + t * v_head_dim;
        const size_t beta_row = bh * token_stride_s + t;

        for (size_t k_idx = 0; k_idx < k_head_dim; ++k_idx) {
            state[k_idx] *= g_t;
        }

        float kv_mem = 0.0f;
        for (size_t k_idx = 0; k_idx < k_head_dim; ++k_idx) {
            kv_mem += state[k_idx] * qwen35_to_float(key[key_row + k_idx]);
        }

        const float delta =
            (qwen35_to_float(value[value_row + v_idx]) - kv_mem) * qwen35_to_float(beta[beta_row]);

        for (size_t k_idx = 0; k_idx < k_head_dim; ++k_idx) {
            state[k_idx] += qwen35_to_float(key[key_row + k_idx]) * delta;
        }

        float out_t = 0.0f;
        for (size_t k_idx = 0; k_idx < k_head_dim; ++k_idx) {
            out_t += state[k_idx] * qwen35_to_float(query[key_row + k_idx]);
        }
        out[out_base + t * v_head_dim + v_idx] = qwen35_from_float<T>(out_t);
    }

    const size_t state_out = out_base + seq_len * v_head_dim;
    for (size_t k_idx = 0; k_idx < k_head_dim; ++k_idx) {
        out[state_out + k_idx * v_head_dim + v_idx] = qwen35_from_float<T>(state[k_idx]);
    }
}

template <typename T, int MAX_K = 256>
__device__ inline void delta_chunk_step_impl(
    size_t batch_heads,
    size_t chunk_size,
    size_t k_head_dim,
    size_t v_head_dim,
    const T *prev_state,
    const T *query,
    const T *key,
    const T *value,
    const T *beta,
    const T *g,
    T *out,
    size_t tid
) {
    const size_t total_threads = batch_heads * v_head_dim;
    if (tid >= total_threads || k_head_dim > static_cast<size_t>(MAX_K)) {
        return;
    }

    const size_t bh = tid / v_head_dim;
    const size_t v_idx = tid - bh * v_head_dim;
    const size_t state_stride = k_head_dim * v_head_dim;
    const size_t token_stride_k = chunk_size * k_head_dim;
    const size_t token_stride_v = chunk_size * v_head_dim;
    const size_t token_stride_s = chunk_size;
    const size_t out_base = bh * (chunk_size + k_head_dim) * v_head_dim;

    float state[MAX_K];
    for (size_t k_idx = 0; k_idx < k_head_dim; ++k_idx) {
        state[k_idx] = qwen35_to_float(prev_state[bh * state_stride + k_idx * v_head_dim + v_idx]);
    }

    for (size_t t = 0; t < chunk_size; ++t) {
        const float g_t = expf(qwen35_to_float(g[bh * token_stride_s + t]));
        const size_t key_row = bh * token_stride_k + t * k_head_dim;
        const size_t value_row = bh * token_stride_v + t * v_head_dim;
        const size_t beta_row = bh * token_stride_s + t;

        for (size_t k_idx = 0; k_idx < k_head_dim; ++k_idx) {
            state[k_idx] *= g_t;
        }

        float kv_mem = 0.0f;
        for (size_t k_idx = 0; k_idx < k_head_dim; ++k_idx) {
            kv_mem += state[k_idx] * qwen35_to_float(key[key_row + k_idx]);
        }

        const float delta =
            (qwen35_to_float(value[value_row + v_idx]) - kv_mem) * qwen35_to_float(beta[beta_row]);

        for (size_t k_idx = 0; k_idx < k_head_dim; ++k_idx) {
            state[k_idx] += qwen35_to_float(key[key_row + k_idx]) * delta;
        }

        float out_t = 0.0f;
        for (size_t k_idx = 0; k_idx < k_head_dim; ++k_idx) {
            out_t += state[k_idx] * qwen35_to_float(query[key_row + k_idx]);
        }
        out[out_base + t * v_head_dim + v_idx] = qwen35_from_float<T>(out_t);
    }

    const size_t state_out = out_base + chunk_size * v_head_dim;
    for (size_t k_idx = 0; k_idx < k_head_dim; ++k_idx) {
        out[state_out + k_idx * v_head_dim + v_idx] = qwen35_from_float<T>(state[k_idx]);
    }
}

#define DEFINE_LINEAR_PREFILL_CONV_PACK_KERNEL(name, type) \
extern "C" __global__ void name( \
    size_t batch_size, \
    size_t conv_dim, \
    size_t total_len, \
    size_t seq_len, \
    size_t kernel_size, \
    const type *mixed_qkv, \
    const type *weights, \
    type *out \
) { \
    const size_t tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; \
    linear_prefill_conv_pack_impl<type>( \
        batch_size, conv_dim, total_len, seq_len, kernel_size, mixed_qkv, weights, out, tid); \
}

#define DEFINE_FULL_ATTN_PREFILL_KERNEL(name, type) \
extern "C" __global__ void name( \
    size_t batch_size, \
    size_t q_heads, \
    size_t kv_heads, \
    size_t q_len, \
    size_t kv_len, \
    size_t head_dim, \
    size_t num_kv_groups, \
    float scale, \
    size_t seqlen_offset, \
    const type *query, \
    const type *key, \
    const type *value, \
    type *out \
) { \
    const size_t tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; \
    full_attention_prefill_impl<type>( \
        batch_size, q_heads, kv_heads, q_len, kv_len, head_dim, num_kv_groups, scale, \
        seqlen_offset, query, key, value, out, tid); \
}

#define DEFINE_DELTA_RECURRENT_PREFILL_KERNEL(name, type) \
extern "C" __global__ void name( \
    size_t batch_heads, \
    size_t seq_len, \
    size_t k_head_dim, \
    size_t v_head_dim, \
    const type *initial_state, \
    const type *query, \
    const type *key, \
    const type *value, \
    const type *beta, \
    const type *g, \
    type *out \
) { \
    const size_t tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; \
    delta_recurrent_prefill_impl<type>( \
        batch_heads, seq_len, k_head_dim, v_head_dim, initial_state, query, key, value, beta, \
        g, out, tid); \
}

#define DEFINE_DELTA_CHUNK_STEP_KERNEL(name, type) \
extern "C" __global__ void name( \
    size_t batch_heads, \
    size_t chunk_size, \
    size_t k_head_dim, \
    size_t v_head_dim, \
    const type *prev_state, \
    const type *query, \
    const type *key, \
    const type *value, \
    const type *beta, \
    const type *g, \
    type *out \
) { \
    const size_t tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; \
    delta_chunk_step_impl<type>( \
        batch_heads, chunk_size, k_head_dim, v_head_dim, prev_state, query, key, value, beta, \
        g, out, tid); \
}

#define DEFINE_PLACEHOLDER_DELTA_CHUNK_WINDOWED_KERNEL(name, type) \
extern "C" __global__ void name( \
    size_t, size_t, size_t, size_t, size_t, const type *, const type *, const type *, \
    const type *, const type *, const type *, type * \
) {}

#define DEFINE_PLACEHOLDER_DELTA_STATE_SCAN_KERNEL(name, type) \
extern "C" __global__ void name( \
    size_t, size_t, size_t, size_t, size_t, const type *, const type *, const type *, type * \
) {}

#define DEFINE_PLACEHOLDER_DELTA_CHUNK_FUSED_KERNEL(name, type) \
extern "C" __global__ void name( \
    size_t, size_t, size_t, size_t, const type *, const type *, const type *, type * \
) {}

#define DEFINE_PLACEHOLDER_DELTA_FULL_SCAN_KERNEL(name, type) \
extern "C" __global__ void name( \
    size_t, size_t, size_t, size_t, size_t, const type *, const type *, const type *, \
    const type *, const type *, const type *, const type *, type * \
) {}

#define DEFINE_PLACEHOLDER_QWEN35_KERNEL(name) \
extern "C" __global__ void name() {}

DEFINE_LINEAR_PREFILL_CONV_PACK_KERNEL(linear_prefill_conv_pack_f16, half)
DEFINE_LINEAR_PREFILL_CONV_PACK_KERNEL(linear_prefill_conv_pack_f32, float)
DEFINE_LINEAR_PREFILL_CONV_PACK_KERNEL(linear_prefill_conv_pack_bf16, hip_bfloat16)

DEFINE_FULL_ATTN_PREFILL_KERNEL(full_attention_prefill_f16, half)
DEFINE_FULL_ATTN_PREFILL_KERNEL(full_attention_prefill_f32, float)
DEFINE_FULL_ATTN_PREFILL_KERNEL(full_attention_prefill_bf16, hip_bfloat16)

DEFINE_DELTA_RECURRENT_PREFILL_KERNEL(delta_recurrent_prefill_f16, half)
DEFINE_DELTA_RECURRENT_PREFILL_KERNEL(delta_recurrent_prefill_f32, float)
DEFINE_DELTA_RECURRENT_PREFILL_KERNEL(delta_recurrent_prefill_bf16, hip_bfloat16)

DEFINE_DELTA_CHUNK_STEP_KERNEL(delta_chunk_step_f16, half)
DEFINE_DELTA_CHUNK_STEP_KERNEL(delta_chunk_step_f32, float)
DEFINE_DELTA_CHUNK_STEP_KERNEL(delta_chunk_step_bf16, hip_bfloat16)

DEFINE_PLACEHOLDER_DELTA_CHUNK_WINDOWED_KERNEL(delta_chunk_step_windowed_f16, half)
DEFINE_PLACEHOLDER_DELTA_CHUNK_WINDOWED_KERNEL(delta_chunk_step_windowed_f32, float)
DEFINE_PLACEHOLDER_DELTA_CHUNK_WINDOWED_KERNEL(delta_chunk_step_windowed_bf16, hip_bfloat16)

DEFINE_PLACEHOLDER_DELTA_CHUNK_WINDOWED_KERNEL(delta_chunk_scan_raw_f16, half)
DEFINE_PLACEHOLDER_DELTA_CHUNK_WINDOWED_KERNEL(delta_chunk_scan_raw_f32, float)
DEFINE_PLACEHOLDER_DELTA_CHUNK_WINDOWED_KERNEL(delta_chunk_scan_raw_bf16, hip_bfloat16)

DEFINE_PLACEHOLDER_DELTA_STATE_SCAN_KERNEL(delta_state_scan_f16, half)
DEFINE_PLACEHOLDER_DELTA_STATE_SCAN_KERNEL(delta_state_scan_f32, float)
DEFINE_PLACEHOLDER_DELTA_STATE_SCAN_KERNEL(delta_state_scan_bf16, hip_bfloat16)

DEFINE_PLACEHOLDER_DELTA_CHUNK_FUSED_KERNEL(delta_chunk_fused_f16, half)
DEFINE_PLACEHOLDER_DELTA_CHUNK_FUSED_KERNEL(delta_chunk_fused_f32, float)
DEFINE_PLACEHOLDER_DELTA_CHUNK_FUSED_KERNEL(delta_chunk_fused_bf16, hip_bfloat16)

DEFINE_PLACEHOLDER_DELTA_FULL_SCAN_KERNEL(delta_full_scan_f16, half)
DEFINE_PLACEHOLDER_DELTA_FULL_SCAN_KERNEL(delta_full_scan_f32, float)
DEFINE_PLACEHOLDER_DELTA_FULL_SCAN_KERNEL(delta_full_scan_bf16, hip_bfloat16)

DEFINE_PLACEHOLDER_QWEN35_KERNEL(delta_chunk_step_windowed_2d_f16)
DEFINE_PLACEHOLDER_QWEN35_KERNEL(delta_chunk_step_windowed_2d_f32)
DEFINE_PLACEHOLDER_QWEN35_KERNEL(delta_chunk_step_windowed_2d_bf16)
