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
__device__ inline float qwen35_to_float(T value) {
    return static_cast<float>(value);
}

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
__device__ inline T qwen35_from_float(float value) {
    return static_cast<T>(value);
}

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

template <typename Src, typename Dst>
__device__ inline Dst candle_hip_cast_value(Src value) {
    return qwen35_from_float<Dst>(qwen35_to_float(value));
}

template <int BLOCK_SIZE>
__device__ inline float candle_hip_block_reduce_sum(float value) {
    __shared__ float shared[BLOCK_SIZE];
    shared[threadIdx.x] = value;
    __syncthreads();
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared[threadIdx.x] += shared[threadIdx.x + stride];
        }
        __syncthreads();
    }
    return shared[0];
}

template <int BLOCK_SIZE>
__device__ inline float candle_hip_block_reduce_max(float value) {
    __shared__ float shared[BLOCK_SIZE];
    shared[threadIdx.x] = value;
    __syncthreads();
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared[threadIdx.x] = fmaxf(shared[threadIdx.x], shared[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    return shared[0];
}

template <typename Src, typename Dst>
__global__ void candle_hip_cast_kernel(size_t elem_count, const Src *src, Dst *dst) {
    const size_t tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid < elem_count) {
        dst[tid] = candle_hip_cast_value<Src, Dst>(src[tid]);
    }
}

template <typename T>
__global__ void candle_hip_affine_kernel(
    size_t elem_count,
    const T *src,
    float mul,
    float add,
    T *dst
) {
    const size_t tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid < elem_count) {
        const float value = qwen35_to_float(src[tid]);
        dst[tid] = qwen35_from_float<T>(value * mul + add);
    }
}

template <typename T>
__global__ void candle_hip_sigmoid_kernel(size_t elem_count, const T *src, T *dst) {
    const size_t tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid < elem_count) {
        const float x = qwen35_to_float(src[tid]);
        dst[tid] = qwen35_from_float<T>(1.0f / (1.0f + expf(-x)));
    }
}

template <typename Pred, typename T>
__global__ void candle_hip_where_cond_kernel(
    size_t elem_count,
    const Pred *pred,
    const T *on_true,
    const T *on_false,
    T *dst
) {
    const size_t tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid < elem_count) {
        dst[tid] = pred[tid] != static_cast<Pred>(0) ? on_true[tid] : on_false[tid];
    }
}

template <typename T, int BLOCK_SIZE = 256>
__global__ void candle_hip_reduce_sum_kernel(
    size_t n_rows,
    size_t n_cols,
    const T *src,
    T *dst
) {
    const size_t row = static_cast<size_t>(blockIdx.x);
    const size_t tid = static_cast<size_t>(threadIdx.x);
    if (row >= n_rows) {
        return;
    }

    const T *src_row = src + row * n_cols;
    float local = 0.0f;
    for (size_t col = tid; col < n_cols; col += BLOCK_SIZE) {
        local += qwen35_to_float(src_row[col]);
    }
    const float reduced = candle_hip_block_reduce_sum<BLOCK_SIZE>(local);
    if (tid == 0) {
        dst[row] = qwen35_from_float<T>(reduced);
    }
}

template <typename T, int BLOCK_SIZE = 256>
__global__ void candle_hip_reduce_max_kernel(
    size_t n_rows,
    size_t n_cols,
    const T *src,
    T *dst
) {
    const size_t row = static_cast<size_t>(blockIdx.x);
    const size_t tid = static_cast<size_t>(threadIdx.x);
    if (row >= n_rows) {
        return;
    }

    const T *src_row = src + row * n_cols;
    float local = -INFINITY;
    for (size_t col = tid; col < n_cols; col += BLOCK_SIZE) {
        local = fmaxf(local, qwen35_to_float(src_row[col]));
    }
    const float reduced = candle_hip_block_reduce_max<BLOCK_SIZE>(local);
    if (tid == 0) {
        dst[row] = qwen35_from_float<T>(reduced);
    }
}

template <typename T, int BLOCK_SIZE = 256>
__global__ void candle_hip_reduce_min_kernel(
    size_t n_rows,
    size_t n_cols,
    const T *src,
    T *dst
) {
    const size_t row = static_cast<size_t>(blockIdx.x);
    const size_t tid = static_cast<size_t>(threadIdx.x);
    if (row >= n_rows) {
        return;
    }

    const T *src_row = src + row * n_cols;
    float local = INFINITY;
    for (size_t col = tid; col < n_cols; col += BLOCK_SIZE) {
        local = fminf(local, qwen35_to_float(src_row[col]));
    }
    __shared__ float shared[BLOCK_SIZE];
    shared[threadIdx.x] = local;
    __syncthreads();
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared[threadIdx.x] = fminf(shared[threadIdx.x], shared[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    if (tid == 0) {
        dst[row] = qwen35_from_float<T>(shared[0]);
    }
}

template <typename T, int BLOCK_SIZE = 256>
__global__ void candle_hip_softmax_last_dim_kernel(
    size_t n_rows,
    size_t n_cols,
    const T *src,
    T *dst
) {
    const size_t row = static_cast<size_t>(blockIdx.x);
    const size_t tid = static_cast<size_t>(threadIdx.x);
    if (row >= n_rows) {
        return;
    }

    const T *src_row = src + row * n_cols;
    T *dst_row = dst + row * n_cols;

    float local_max = -INFINITY;
    for (size_t col = tid; col < n_cols; col += BLOCK_SIZE) {
        local_max = fmaxf(local_max, qwen35_to_float(src_row[col]));
    }
    const float row_max = candle_hip_block_reduce_max<BLOCK_SIZE>(local_max);

    float local_sum = 0.0f;
    for (size_t col = tid; col < n_cols; col += BLOCK_SIZE) {
        const float value = expf(qwen35_to_float(src_row[col]) - row_max);
        dst_row[col] = qwen35_from_float<T>(value);
        local_sum += value;
    }
    const float row_sum = candle_hip_block_reduce_sum<BLOCK_SIZE>(local_sum);
    const float inv_sum = row_sum > 0.0f ? 1.0f / row_sum : 0.0f;
    for (size_t col = tid; col < n_cols; col += BLOCK_SIZE) {
        dst_row[col] = qwen35_from_float<T>(qwen35_to_float(dst_row[col]) * inv_sum);
    }
}

template <typename T>
__device__ inline void candle_hip_ropei(
    const T *src,
    const T *cos,
    const T *sin,
    T *dst,
    uint32_t bh,
    uint32_t td,
    uint32_t stride_b
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (2 * idx >= bh * td) {
        return;
    }

    uint32_t rope_idx = idx % (td / 2);
    if (stride_b > 0) {
        const uint32_t b_idx = (2 * idx) / stride_b;
        rope_idx += b_idx * (td / 2);
    }
    const T c = cos[rope_idx];
    const T s = sin[rope_idx];

    dst[2 * idx] = src[2 * idx] * c - src[2 * idx + 1] * s;
    dst[2 * idx + 1] = src[2 * idx] * s + src[2 * idx + 1] * c;
}

template <typename T>
__device__ inline void candle_hip_rope(
    const T *src,
    const T *cos,
    const T *sin,
    T *dst,
    uint32_t bh,
    uint32_t td,
    uint32_t d,
    uint32_t stride_b
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (2 * idx >= bh * td) {
        return;
    }

    const uint32_t i_bh = idx / (td / 2);
    const uint32_t i_td = idx - (td / 2) * i_bh;
    const uint32_t i_t = i_td / (d / 2);
    const uint32_t i_d = i_td - (d / 2) * i_t;
    const uint32_t i1 = i_bh * td + i_t * d + i_d;
    const uint32_t i2 = i1 + d / 2;
    uint32_t i_cs = i_t * (d / 2) + i_d;
    if (stride_b > 0) {
        const uint32_t b_idx = (2 * idx) / stride_b;
        i_cs += b_idx * (td / 2);
    }
    const T c = cos[i_cs];
    const T s = sin[i_cs];

    dst[i1] = src[i1] * c - src[i2] * s;
    dst[i2] = src[i1] * s + src[i2] * c;
}

template <typename T>
__device__ inline void candle_hip_rope_thd(
    const T *src,
    const T *cos,
    const T *sin,
    T *dst,
    uint32_t b,
    uint32_t t,
    uint32_t h,
    uint32_t d,
    uint32_t stride_b
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (2 * idx >= b * t * h * d) {
        return;
    }

    const uint32_t i_bth = idx / (d / 2);
    const uint32_t i_d = idx - (d / 2) * i_bth;
    const uint32_t i_t = (i_bth / h) % t;
    const uint32_t i1 = i_bth * d + i_d;
    const uint32_t i2 = i1 + d / 2;
    uint32_t i_cs = i_t * (d / 2) + i_d;
    if (stride_b > 0) {
        const uint32_t b_idx = (2 * idx) / stride_b;
        i_cs += b_idx * ((t * d) / 2);
    }
    const T c = cos[i_cs];
    const T s = sin[i_cs];

    dst[i1] = src[i1] * c - src[i2] * s;
    dst[i2] = src[i1] * s + src[i2] * c;
}

template <typename T>
__global__ void candle_hip_rope_i_kernel(
    const T *src,
    const T *cos,
    const T *sin,
    T *dst,
    uint32_t bh,
    uint32_t td,
    uint32_t stride_b
) {
    candle_hip_ropei<T>(src, cos, sin, dst, bh, td, stride_b);
}

template <typename T>
__global__ void candle_hip_rope_kernel(
    const T *src,
    const T *cos,
    const T *sin,
    T *dst,
    uint32_t bh,
    uint32_t td,
    uint32_t d,
    uint32_t stride_b
) {
    candle_hip_rope<T>(src, cos, sin, dst, bh, td, d, stride_b);
}

template <typename T>
__global__ void candle_hip_rope_thd_kernel(
    const T *src,
    const T *cos,
    const T *sin,
    T *dst,
    uint32_t b,
    uint32_t t,
    uint32_t h,
    uint32_t d,
    uint32_t stride_b
) {
    candle_hip_rope_thd<T>(src, cos, sin, dst, b, t, h, d, stride_b);
}

__device__ inline float qwen35_wave_sum(float value, int wave_size) {
    for (int offset = wave_size / 2; offset > 0; offset /= 2) {
        value += __shfl_down(value, offset, wave_size);
    }
    return __shfl(value, 0, wave_size);
}

__device__ inline unsigned int qwen35_wave_broadcast_u32(
    unsigned int value,
    int src_lane,
    int wave_size
) {
    return __shfl(value, src_lane, wave_size);
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

template <typename T, int MAX_HEAD_DIM = 128>
__device__ inline void full_attention_prefill_persistent_impl(
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
    unsigned int *row_counter
) {
    if (head_dim > static_cast<size_t>(MAX_HEAD_DIM) || kv_heads == 0 || num_kv_groups == 0) {
        return;
    }

    constexpr int MAX_FRAGMENTS = (MAX_HEAD_DIM + 31) / 32;
    const int wave_size = warpSize;
    const int lane = static_cast<int>(threadIdx.x % wave_size);
    const size_t total_rows = batch_size * q_heads * q_len;

    for (;;) {
        unsigned int row_idx = lane == 0 ? atomicAdd(row_counter, 1u) : 0u;
        row_idx = qwen35_wave_broadcast_u32(row_idx, 0, wave_size);
        const size_t tid = static_cast<size_t>(row_idx);
        if (tid >= total_rows) {
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

        float q_fragments[MAX_FRAGMENTS];
        float out_fragments[MAX_FRAGMENTS];
        size_t d_fragments[MAX_FRAGMENTS];
        int fragment_count = 0;
        for (size_t d = static_cast<size_t>(lane); d < head_dim; d += static_cast<size_t>(wave_size)) {
            d_fragments[fragment_count] = d;
            q_fragments[fragment_count] = qwen35_to_float(query[query_row_offset + d]);
            out_fragments[fragment_count] = 0.0f;
            ++fragment_count;
        }

        if (causal_limit == 0) {
            for (int fragment_idx = 0; fragment_idx < fragment_count; ++fragment_idx) {
                out[out_row_offset + d_fragments[fragment_idx]] = qwen35_from_float<T>(0.0f);
            }
            continue;
        }

        float max_score = -INFINITY;
        float denom = 0.0f;

        for (size_t k_pos = 0; k_pos < causal_limit; ++k_pos) {
            const size_t key_row_offset = key_head_offset + k_pos * head_dim;
            const size_t value_row_offset = value_head_offset + k_pos * head_dim;
            float partial_score = 0.0f;
            for (int fragment_idx = 0; fragment_idx < fragment_count; ++fragment_idx) {
                partial_score +=
                    q_fragments[fragment_idx] * qwen35_to_float(key[key_row_offset + d_fragments[fragment_idx]]);
            }
            float score = qwen35_wave_sum(partial_score, wave_size) * scale;

            if (!isfinite(max_score)) {
                max_score = score;
                denom = 1.0f;
                for (int fragment_idx = 0; fragment_idx < fragment_count; ++fragment_idx) {
                    out_fragments[fragment_idx] =
                        qwen35_to_float(value[value_row_offset + d_fragments[fragment_idx]]);
                }
                continue;
            }

            const float new_max = max_score > score ? max_score : score;
            const float prev_scale = expf(max_score - new_max);
            const float curr_scale = expf(score - new_max);
            denom = denom * prev_scale + curr_scale;
            for (int fragment_idx = 0; fragment_idx < fragment_count; ++fragment_idx) {
                out_fragments[fragment_idx] =
                    out_fragments[fragment_idx] * prev_scale
                    + curr_scale * qwen35_to_float(value[value_row_offset + d_fragments[fragment_idx]]);
            }
            max_score = new_max;
        }

        const float inv_denom = denom > 0.0f ? 1.0f / denom : 0.0f;
        for (int fragment_idx = 0; fragment_idx < fragment_count; ++fragment_idx) {
            out[out_row_offset + d_fragments[fragment_idx]] =
                qwen35_from_float<T>(out_fragments[fragment_idx] * inv_denom);
        }
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

template <typename T, int MAX_K = 256>
__device__ inline void delta_chunk_windowed_impl(
    size_t batch_heads,
    size_t num_chunks,
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
    const size_t total_tokens = num_chunks * chunk_size;
    const size_t state_stride = k_head_dim * v_head_dim;
    const size_t token_stride_k = total_tokens * k_head_dim;
    const size_t token_stride_v = total_tokens * v_head_dim;
    const size_t token_stride_s = total_tokens;
    const size_t out_base = bh * (total_tokens + k_head_dim) * v_head_dim;

    float state[MAX_K];
    for (size_t k_idx = 0; k_idx < k_head_dim; ++k_idx) {
        state[k_idx] = qwen35_to_float(prev_state[bh * state_stride + k_idx * v_head_dim + v_idx]);
    }

    for (size_t t = 0; t < total_tokens; ++t) {
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

    const size_t state_out = out_base + total_tokens * v_head_dim;
    for (size_t k_idx = 0; k_idx < k_head_dim; ++k_idx) {
        out[state_out + k_idx * v_head_dim + v_idx] = qwen35_from_float<T>(state[k_idx]);
    }
}

template <typename T, int MAX_K = 256>
__device__ inline void delta_state_scan_impl(
    size_t batch_heads,
    size_t num_chunks,
    size_t chunk_size,
    size_t k_head_dim,
    size_t v_head_dim,
    const T *initial_state,
    const T *packed_scan,
    const T *value,
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
    const size_t packed_width = 2 * k_head_dim + 1;

    float state[MAX_K];
    for (size_t k_idx = 0; k_idx < k_head_dim; ++k_idx) {
        const size_t idx = bh * state_stride + k_idx * v_head_dim + v_idx;
        state[k_idx] = qwen35_to_float(initial_state[idx]);
        out[idx] = qwen35_from_float<T>(state[k_idx]);
    }

    for (size_t chunk = 0; chunk < num_chunks; ++chunk) {
        const size_t packed_chunk_base = ((bh * num_chunks) + chunk) * chunk_size * packed_width;
        const size_t value_chunk_base = ((bh * num_chunks) + chunk) * chunk_size * v_head_dim;
        const float state_decay = qwen35_to_float(packed_scan[packed_chunk_base + 2 * k_head_dim]);
        float update[MAX_K];
        for (size_t k_idx = 0; k_idx < k_head_dim; ++k_idx) {
            update[k_idx] = 0.0f;
        }

        for (size_t t = 0; t < chunk_size; ++t) {
            const size_t packed_row = packed_chunk_base + t * packed_width;
            const size_t value_row = value_chunk_base + t * v_head_dim;
            float v_prime = 0.0f;
            for (size_t k_idx = 0; k_idx < k_head_dim; ++k_idx) {
                v_prime += qwen35_to_float(packed_scan[packed_row + k_head_dim + k_idx]) * state[k_idx];
            }
            const float v_new = qwen35_to_float(value[value_row + v_idx]) - v_prime;
            for (size_t k_idx = 0; k_idx < k_head_dim; ++k_idx) {
                update[k_idx] += qwen35_to_float(packed_scan[packed_row + k_idx]) * v_new;
            }
        }

        const size_t out_chunk_base = ((bh * (num_chunks + 1)) + (chunk + 1)) * state_stride;
        for (size_t k_idx = 0; k_idx < k_head_dim; ++k_idx) {
            state[k_idx] = state_decay * state[k_idx] + update[k_idx];
            out[out_chunk_base + k_idx * v_head_dim + v_idx] = qwen35_from_float<T>(state[k_idx]);
        }
    }
}

template <typename T, int MAX_K = 256, int MAX_CHUNK = 64>
__device__ inline void delta_chunk_fused_impl(
    size_t batch_heads,
    size_t chunk_size,
    size_t k_head_dim,
    size_t v_head_dim,
    const T *prev_state,
    const T *packed_chunk,
    const T *value,
    T *out,
    size_t tid
) {
    const size_t total_threads = batch_heads * v_head_dim;
    if (tid >= total_threads || k_head_dim > static_cast<size_t>(MAX_K) ||
        chunk_size > static_cast<size_t>(MAX_CHUNK)) {
        return;
    }

    const size_t bh = tid / v_head_dim;
    const size_t v_idx = tid - bh * v_head_dim;
    const size_t state_stride = k_head_dim * v_head_dim;
    const size_t packed_width = 3 * k_head_dim + 1;
    const size_t chunk_out_stride = (2 * chunk_size + k_head_dim) * v_head_dim;
    const size_t packed_base = bh * chunk_size * packed_width;
    const size_t value_base = bh * chunk_size * v_head_dim;
    const size_t out_base = bh * chunk_out_stride;

    float state[MAX_K];
    for (size_t k_idx = 0; k_idx < k_head_dim; ++k_idx) {
        state[k_idx] = qwen35_to_float(prev_state[bh * state_stride + k_idx * v_head_dim + v_idx]);
    }

    float v_new[MAX_CHUNK];
    float attn_inter[MAX_CHUNK];
    for (size_t t = 0; t < chunk_size; ++t) {
        const size_t packed_row = packed_base + t * packed_width;
        float v_prime = 0.0f;
        float attn = 0.0f;
        for (size_t k_idx = 0; k_idx < k_head_dim; ++k_idx) {
            v_prime += qwen35_to_float(packed_chunk[packed_row + k_head_dim + k_idx]) * state[k_idx];
            attn += qwen35_to_float(packed_chunk[packed_row + 2 * k_head_dim + k_idx]) * state[k_idx];
        }
        v_new[t] = qwen35_to_float(value[value_base + t * v_head_dim + v_idx]) - v_prime;
        attn_inter[t] = attn;
        out[out_base + t * v_head_dim + v_idx] = qwen35_from_float<T>(v_new[t]);
        out[out_base + (chunk_size + t) * v_head_dim + v_idx] = qwen35_from_float<T>(attn_inter[t]);
    }

    const float state_decay = qwen35_to_float(packed_chunk[packed_base + 3 * k_head_dim]);
    for (size_t k_idx = 0; k_idx < k_head_dim; ++k_idx) {
        float update = 0.0f;
        for (size_t t = 0; t < chunk_size; ++t) {
            const size_t packed_row = packed_base + t * packed_width;
            update += qwen35_to_float(packed_chunk[packed_row + k_idx]) * v_new[t];
        }
        out[out_base + (2 * chunk_size + k_idx) * v_head_dim + v_idx] =
            qwen35_from_float<T>(state_decay * state[k_idx] + update);
    }
}

template <typename T, int MAX_K = 256, int MAX_CHUNK = 64>
__device__ inline void delta_full_scan_impl(
    size_t batch_heads,
    size_t num_chunks,
    size_t chunk_size,
    size_t k_head_dim,
    size_t v_head_dim,
    const T *initial_state,
    const T *weighted_key_scan,
    const T *k_cumdecay_scan,
    const T *q_state_scan,
    const T *local_attn_scan,
    const T *state_decay_scan,
    const T *value,
    T *out,
    size_t tid
) {
    const size_t total_threads = batch_heads * v_head_dim;
    if (tid >= total_threads || k_head_dim > static_cast<size_t>(MAX_K) ||
        chunk_size > static_cast<size_t>(MAX_CHUNK)) {
        return;
    }

    const size_t bh = tid / v_head_dim;
    const size_t v_idx = tid - bh * v_head_dim;
    const size_t state_stride = k_head_dim * v_head_dim;
    const size_t token_count = num_chunks * chunk_size;
    const size_t scan_base = bh * num_chunks * chunk_size * k_head_dim;
    const size_t local_base = bh * num_chunks * chunk_size * chunk_size;
    const size_t decay_base = bh * num_chunks;
    const size_t value_base = bh * token_count * v_head_dim;
    const size_t out_base = bh * (token_count + k_head_dim) * v_head_dim;

    float state[MAX_K];
    for (size_t k_idx = 0; k_idx < k_head_dim; ++k_idx) {
        state[k_idx] = qwen35_to_float(initial_state[bh * state_stride + k_idx * v_head_dim + v_idx]);
    }

    float v_new[MAX_CHUNK];
    float attn_inter[MAX_CHUNK];
    for (size_t chunk = 0; chunk < num_chunks; ++chunk) {
        const size_t chunk_scan = scan_base + chunk * chunk_size * k_head_dim;
        const size_t chunk_local = local_base + chunk * chunk_size * chunk_size;
        const size_t chunk_value = value_base + chunk * chunk_size * v_head_dim;
        for (size_t t = 0; t < chunk_size; ++t) {
            float v_prime = 0.0f;
            float attn = 0.0f;
            const size_t row = chunk_scan + t * k_head_dim;
            for (size_t k_idx = 0; k_idx < k_head_dim; ++k_idx) {
                v_prime += qwen35_to_float(k_cumdecay_scan[row + k_idx]) * state[k_idx];
                attn += qwen35_to_float(q_state_scan[row + k_idx]) * state[k_idx];
            }
            v_new[t] = qwen35_to_float(value[chunk_value + t * v_head_dim + v_idx]) - v_prime;
            attn_inter[t] = attn;
        }

        for (size_t t = 0; t < chunk_size; ++t) {
            float local = 0.0f;
            const size_t row = chunk_local + t * chunk_size;
            for (size_t s = 0; s < chunk_size; ++s) {
                local += qwen35_to_float(local_attn_scan[row + s]) * v_new[s];
            }
            out[out_base + (chunk * chunk_size + t) * v_head_dim + v_idx] =
                qwen35_from_float<T>(attn_inter[t] + local);
        }

        const float state_decay = qwen35_to_float(state_decay_scan[decay_base + chunk]);
        for (size_t k_idx = 0; k_idx < k_head_dim; ++k_idx) {
            float update = 0.0f;
            for (size_t t = 0; t < chunk_size; ++t) {
                const size_t row = chunk_scan + t * k_head_dim;
                update += qwen35_to_float(weighted_key_scan[row + k_idx]) * v_new[t];
            }
            state[k_idx] = state_decay * state[k_idx] + update;
        }
    }

    const size_t state_out = out_base + token_count * v_head_dim;
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

#define DEFINE_FULL_ATTN_PREFILL_PERSISTENT_KERNEL(name, type) \
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
    type *out, \
    unsigned int *row_counter \
) { \
    full_attention_prefill_persistent_impl<type>( \
        batch_size, q_heads, kv_heads, q_len, kv_len, head_dim, num_kv_groups, scale, \
        seqlen_offset, query, key, value, out, row_counter); \
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

#define DEFINE_DELTA_CHUNK_WINDOWED_KERNEL(name, type) \
extern "C" __global__ void name( \
    size_t batch_heads, \
    size_t num_chunks, \
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
    delta_chunk_windowed_impl<type>( \
        batch_heads, num_chunks, chunk_size, k_head_dim, v_head_dim, prev_state, query, key, \
        value, beta, g, out, tid); \
}

#define DEFINE_DELTA_STATE_SCAN_KERNEL(name, type) \
extern "C" __global__ void name( \
    size_t batch_heads, \
    size_t num_chunks, \
    size_t chunk_size, \
    size_t k_head_dim, \
    size_t v_head_dim, \
    const type *initial_state, \
    const type *packed_scan, \
    const type *value, \
    type *out \
) { \
    const size_t tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; \
    delta_state_scan_impl<type>( \
        batch_heads, num_chunks, chunk_size, k_head_dim, v_head_dim, initial_state, packed_scan, \
        value, out, tid); \
}

#define DEFINE_DELTA_CHUNK_FUSED_KERNEL(name, type) \
extern "C" __global__ void name( \
    size_t batch_heads, \
    size_t chunk_size, \
    size_t k_head_dim, \
    size_t v_head_dim, \
    const type *prev_state, \
    const type *packed_chunk, \
    const type *value, \
    type *out \
) { \
    const size_t tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; \
    delta_chunk_fused_impl<type>( \
        batch_heads, chunk_size, k_head_dim, v_head_dim, prev_state, packed_chunk, value, out, \
        tid); \
}

#define DEFINE_DELTA_FULL_SCAN_KERNEL(name, type) \
extern "C" __global__ void name( \
    size_t batch_heads, \
    size_t num_chunks, \
    size_t chunk_size, \
    size_t k_head_dim, \
    size_t v_head_dim, \
    const type *initial_state, \
    const type *weighted_key_scan, \
    const type *k_cumdecay_scan, \
    const type *q_state_scan, \
    const type *local_attn_scan, \
    const type *state_decay_scan, \
    const type *value, \
    type *out \
) { \
    const size_t tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; \
    delta_full_scan_impl<type>( \
        batch_heads, num_chunks, chunk_size, k_head_dim, v_head_dim, initial_state, \
        weighted_key_scan, k_cumdecay_scan, q_state_scan, local_attn_scan, state_decay_scan, \
        value, out, tid); \
}

#define DEFINE_PLACEHOLDER_QWEN35_KERNEL(name) \
extern "C" __global__ void name() {}

DEFINE_LINEAR_PREFILL_CONV_PACK_KERNEL(linear_prefill_conv_pack_f16, half)
DEFINE_LINEAR_PREFILL_CONV_PACK_KERNEL(linear_prefill_conv_pack_f32, float)
DEFINE_LINEAR_PREFILL_CONV_PACK_KERNEL(linear_prefill_conv_pack_bf16, hip_bfloat16)

DEFINE_FULL_ATTN_PREFILL_KERNEL(full_attention_prefill_f16, half)
DEFINE_FULL_ATTN_PREFILL_KERNEL(full_attention_prefill_f32, float)
DEFINE_FULL_ATTN_PREFILL_KERNEL(full_attention_prefill_bf16, hip_bfloat16)

DEFINE_FULL_ATTN_PREFILL_PERSISTENT_KERNEL(full_attention_prefill_persistent_f16, half)
DEFINE_FULL_ATTN_PREFILL_PERSISTENT_KERNEL(full_attention_prefill_persistent_f32, float)
DEFINE_FULL_ATTN_PREFILL_PERSISTENT_KERNEL(full_attention_prefill_persistent_bf16, hip_bfloat16)

DEFINE_DELTA_RECURRENT_PREFILL_KERNEL(delta_recurrent_prefill_f16, half)
DEFINE_DELTA_RECURRENT_PREFILL_KERNEL(delta_recurrent_prefill_f32, float)
DEFINE_DELTA_RECURRENT_PREFILL_KERNEL(delta_recurrent_prefill_bf16, hip_bfloat16)

DEFINE_DELTA_CHUNK_STEP_KERNEL(delta_chunk_step_f16, half)
DEFINE_DELTA_CHUNK_STEP_KERNEL(delta_chunk_step_f32, float)
DEFINE_DELTA_CHUNK_STEP_KERNEL(delta_chunk_step_bf16, hip_bfloat16)

DEFINE_DELTA_CHUNK_WINDOWED_KERNEL(delta_chunk_step_windowed_f16, half)
DEFINE_DELTA_CHUNK_WINDOWED_KERNEL(delta_chunk_step_windowed_f32, float)
DEFINE_DELTA_CHUNK_WINDOWED_KERNEL(delta_chunk_step_windowed_bf16, hip_bfloat16)

DEFINE_DELTA_CHUNK_WINDOWED_KERNEL(delta_chunk_scan_raw_f16, half)
DEFINE_DELTA_CHUNK_WINDOWED_KERNEL(delta_chunk_scan_raw_f32, float)
DEFINE_DELTA_CHUNK_WINDOWED_KERNEL(delta_chunk_scan_raw_bf16, hip_bfloat16)

DEFINE_DELTA_STATE_SCAN_KERNEL(delta_state_scan_f16, half)
DEFINE_DELTA_STATE_SCAN_KERNEL(delta_state_scan_f32, float)
DEFINE_DELTA_STATE_SCAN_KERNEL(delta_state_scan_bf16, hip_bfloat16)

DEFINE_DELTA_CHUNK_FUSED_KERNEL(delta_chunk_fused_f16, half)
DEFINE_DELTA_CHUNK_FUSED_KERNEL(delta_chunk_fused_f32, float)
DEFINE_DELTA_CHUNK_FUSED_KERNEL(delta_chunk_fused_bf16, hip_bfloat16)

DEFINE_DELTA_FULL_SCAN_KERNEL(delta_full_scan_f16, half)
DEFINE_DELTA_FULL_SCAN_KERNEL(delta_full_scan_f32, float)
DEFINE_DELTA_FULL_SCAN_KERNEL(delta_full_scan_bf16, hip_bfloat16)

DEFINE_PLACEHOLDER_QWEN35_KERNEL(delta_chunk_step_windowed_2d_f16)
DEFINE_PLACEHOLDER_QWEN35_KERNEL(delta_chunk_step_windowed_2d_f32)
DEFINE_PLACEHOLDER_QWEN35_KERNEL(delta_chunk_step_windowed_2d_bf16)
