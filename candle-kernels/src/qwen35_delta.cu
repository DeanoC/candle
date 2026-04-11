#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <math.h>
#include <stdint.h>

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
__device__ inline float qwen35_to_float<__nv_bfloat16>(__nv_bfloat16 value) {
    return __bfloat162float(value);
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
__device__ inline __nv_bfloat16 qwen35_from_float<__nv_bfloat16>(float value) {
    return __float2bfloat16(value);
}

__device__ inline float qwen35_exp_fast(float x) {
    return __expf(x);
}

__device__ inline float qwen35_sigmoid_fast(float x) {
    if (x >= 0.0f) {
        const float e = qwen35_exp_fast(-x);
        return 1.0f / (1.0f + e);
    }
    const float e = qwen35_exp_fast(x);
    return e / (1.0f + e);
}

__device__ inline float qwen35_softplus_fast(float x) {
    if (x > 20.0f) {
        return x;
    }
    if (x < -20.0f) {
        return qwen35_exp_fast(x);
    }
    return log1pf(qwen35_exp_fast(x));
}

__device__ inline float qwen35_warp_reduce_sum(float value) {
    for (int offset = 16; offset > 0; offset /= 2) {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }
    return value;
}

__device__ inline float qwen35_warp_reduce_max(float value) {
    for (int offset = 16; offset > 0; offset /= 2) {
        value = fmaxf(value, __shfl_down_sync(0xffffffff, value, offset));
    }
    return value;
}

__device__ inline float qwen35_block_reduce_sum(float value) {
    __shared__ float warp_sums[32];
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    const int num_warps = (blockDim.x + 31) / 32;
    value = qwen35_warp_reduce_sum(value);
    if (lane_id == 0) {
        warp_sums[warp_id] = value;
    }
    __syncthreads();
    float total = (lane_id < num_warps) ? warp_sums[lane_id] : 0.0f;
    if (warp_id == 0) {
        total = qwen35_warp_reduce_sum(total);
        if (lane_id == 0) {
            warp_sums[0] = total;
        }
    }
    __syncthreads();
    return warp_sums[0];
}

template <typename T, bool ADD_UNIT_OFFSET>
__device__ inline void rms_norm_impl(
    size_t n_rows,
    size_t n_cols,
    float eps,
    const T *xs,
    const T *weight,
    T *out
) {
    const size_t row = static_cast<size_t>(blockIdx.x);
    if (row >= n_rows) {
        return;
    }

    __shared__ float warp_sums[32];
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    const int num_warps = blockDim.x / 32;
    const size_t row_offset = row * n_cols;

    float sq_sum = 0.0f;
    for (size_t col = threadIdx.x; col < n_cols; col += blockDim.x) {
        const float x = qwen35_to_float(xs[row_offset + col]);
        sq_sum += x * x;
    }
    sq_sum = qwen35_warp_reduce_sum(sq_sum);
    if (lane_id == 0) {
        warp_sums[warp_id] = sq_sum;
    }
    __syncthreads();

    float total_sq_sum = (lane_id < num_warps) ? warp_sums[lane_id] : 0.0f;
    if (warp_id == 0) {
        total_sq_sum = qwen35_warp_reduce_sum(total_sq_sum);
        if (lane_id == 0) {
            warp_sums[0] = total_sq_sum;
        }
    }
    __syncthreads();

    const float inv_rms = rsqrtf(warp_sums[0] / static_cast<float>(n_cols) + eps);
    for (size_t col = threadIdx.x; col < n_cols; col += blockDim.x) {
        float w = qwen35_to_float(weight[col]);
        if (ADD_UNIT_OFFSET) {
            w += 1.0f;
        }
        const float x = qwen35_to_float(xs[row_offset + col]);
        out[row_offset + col] = qwen35_from_float<T>(x * inv_rms * w);
    }
}

template <typename T>
__device__ inline void l2norm_impl(
    size_t n_rows,
    size_t n_cols,
    float eps,
    const T *xs,
    T *out
) {
    const size_t row = static_cast<size_t>(blockIdx.x);
    if (row >= n_rows) {
        return;
    }

    __shared__ float warp_sums[32];
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    const int num_warps = blockDim.x / 32;
    const size_t row_offset = row * n_cols;

    float sq_sum = 0.0f;
    for (size_t col = threadIdx.x; col < n_cols; col += blockDim.x) {
        const float x = qwen35_to_float(xs[row_offset + col]);
        sq_sum += x * x;
    }
    sq_sum = qwen35_warp_reduce_sum(sq_sum);
    if (lane_id == 0) {
        warp_sums[warp_id] = sq_sum;
    }
    __syncthreads();

    float total_sq_sum = (lane_id < num_warps) ? warp_sums[lane_id] : 0.0f;
    if (warp_id == 0) {
        total_sq_sum = qwen35_warp_reduce_sum(total_sq_sum);
        if (lane_id == 0) {
            warp_sums[0] = total_sq_sum;
        }
    }
    __syncthreads();

    const float inv_norm = rsqrtf(warp_sums[0] + eps);
    for (size_t col = threadIdx.x; col < n_cols; col += blockDim.x) {
        const float x = qwen35_to_float(xs[row_offset + col]);
        out[row_offset + col] = qwen35_from_float<T>(x * inv_norm);
    }
}

template <typename T>
__device__ inline void swiglu_mul_impl(
    size_t elem_count,
    const T *gate,
    const T *up,
    T *out,
    size_t tid
) {
    if (tid >= elem_count) {
        return;
    }
    const float g = qwen35_to_float(gate[tid]);
    const float u = qwen35_to_float(up[tid]);
    const float silu = g * qwen35_sigmoid_fast(g);
    out[tid] = qwen35_from_float<T>(silu * u);
}

template <typename T>
__device__ inline void linear_decode_gemv_impl(
    size_t rows,
    size_t out_dim,
    size_t in_dim,
    const T *xs,
    const T *weight,
    T *out
) {
    const size_t out_idx = static_cast<size_t>(blockIdx.x);
    const size_t row = static_cast<size_t>(blockIdx.y);
    if (row >= rows || out_idx >= out_dim) {
        return;
    }

    __shared__ float partial[256];
    const size_t row_offset = row * in_dim;
    const size_t weight_offset = out_idx * in_dim;
    float sum = 0.0f;
    for (size_t col = threadIdx.x; col < in_dim; col += blockDim.x) {
        sum += qwen35_to_float(xs[row_offset + col]) * qwen35_to_float(weight[weight_offset + col]);
    }
    partial[threadIdx.x] = sum;
    __syncthreads();

    for (size_t stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            partial[threadIdx.x] += partial[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        out[row * out_dim + out_idx] = qwen35_from_float<T>(partial[0]);
    }
}

template <typename T, int MAX_HEAD_DIM = 256>
__device__ inline void full_attention_decode_impl(
    size_t batch_size,
    size_t q_heads,
    size_t kv_heads,
    size_t q_len,
    size_t kv_len,
    size_t head_dim,
    size_t num_kv_groups,
    float scale,
    size_t /*seqlen_offset*/,
    const T *query,
    const T *key,
    const T *value,
    T *out
) {
    const size_t row = static_cast<size_t>(blockIdx.x);
    const size_t total_rows = batch_size * q_heads;
    if (row >= total_rows || q_len != 1 || head_dim > MAX_HEAD_DIM) {
        return;
    }

    const size_t batch_idx = row / q_heads;
    const size_t q_head_idx = row % q_heads;
    const size_t kv_head_idx = q_head_idx / num_kv_groups;
    const size_t query_base = ((batch_idx * q_heads + q_head_idx) * q_len) * head_dim;
    const size_t kv_base = (batch_idx * kv_heads + kv_head_idx) * kv_len * head_dim;
    const size_t out_base = query_base;

    __shared__ float acc[MAX_HEAD_DIM];
    __shared__ float row_max;
    __shared__ float row_sum;
    __shared__ float prev_scale;

    for (size_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
        acc[d] = 0.0f;
    }
    if (threadIdx.x == 0) {
        row_max = -INFINITY;
        row_sum = 0.0f;
        prev_scale = 0.0f;
    }
    __syncthreads();

    for (size_t t = 0; t < kv_len; ++t) {
        float dot = 0.0f;
        const size_t key_row = kv_base + t * head_dim;
        for (size_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
            dot += qwen35_to_float(query[query_base + d]) * qwen35_to_float(key[key_row + d]);
        }
        dot = qwen35_block_reduce_sum(dot) * scale;
        if (threadIdx.x == 0) {
            const float new_max = fmaxf(row_max, dot);
            prev_scale = (row_sum == 0.0f) ? 0.0f : qwen35_exp_fast(row_max - new_max);
            row_max = new_max;
            row_sum *= prev_scale;
        }
        __syncthreads();

        if (prev_scale != 1.0f) {
            for (size_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
                acc[d] *= prev_scale;
            }
        }
        const float weight = qwen35_exp_fast(dot - row_max);
        const size_t value_row = kv_base + t * head_dim;
        for (size_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
            acc[d] += weight * qwen35_to_float(value[value_row + d]);
        }
        if (threadIdx.x == 0) {
            row_sum += weight;
        }
        __syncthreads();
    }

    const float inv_sum = (row_sum == 0.0f) ? 0.0f : 1.0f / row_sum;
    for (size_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
        out[out_base + d] = qwen35_from_float<T>(acc[d] * inv_sum);
    }
}

template <typename T>
__device__ inline void rms_norm_gated_impl(
    size_t n_rows,
    size_t n_cols,
    float eps,
    const T *hidden,
    const T *gate,
    const T *weight,
    T *out
) {
    const size_t row = static_cast<size_t>(blockIdx.x);
    if (row >= n_rows) {
        return;
    }

    __shared__ float warp_sums[32];
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    const int num_warps = blockDim.x / 32;
    const size_t row_offset = row * n_cols;

    float sq_sum = 0.0f;
    for (size_t col = threadIdx.x; col < n_cols; col += blockDim.x) {
        const float x = qwen35_to_float(hidden[row_offset + col]);
        sq_sum += x * x;
    }
    sq_sum = qwen35_warp_reduce_sum(sq_sum);
    if (lane_id == 0) {
        warp_sums[warp_id] = sq_sum;
    }
    __syncthreads();

    float total_sq_sum = (lane_id < num_warps) ? warp_sums[lane_id] : 0.0f;
    if (warp_id == 0) {
        total_sq_sum = qwen35_warp_reduce_sum(total_sq_sum);
        if (lane_id == 0) {
            warp_sums[0] = total_sq_sum;
        }
    }
    __syncthreads();

    const float inv_rms = rsqrtf(warp_sums[0] / static_cast<float>(n_cols) + eps);
    for (size_t col = threadIdx.x; col < n_cols; col += blockDim.x) {
        const float x = qwen35_to_float(hidden[row_offset + col]);
        const float w = qwen35_to_float(weight[col]);
        const float g = qwen35_to_float(gate[row_offset + col]);
        const float silu = g * qwen35_sigmoid_fast(g);
        out[row_offset + col] = qwen35_from_float<T>(x * inv_rms * w * silu);
    }
}

template <typename T, typename IndexT>
__device__ inline void embedding_lookup_impl(
    size_t token_count,
    size_t vocab_size,
    size_t hidden_size,
    const T *embeddings,
    const IndexT *indexes,
    T *out,
    size_t tid
) {
    const size_t total_elems = token_count * hidden_size;
    if (tid >= total_elems) {
        return;
    }

    const size_t token_idx = tid / hidden_size;
    const size_t hidden_idx = tid - token_idx * hidden_size;
    const size_t row = static_cast<size_t>(indexes[token_idx]);
    if (row >= vocab_size) {
        out[tid] = qwen35_from_float<T>(0.0f);
        return;
    }
    out[tid] = embeddings[row * hidden_size + hidden_idx];
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

    const float silu = acc * qwen35_sigmoid_fast(acc);
    out[tid] = qwen35_from_float<T>(silu);
}

template <typename T, int MAX_K = 256>
__device__ inline void linear_decode_prepare_impl(
    int batch_size,
    int num_v_heads,
    int head_k_dim,
    int head_v_dim,
    int state_len,
    int kernel_size,
    int head_repeat,
    const T *mixed_qkv,
    const T *prev_conv_state,
    const T *weights,
    const T *a_beta_raw,
    const T *dt_bias,
    const T *a_log_exp,
    float *out,
    size_t pair,
    size_t tid
) {
    const int total_pairs = batch_size * num_v_heads;
    if (pair >= static_cast<size_t>(total_pairs) || head_k_dim > MAX_K) {
        return;
    }

    const int value_dim = num_v_heads * head_v_dim;
    const int key_dim = (num_v_heads / head_repeat) * head_k_dim;
    const int conv_dim = key_dim * 2 + value_dim;
    const int packed_width = 2 * head_k_dim + head_v_dim + 2;

    const int batch = static_cast<int>(pair) / num_v_heads;
    const int v_head = static_cast<int>(pair) - batch * num_v_heads;
    const int k_head = v_head / head_repeat;
    const int mixed_batch_base = batch * conv_dim;
    const int state_batch_base = batch * conv_dim * state_len;
    const int pair_out_base = static_cast<int>(pair) * packed_width;

    auto conv_channel = [&](int channel) -> float {
        const int weight_base = channel * kernel_size;
        const int state_base = state_batch_base + channel * state_len;
        float acc = 0.0f;
        for (int tap = 0; tap < kernel_size; ++tap) {
            float x = 0.0f;
            if (tap + 1 == kernel_size) {
                x = qwen35_to_float(mixed_qkv[mixed_batch_base + channel]);
            } else if (tap < state_len) {
                x = qwen35_to_float(prev_conv_state[state_base + tap]);
            }
            acc += x * qwen35_to_float(weights[weight_base + tap]);
        }
        return acc * qwen35_sigmoid_fast(acc);
    };

    extern __shared__ float shared_heads[];
    float *shared_q = shared_heads;
    float *shared_k = shared_heads + MAX_K;

    if (tid < static_cast<size_t>(head_k_dim)) {
        const int q_base = k_head * head_k_dim;
        const int k_base = key_dim + k_head * head_k_dim;
        shared_q[tid] = conv_channel(q_base + static_cast<int>(tid));
        shared_k[tid] = conv_channel(k_base + static_cast<int>(tid));
    }
    __syncthreads();

    if (tid == 0) {
        float q_sq_sum = 0.0f;
        float k_sq_sum = 0.0f;
        for (int k_idx = 0; k_idx < head_k_dim; ++k_idx) {
            q_sq_sum += shared_q[k_idx] * shared_q[k_idx];
            k_sq_sum += shared_k[k_idx] * shared_k[k_idx];
        }
        const float q_inv = rsqrtf(q_sq_sum + 1e-6f);
        const float k_inv = rsqrtf(k_sq_sum + 1e-6f);
        const float q_scale = rsqrtf(static_cast<float>(head_k_dim));
        for (int k_idx = 0; k_idx < head_k_dim; ++k_idx) {
            out[pair_out_base + k_idx] = shared_q[k_idx] * q_inv * q_scale;
            out[pair_out_base + head_k_dim + k_idx] = shared_k[k_idx] * k_inv;
        }
        const int head_base = batch * (2 * num_v_heads) + v_head;
        const float a_raw = qwen35_to_float(a_beta_raw[head_base]);
        const float beta_raw = qwen35_to_float(a_beta_raw[head_base + num_v_heads]);
        out[pair_out_base + 2 * head_k_dim + head_v_dim] = qwen35_sigmoid_fast(beta_raw);
        const float bias = qwen35_to_float(dt_bias[v_head]);
        const float decay = qwen35_to_float(a_log_exp[v_head]);
        const float g = -qwen35_softplus_fast(a_raw + bias) * decay;
        out[pair_out_base + 2 * head_k_dim + head_v_dim + 1] = qwen35_exp_fast(g);
    }
    if (tid < static_cast<size_t>(head_v_dim)) {
        const int value_channel = key_dim * 2 + v_head * head_v_dim + static_cast<int>(tid);
        out[pair_out_base + 2 * head_k_dim + tid] = conv_channel(value_channel);
    }
}

template <typename T, int MAX_K = 256>
__device__ inline void linear_decode_prepare_packed_cache_impl(
    int batch_size,
    int num_v_heads,
    int head_k_dim,
    int head_v_dim,
    int state_len,
    int kernel_size,
    int head_repeat,
    const T *mixed_qkv,
    const T *prev_conv_state,
    const T *weights,
    const T *a_raw,
    const T *beta_raw,
    const T *value_cache_pack,
    float *out,
    size_t pair,
    size_t tid
) {
    const int total_pairs = batch_size * num_v_heads;
    if (pair >= static_cast<size_t>(total_pairs) || head_k_dim > MAX_K) {
        return;
    }

    const int value_dim = num_v_heads * head_v_dim;
    const int key_dim = (num_v_heads / head_repeat) * head_k_dim;
    const int conv_dim = key_dim * 2 + value_dim;
    const int packed_width = 2 * head_k_dim + head_v_dim + 2;

    const int batch = static_cast<int>(pair) / num_v_heads;
    const int v_head = static_cast<int>(pair) - batch * num_v_heads;
    const int k_head = v_head / head_repeat;
    const int mixed_batch_base = batch * conv_dim;
    const int state_batch_base = batch * conv_dim * state_len;
    const int pair_out_base = static_cast<int>(pair) * packed_width;

    auto conv_channel = [&](int channel) -> float {
        const int weight_base = channel * kernel_size;
        const int state_base = state_batch_base + channel * state_len;
        float acc = 0.0f;
        for (int tap = 0; tap < kernel_size; ++tap) {
            float x = 0.0f;
            if (tap + 1 == kernel_size) {
                x = qwen35_to_float(mixed_qkv[mixed_batch_base + channel]);
            } else if (tap < state_len) {
                x = qwen35_to_float(prev_conv_state[state_base + tap]);
            }
            acc += x * qwen35_to_float(weights[weight_base + tap]);
        }
        return acc * qwen35_sigmoid_fast(acc);
    };

    extern __shared__ float shared_heads[];
    float *shared_q = shared_heads;
    float *shared_k = shared_heads + MAX_K;

    if (tid < static_cast<size_t>(head_k_dim)) {
        const int q_base = k_head * head_k_dim;
        const int k_base = key_dim + k_head * head_k_dim;
        shared_q[tid] = conv_channel(q_base + static_cast<int>(tid));
        shared_k[tid] = conv_channel(k_base + static_cast<int>(tid));
    }
    __syncthreads();

    if (tid == 0) {
        float q_sq_sum = 0.0f;
        float k_sq_sum = 0.0f;
        for (int k_idx = 0; k_idx < head_k_dim; ++k_idx) {
            q_sq_sum += shared_q[k_idx] * shared_q[k_idx];
            k_sq_sum += shared_k[k_idx] * shared_k[k_idx];
        }
        const float q_inv = rsqrtf(q_sq_sum + 1e-6f);
        const float k_inv = rsqrtf(k_sq_sum + 1e-6f);
        const float q_scale = rsqrtf(static_cast<float>(head_k_dim));
        for (int k_idx = 0; k_idx < head_k_dim; ++k_idx) {
            out[pair_out_base + k_idx] = shared_q[k_idx] * q_inv * q_scale;
            out[pair_out_base + head_k_dim + k_idx] = shared_k[k_idx] * k_inv;
        }
        const int head_base = batch * num_v_heads + v_head;
        out[pair_out_base + 2 * head_k_dim + head_v_dim] =
            qwen35_sigmoid_fast(qwen35_to_float(beta_raw[head_base]));
        const float bias = qwen35_to_float(value_cache_pack[v_head]);
        const float decay = qwen35_to_float(value_cache_pack[num_v_heads + v_head]);
        const float g = -qwen35_softplus_fast(qwen35_to_float(a_raw[head_base]) + bias) * decay;
        out[pair_out_base + 2 * head_k_dim + head_v_dim + 1] = qwen35_exp_fast(g);
    }
    if (tid < static_cast<size_t>(head_v_dim)) {
        const int value_channel = key_dim * 2 + v_head * head_v_dim + static_cast<int>(tid);
        out[pair_out_base + 2 * head_k_dim + tid] = conv_channel(value_channel);
    }
}

template <int MAX_K = 256>
__device__ inline void linear_decode_apply_impl(
    int batch_size,
    int num_v_heads,
    int head_k_dim,
    int head_v_dim,
    const float *packed,
    const float *initial_state,
    float *out,
    size_t pair,
    size_t tid
) {
    const int total_pairs = batch_size * num_v_heads;
    if (pair >= static_cast<size_t>(total_pairs) || head_k_dim > MAX_K ||
        tid >= static_cast<size_t>(head_v_dim)) {
        return;
    }

    const int value_dim = num_v_heads * head_v_dim;
    const int packed_width = 2 * head_k_dim + head_v_dim + 2;
    const int batch = static_cast<int>(pair) / num_v_heads;
    const int v_head = static_cast<int>(pair) - batch * num_v_heads;
    const int v_idx = static_cast<int>(tid);
    const int pair_base = static_cast<int>(pair) * packed_width;
    const int state_head_base =
        ((batch * num_v_heads + v_head) * head_k_dim) * head_v_dim + v_idx;
    const int out_base =
        batch * (value_dim + num_v_heads * head_k_dim * head_v_dim) + value_dim +
        (v_head * head_k_dim) * head_v_dim + v_idx;

    const float *q = packed + pair_base;
    const float *k = packed + pair_base + head_k_dim;
    const float *value = packed + pair_base + 2 * head_k_dim;
    const float beta = packed[pair_base + 2 * head_k_dim + head_v_dim];
    const float g_exp = packed[pair_base + 2 * head_k_dim + head_v_dim + 1];

    float state[MAX_K];
    for (int k_idx = 0; k_idx < head_k_dim; ++k_idx) {
        state[k_idx] = initial_state[state_head_base + k_idx * head_v_dim] * g_exp;
    }

    float kv_mem = 0.0f;
    for (int k_idx = 0; k_idx < head_k_dim; ++k_idx) {
        kv_mem += state[k_idx] * k[k_idx];
    }
    const float delta = (value[v_idx] - kv_mem) * beta;

    float out_value = 0.0f;
    for (int k_idx = 0; k_idx < head_k_dim; ++k_idx) {
        state[k_idx] += k[k_idx] * delta;
        out_value += state[k_idx] * q[k_idx];
        out[out_base + k_idx * head_v_dim] = state[k_idx];
    }
    out[batch * (value_dim + num_v_heads * head_k_dim * head_v_dim) + v_head * head_v_dim + v_idx] =
        out_value;
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
    const size_t causal_limit = min(kv_len, seqlen_offset + q_pos + 1);

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

        const float new_max = max(max_score, score);
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
        state[k_idx] = qwen35_to_float(initial_state[bh * state_stride + k_idx * v_head_dim + v_idx]);
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
                v_prime += qwen35_to_float(packed_scan[packed_row + k_head_dim + k_idx])
                    * state[k_idx];
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

#define DEFINE_EMBEDDING_LOOKUP_KERNEL(name, type, index_type) \
extern "C" __global__ void name( \
    size_t token_count, \
    size_t vocab_size, \
    size_t hidden_size, \
    const type *embeddings, \
    const index_type *indexes, \
    type *out \
) { \
    const size_t tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; \
    embedding_lookup_impl<type, index_type>( \
        token_count, vocab_size, hidden_size, embeddings, indexes, out, tid); \
}

#define DEFINE_RMS_NORM_KERNEL(name, type) \
extern "C" __global__ void name( \
    size_t n_rows, \
    size_t n_cols, \
    float eps, \
    int add_unit_offset, \
    const type *xs, \
    const type *weight, \
    type *out \
) { \
    if (add_unit_offset != 0) { \
        rms_norm_impl<type, true>(n_rows, n_cols, eps, xs, weight, out); \
    } else { \
        rms_norm_impl<type, false>(n_rows, n_cols, eps, xs, weight, out); \
    } \
}

#define DEFINE_RMS_NORM_GATED_KERNEL(name, type) \
extern "C" __global__ void name( \
    size_t n_rows, \
    size_t n_cols, \
    float eps, \
    const type *hidden, \
    const type *gate, \
    const type *weight, \
    type *out \
) { \
    rms_norm_gated_impl<type>(n_rows, n_cols, eps, hidden, gate, weight, out); \
}

#define DEFINE_L2NORM_KERNEL(name, type) \
extern "C" __global__ void name( \
    size_t n_rows, \
    size_t n_cols, \
    float eps, \
    const type *xs, \
    type *out \
) { \
    l2norm_impl<type>(n_rows, n_cols, eps, xs, out); \
}

#define DEFINE_SWIGLU_MUL_KERNEL(name, type) \
extern "C" __global__ void name( \
    size_t elem_count, \
    const type *gate, \
    const type *up, \
    type *out \
) { \
    const size_t tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; \
    swiglu_mul_impl<type>(elem_count, gate, up, out, tid); \
}

#define DEFINE_LINEAR_DECODE_GEMV_KERNEL(name, type) \
extern "C" __global__ void name( \
    size_t rows, \
    size_t out_dim, \
    size_t in_dim, \
    const type *xs, \
    const type *weight, \
    type *out \
) { \
    linear_decode_gemv_impl<type>(rows, out_dim, in_dim, xs, weight, out); \
}

#define DEFINE_FULL_ATTENTION_DECODE_KERNEL(name, type) \
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
    full_attention_decode_impl<type>( \
        batch_size, q_heads, kv_heads, q_len, kv_len, head_dim, num_kv_groups, scale, \
        seqlen_offset, query, key, value, out); \
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
        batch_heads, seq_len, k_head_dim, v_head_dim, initial_state, query, key, value, beta, g, \
        out, tid); \
}

#define DEFINE_LINEAR_DECODE_PREPARE_KERNEL(name, type) \
extern "C" __global__ void name( \
    int batch_size, \
    int num_v_heads, \
    int head_k_dim, \
    int head_v_dim, \
    int state_len, \
    int kernel_size, \
    int head_repeat, \
    const type *mixed_qkv, \
    const type *prev_conv_state, \
    const type *weights, \
    const type *a_beta_raw, \
    const type *dt_bias, \
    const type *a_log_exp, \
    float *out \
) { \
    const size_t pair = static_cast<size_t>(blockIdx.x); \
    const size_t tid = static_cast<size_t>(threadIdx.x); \
    linear_decode_prepare_impl<type>( \
        batch_size, num_v_heads, head_k_dim, head_v_dim, state_len, kernel_size, head_repeat, \
        mixed_qkv, prev_conv_state, weights, a_beta_raw, dt_bias, a_log_exp, out, pair, tid); \
}

#define DEFINE_LINEAR_DECODE_PREPARE_PACKED_CACHE_KERNEL(name, type) \
extern "C" __global__ void name( \
    int batch_size, \
    int num_v_heads, \
    int head_k_dim, \
    int head_v_dim, \
    int state_len, \
    int kernel_size, \
    int head_repeat, \
    const type *mixed_qkv, \
    const type *prev_conv_state, \
    const type *weights, \
    const type *a_raw, \
    const type *beta_raw, \
    const type *value_cache_pack, \
    float *out \
) { \
    const size_t pair = static_cast<size_t>(blockIdx.x); \
    const size_t tid = static_cast<size_t>(threadIdx.x); \
    linear_decode_prepare_packed_cache_impl<type>( \
        batch_size, num_v_heads, head_k_dim, head_v_dim, state_len, kernel_size, head_repeat, \
        mixed_qkv, prev_conv_state, weights, a_raw, beta_raw, value_cache_pack, out, pair, tid); \
}

#define DEFINE_LINEAR_DECODE_APPLY_KERNEL(name) \
extern "C" __global__ void name( \
    int batch_size, \
    int num_v_heads, \
    int head_k_dim, \
    int head_v_dim, \
    const float *packed, \
    const float *initial_state, \
    float *out \
) { \
    const size_t pair = static_cast<size_t>(blockIdx.x); \
    const size_t tid = static_cast<size_t>(threadIdx.x); \
    linear_decode_apply_impl<>( \
        batch_size, num_v_heads, head_k_dim, head_v_dim, packed, initial_state, out, pair, tid); \
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
        batch_heads, chunk_size, k_head_dim, v_head_dim, prev_state, query, key, value, beta, g, \
        out, tid); \
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

DEFINE_LINEAR_PREFILL_CONV_PACK_KERNEL(linear_prefill_conv_pack_f16, half)
DEFINE_LINEAR_PREFILL_CONV_PACK_KERNEL(linear_prefill_conv_pack_f32, float)
DEFINE_LINEAR_PREFILL_CONV_PACK_KERNEL(linear_prefill_conv_pack_bf16, __nv_bfloat16)

DEFINE_EMBEDDING_LOOKUP_KERNEL(embedding_lookup_u32_f16, half, uint32_t)
DEFINE_EMBEDDING_LOOKUP_KERNEL(embedding_lookup_u32_f32, float, uint32_t)
DEFINE_EMBEDDING_LOOKUP_KERNEL(embedding_lookup_u32_bf16, __nv_bfloat16, uint32_t)
DEFINE_RMS_NORM_KERNEL(rms_norm_f16, half)
DEFINE_RMS_NORM_KERNEL(rms_norm_f32, float)
DEFINE_RMS_NORM_KERNEL(rms_norm_bf16, __nv_bfloat16)

DEFINE_RMS_NORM_GATED_KERNEL(rms_norm_gated_f16, half)
DEFINE_RMS_NORM_GATED_KERNEL(rms_norm_gated_f32, float)
DEFINE_RMS_NORM_GATED_KERNEL(rms_norm_gated_bf16, __nv_bfloat16)

DEFINE_L2NORM_KERNEL(l2norm_f16, half)
DEFINE_L2NORM_KERNEL(l2norm_f32, float)
DEFINE_L2NORM_KERNEL(l2norm_bf16, __nv_bfloat16)

DEFINE_SWIGLU_MUL_KERNEL(swiglu_mul_f16, half)
DEFINE_SWIGLU_MUL_KERNEL(swiglu_mul_f32, float)
DEFINE_SWIGLU_MUL_KERNEL(swiglu_mul_bf16, __nv_bfloat16)
DEFINE_LINEAR_DECODE_GEMV_KERNEL(linear_decode_gemv_f16, half)
DEFINE_LINEAR_DECODE_GEMV_KERNEL(linear_decode_gemv_f32, float)
DEFINE_LINEAR_DECODE_GEMV_KERNEL(linear_decode_gemv_bf16, __nv_bfloat16)
DEFINE_FULL_ATTENTION_DECODE_KERNEL(full_attention_decode_f16, half)
DEFINE_FULL_ATTENTION_DECODE_KERNEL(full_attention_decode_f32, float)
DEFINE_FULL_ATTENTION_DECODE_KERNEL(full_attention_decode_bf16, __nv_bfloat16)

DEFINE_FULL_ATTN_PREFILL_KERNEL(full_attention_prefill_f16, half)
DEFINE_FULL_ATTN_PREFILL_KERNEL(full_attention_prefill_f32, float)
DEFINE_FULL_ATTN_PREFILL_KERNEL(full_attention_prefill_bf16, __nv_bfloat16)

DEFINE_DELTA_RECURRENT_PREFILL_KERNEL(delta_recurrent_prefill_f16, half)
DEFINE_DELTA_RECURRENT_PREFILL_KERNEL(delta_recurrent_prefill_f32, float)
DEFINE_DELTA_RECURRENT_PREFILL_KERNEL(delta_recurrent_prefill_bf16, __nv_bfloat16)

DEFINE_LINEAR_DECODE_PREPARE_KERNEL(linear_decode_prepare_f16, half)
DEFINE_LINEAR_DECODE_PREPARE_KERNEL(linear_decode_prepare_f32, float)
DEFINE_LINEAR_DECODE_PREPARE_KERNEL(linear_decode_prepare_bf16, __nv_bfloat16)
DEFINE_LINEAR_DECODE_PREPARE_PACKED_CACHE_KERNEL(linear_decode_prepare_packed_cache_f16, half)
DEFINE_LINEAR_DECODE_PREPARE_PACKED_CACHE_KERNEL(linear_decode_prepare_packed_cache_f32, float)
DEFINE_LINEAR_DECODE_PREPARE_PACKED_CACHE_KERNEL(linear_decode_prepare_packed_cache_bf16, __nv_bfloat16)
DEFINE_LINEAR_DECODE_APPLY_KERNEL(linear_decode_apply_f32)

DEFINE_DELTA_CHUNK_STEP_KERNEL(delta_chunk_step_f16, half)
DEFINE_DELTA_CHUNK_STEP_KERNEL(delta_chunk_step_f32, float)
DEFINE_DELTA_CHUNK_STEP_KERNEL(delta_chunk_step_bf16, __nv_bfloat16)

DEFINE_DELTA_CHUNK_WINDOWED_KERNEL(delta_chunk_step_windowed_f16, half)
DEFINE_DELTA_CHUNK_WINDOWED_KERNEL(delta_chunk_step_windowed_f32, float)
DEFINE_DELTA_CHUNK_WINDOWED_KERNEL(delta_chunk_step_windowed_bf16, __nv_bfloat16)

DEFINE_DELTA_CHUNK_WINDOWED_KERNEL(delta_chunk_scan_raw_f16, half)
DEFINE_DELTA_CHUNK_WINDOWED_KERNEL(delta_chunk_scan_raw_f32, float)
DEFINE_DELTA_CHUNK_WINDOWED_KERNEL(delta_chunk_scan_raw_bf16, __nv_bfloat16)

DEFINE_DELTA_STATE_SCAN_KERNEL(delta_state_scan_f16, half)
DEFINE_DELTA_STATE_SCAN_KERNEL(delta_state_scan_f32, float)
DEFINE_DELTA_STATE_SCAN_KERNEL(delta_state_scan_bf16, __nv_bfloat16)

DEFINE_DELTA_CHUNK_FUSED_KERNEL(delta_chunk_fused_f16, half)
DEFINE_DELTA_CHUNK_FUSED_KERNEL(delta_chunk_fused_f32, float)
DEFINE_DELTA_CHUNK_FUSED_KERNEL(delta_chunk_fused_bf16, __nv_bfloat16)

DEFINE_DELTA_FULL_SCAN_KERNEL(delta_full_scan_f16, half)
DEFINE_DELTA_FULL_SCAN_KERNEL(delta_full_scan_f32, float)
DEFINE_DELTA_FULL_SCAN_KERNEL(delta_full_scan_bf16, __nv_bfloat16)

// Reserved entry points for the remaining CUDA-specific tiling work.

#define DEFINE_PLACEHOLDER_QWEN35_KERNEL(name) \
extern "C" __global__ void name() {}

DEFINE_PLACEHOLDER_QWEN35_KERNEL(delta_chunk_step_windowed_2d_f16)
DEFINE_PLACEHOLDER_QWEN35_KERNEL(delta_chunk_step_windowed_2d_f32)
DEFINE_PLACEHOLDER_QWEN35_KERNEL(delta_chunk_step_windowed_2d_bf16)
