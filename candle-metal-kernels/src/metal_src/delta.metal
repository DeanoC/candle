#include <metal_stdlib>
using namespace metal;

template <typename T, typename AccumT>
kernel void linear_prefill_conv_pack(
    constant size_t &batch_size,
    constant size_t &conv_dim,
    constant size_t &total_len,
    constant size_t &seq_len,
    constant size_t &kernel_size,
    device const T *mixed_qkv,
    device const T *weights,
    device T *out,
    uint tid [[thread_position_in_grid]]
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

    AccumT acc = AccumT(0);
    for (size_t tap = 0; tap < kernel_size; ++tap) {
        acc += AccumT(mixed_qkv[input_c_offset + t + tap]) * AccumT(weights[weight_offset + tap]);
    }

    const AccumT silu = acc / (AccumT(1) + exp(-acc));
    out[tid] = T(silu);
}

template [[host_name("linear_prefill_conv_pack_f16")]] [[kernel]] decltype(linear_prefill_conv_pack<half, float>) linear_prefill_conv_pack<half, float>;
template [[host_name("linear_prefill_conv_pack_f32")]] [[kernel]] decltype(linear_prefill_conv_pack<float, float>) linear_prefill_conv_pack<float, float>;
#if defined(__HAVE_BFLOAT__)
template [[host_name("linear_prefill_conv_pack_bf16")]] [[kernel]] decltype(linear_prefill_conv_pack<bfloat, float>) linear_prefill_conv_pack<bfloat, float>;
#endif

template <typename T, typename AccumT, ushort MAX_HEAD_DIM = 128>
kernel void full_attention_prefill(
    constant size_t &batch_size,
    constant size_t &q_heads,
    constant size_t &kv_heads,
    constant size_t &q_len,
    constant size_t &kv_len,
    constant size_t &head_dim,
    constant size_t &num_kv_groups,
    constant float &scale,
    constant size_t &seqlen_offset,
    device const T *query,
    device const T *key,
    device const T *value,
    device T *out,
    uint tid [[thread_position_in_grid]]
) {
    const size_t total_rows = batch_size * q_heads * q_len;
    if (tid >= total_rows || head_dim > MAX_HEAD_DIM || kv_heads == 0 || num_kv_groups == 0) {
        return;
    }

    const size_t rows_per_batch = q_heads * q_len;
    const size_t b = tid / rows_per_batch;
    const size_t row = tid - b * rows_per_batch;
    const size_t q_head = row / q_len;
    const size_t q_pos = row - q_head * q_len;
    const size_t kv_head = min(q_head / num_kv_groups, kv_heads - 1);

    const size_t query_row_offset =
        ((b * q_heads + q_head) * q_len + q_pos) * head_dim;
    const size_t key_head_offset = (b * kv_heads + kv_head) * kv_len * head_dim;
    const size_t value_head_offset = key_head_offset;
    const size_t out_row_offset =
        ((b * q_heads + q_head) * q_len + q_pos) * head_dim;
    const size_t causal_limit = min(kv_len, seqlen_offset + q_pos + 1);

    AccumT q_row[MAX_HEAD_DIM];
    AccumT out_row[MAX_HEAD_DIM];
    for (size_t d = 0; d < head_dim; ++d) {
        q_row[d] = AccumT(query[query_row_offset + d]);
        out_row[d] = AccumT(0);
    }
    for (size_t d = head_dim; d < MAX_HEAD_DIM; ++d) {
        out_row[d] = AccumT(0);
    }

    if (causal_limit == 0) {
        for (size_t d = 0; d < head_dim; ++d) {
            out[out_row_offset + d] = T(0);
        }
        return;
    }

    AccumT max_score = -INFINITY;
    AccumT denom = AccumT(0);

    for (size_t k_pos = 0; k_pos < causal_limit; ++k_pos) {
        const size_t key_row_offset = key_head_offset + k_pos * head_dim;
        const size_t value_row_offset = value_head_offset + k_pos * head_dim;
        AccumT score = AccumT(0);
        for (size_t d = 0; d < head_dim; ++d) {
            score += q_row[d] * AccumT(key[key_row_offset + d]);
        }
        score *= AccumT(scale);

        if (!isfinite(max_score)) {
            max_score = score;
            denom = AccumT(1);
            for (size_t d = 0; d < head_dim; ++d) {
                out_row[d] = AccumT(value[value_row_offset + d]);
            }
            continue;
        }

        const AccumT new_max = max(max_score, score);
        const AccumT prev_scale = exp(max_score - new_max);
        const AccumT curr_scale = exp(score - new_max);
        denom = denom * prev_scale + curr_scale;
        for (size_t d = 0; d < head_dim; ++d) {
            out_row[d] =
                out_row[d] * prev_scale + curr_scale * AccumT(value[value_row_offset + d]);
        }
        max_score = new_max;
    }

    const AccumT inv_denom = denom > AccumT(0) ? AccumT(1) / denom : AccumT(0);
    for (size_t d = 0; d < head_dim; ++d) {
        out[out_row_offset + d] = T(out_row[d] * inv_denom);
    }
}

template [[host_name("full_attention_prefill_f16")]] [[kernel]] decltype(full_attention_prefill<half, float>) full_attention_prefill<half, float>;
template [[host_name("full_attention_prefill_f32")]] [[kernel]] decltype(full_attention_prefill<float, float>) full_attention_prefill<float, float>;
#if defined(__HAVE_BFLOAT__)
template [[host_name("full_attention_prefill_bf16")]] [[kernel]] decltype(full_attention_prefill<bfloat, float>) full_attention_prefill<bfloat, float>;
#endif

template <typename T, typename AccumT>
kernel void delta_state_update(
    constant size_t &batch_heads,
    constant size_t &chunk_size,
    constant size_t &k_head_dim,
    constant size_t &v_head_dim,
    device const T *prev_state,
    device const T *weighted_key,
    device const T *value,
    device T *out,
    uint tid [[thread_position_in_grid]]
) {
    const size_t output_elems = batch_heads * k_head_dim * v_head_dim;
    if (tid >= output_elems) {
        return;
    }

    const size_t bh_stride_out = k_head_dim * v_head_dim;
    const size_t bh = tid / bh_stride_out;
    const size_t rem = tid - bh * bh_stride_out;
    const size_t k_idx = rem / v_head_dim;
    const size_t v_idx = rem - k_idx * v_head_dim;

    const size_t prev_idx = bh * bh_stride_out + k_idx * v_head_dim + v_idx;
    const size_t key_bh_offset = bh * chunk_size * k_head_dim;
    const size_t value_bh_offset = bh * chunk_size * v_head_dim;

    AccumT acc = AccumT(prev_state[prev_idx]);
    for (size_t t = 0; t < chunk_size; ++t) {
        const size_t key_idx = key_bh_offset + t * k_head_dim + k_idx;
        const size_t value_idx = value_bh_offset + t * v_head_dim + v_idx;
        acc += AccumT(weighted_key[key_idx]) * AccumT(value[value_idx]);
    }
    out[prev_idx] = T(acc);
}

template [[host_name("delta_state_update_f16")]] [[kernel]] decltype(delta_state_update<half, float>) delta_state_update<half, float>;
template [[host_name("delta_state_update_f32")]] [[kernel]] decltype(delta_state_update<float, float>) delta_state_update<float, float>;
#if defined(__HAVE_BFLOAT__)
template [[host_name("delta_state_update_bf16")]] [[kernel]] decltype(delta_state_update<bfloat, float>) delta_state_update<bfloat, float>;
#endif

template <typename T, typename AccumT, ushort MAX_K = 256>
kernel void delta_recurrent_prefill(
    constant size_t &batch_heads,
    constant size_t &seq_len,
    constant size_t &k_head_dim,
    constant size_t &v_head_dim,
    device const T *initial_state,
    device const T *query,
    device const T *key,
    device const T *value,
    device const T *beta,
    device const T *g,
    device T *out,
    uint tid [[thread_position_in_grid]]
) {
    const size_t total_threads = batch_heads * v_head_dim;
    if (tid >= total_threads || k_head_dim > MAX_K) {
        return;
    }

    const size_t bh = tid / v_head_dim;
    const size_t v_idx = tid - bh * v_head_dim;
    const size_t state_stride = k_head_dim * v_head_dim;
    const size_t token_stride_k = seq_len * k_head_dim;
    const size_t token_stride_v = seq_len * v_head_dim;
    const size_t token_stride_s = seq_len;
    const size_t out_base = bh * (seq_len + k_head_dim) * v_head_dim;

    AccumT state[MAX_K];
    for (size_t k_idx = 0; k_idx < k_head_dim; ++k_idx) {
        state[k_idx] = AccumT(initial_state[bh * state_stride + k_idx * v_head_dim + v_idx]);
    }

    for (size_t t = 0; t < seq_len; ++t) {
        const AccumT g_t = exp(AccumT(g[bh * token_stride_s + t]));
        const size_t key_row = bh * token_stride_k + t * k_head_dim;
        const size_t value_row = bh * token_stride_v + t * v_head_dim;
        const size_t beta_row = bh * token_stride_s + t;

        for (size_t k_idx = 0; k_idx < k_head_dim; ++k_idx) {
            state[k_idx] *= g_t;
        }

        AccumT kv_mem = AccumT(0);
        for (size_t k_idx = 0; k_idx < k_head_dim; ++k_idx) {
            kv_mem += state[k_idx] * AccumT(key[key_row + k_idx]);
        }

        const AccumT delta =
            (AccumT(value[value_row + v_idx]) - kv_mem) * AccumT(beta[beta_row]);

        for (size_t k_idx = 0; k_idx < k_head_dim; ++k_idx) {
            state[k_idx] += AccumT(key[key_row + k_idx]) * delta;
        }

        AccumT out_t = AccumT(0);
        for (size_t k_idx = 0; k_idx < k_head_dim; ++k_idx) {
            out_t += state[k_idx] * AccumT(query[key_row + k_idx]);
        }
        out[out_base + t * v_head_dim + v_idx] = T(out_t);
    }

    const size_t state_out = out_base + seq_len * v_head_dim;
    for (size_t k_idx = 0; k_idx < k_head_dim; ++k_idx) {
        out[state_out + k_idx * v_head_dim + v_idx] = T(state[k_idx]);
    }
}

template [[host_name("delta_recurrent_prefill_f16")]] [[kernel]] decltype(delta_recurrent_prefill<half, float>) delta_recurrent_prefill<half, float>;
template [[host_name("delta_recurrent_prefill_f32")]] [[kernel]] decltype(delta_recurrent_prefill<float, float>) delta_recurrent_prefill<float, float>;
#if defined(__HAVE_BFLOAT__)
template [[host_name("delta_recurrent_prefill_bf16")]] [[kernel]] decltype(delta_recurrent_prefill<bfloat, float>) delta_recurrent_prefill<bfloat, float>;
#endif

template <typename T, typename AccumT, ushort MAX_K = 256, ushort MAX_CHUNK = 24>
kernel void delta_chunk_step(
    constant size_t &batch_heads,
    constant size_t &chunk_size,
    constant size_t &k_head_dim,
    constant size_t &v_head_dim,
    device const T *prev_state,
    device const T *query,
    device const T *key,
    device const T *value,
    device const T *beta,
    device const T *g,
    device T *out,
    uint tid [[thread_index_in_threadgroup]],
    uint group_id [[threadgroup_position_in_grid]],
    uint threads_per_group [[threads_per_threadgroup]]
) {
    if (group_id >= batch_heads || chunk_size > MAX_CHUNK || k_head_dim > MAX_K) {
        return;
    }

    threadgroup AccumT tg_attn[MAX_CHUNK][MAX_CHUNK];
    threadgroup AccumT tg_local_attn[MAX_CHUNK][MAX_CHUNK];
    threadgroup AccumT tg_k_cumdecay[MAX_CHUNK][MAX_K];
    threadgroup AccumT tg_exp_g[MAX_CHUNK];
    threadgroup AccumT tg_row[MAX_CHUNK];
    threadgroup AccumT tg_state_decay[1];

    const size_t bh = group_id;
    const size_t state_stride = k_head_dim * v_head_dim;
    const size_t token_stride_k = chunk_size * k_head_dim;
    const size_t token_stride_v = chunk_size * v_head_dim;
    const size_t token_stride_s = chunk_size;
    const size_t query_base = bh * token_stride_k;
    const size_t key_base = bh * token_stride_k;
    const size_t value_base = bh * token_stride_v;
    const size_t scalar_base = bh * token_stride_s;
    const size_t out_base = bh * (chunk_size + k_head_dim) * v_head_dim;

    if (tid == 0) {
        AccumT g_cum[MAX_CHUNK];
        AccumT exp_g_last = AccumT(1);
        for (size_t t = 0; t < chunk_size; ++t) {
            const AccumT g_value = AccumT(g[scalar_base + t]);
            g_cum[t] = t == 0 ? g_value : g_cum[t - 1] + g_value;
            tg_exp_g[t] = exp(g_cum[t]);
            exp_g_last = tg_exp_g[t];
        }
        tg_state_decay[0] = exp_g_last;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (size_t index = tid; index < chunk_size * chunk_size; index += threads_per_group) {
        const size_t t = index / chunk_size;
        const size_t s = index - t * chunk_size;
        tg_attn[t][s] = AccumT(0);
        tg_local_attn[t][s] = AccumT(0);
    }
    for (size_t index = tid; index < chunk_size * k_head_dim; index += threads_per_group) {
        const size_t t = index / k_head_dim;
        const size_t k_idx = index - t * k_head_dim;
        tg_k_cumdecay[t][k_idx] = AccumT(0);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        tg_attn[0][0] = AccumT(1);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (size_t t = 1; t < chunk_size; ++t) {
        if (tid < t) {
            const size_t s = tid;
            AccumT dot_k = AccumT(0);
            AccumT dot_q = AccumT(0);
            const AccumT decay = tg_exp_g[t] / tg_exp_g[s];
            for (size_t k_idx = 0; k_idx < k_head_dim; ++k_idx) {
                const AccumT key_t = AccumT(key[key_base + t * k_head_dim + k_idx]);
                const AccumT key_s = AccumT(key[key_base + s * k_head_dim + k_idx]);
                const AccumT query_t = AccumT(query[query_base + t * k_head_dim + k_idx]);
                dot_k += key_t * AccumT(beta[scalar_base + t]) * key_s;
                dot_q += query_t * key_s;
            }
            tg_row[s] = -dot_k * decay;
            tg_local_attn[t][s] = dot_q * decay;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid < t) {
            const size_t s = tid;
            AccumT correction = AccumT(0);
            for (size_t m = 0; m < t; ++m) {
                correction += tg_row[m] * tg_attn[m][s];
            }
            tg_attn[t][s] = tg_row[s] + correction;
        } else if (tid == t) {
            tg_attn[t][t] = AccumT(1);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (size_t index = tid; index < chunk_size * k_head_dim; index += threads_per_group) {
        const size_t t = index / k_head_dim;
        const size_t k_idx = index - t * k_head_dim;
        AccumT acc = AccumT(0);
        for (size_t s = 0; s <= t; ++s) {
            const AccumT key_s = AccumT(key[key_base + s * k_head_dim + k_idx]);
            const AccumT beta_s = AccumT(beta[scalar_base + s]);
            acc += tg_attn[t][s] * key_s * beta_s * tg_exp_g[s];
        }
        tg_k_cumdecay[t][k_idx] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid >= v_head_dim) {
        return;
    }

    const size_t v_idx = tid;
    AccumT state[MAX_K];
    for (size_t k_idx = 0; k_idx < k_head_dim; ++k_idx) {
        state[k_idx] = AccumT(prev_state[bh * state_stride + k_idx * v_head_dim + v_idx]);
    }

    AccumT v_new[MAX_CHUNK];
    for (size_t t = 0; t < chunk_size; ++t) {
        AccumT solved_v = AccumT(0);
        for (size_t s = 0; s <= t; ++s) {
            solved_v += tg_attn[t][s]
                * AccumT(value[value_base + s * v_head_dim + v_idx])
                * AccumT(beta[scalar_base + s]);
        }

        AccumT v_prime = AccumT(0);
        AccumT attn_inter = AccumT(0);
        const size_t q_row = query_base + t * k_head_dim;
        const AccumT exp_g_t = tg_exp_g[t];
        size_t k_idx = 0;
        for (; k_idx + 3 < k_head_dim; k_idx += 4) {
            const AccumT state0 = state[k_idx];
            const AccumT state1 = state[k_idx + 1];
            const AccumT state2 = state[k_idx + 2];
            const AccumT state3 = state[k_idx + 3];

            v_prime += tg_k_cumdecay[t][k_idx] * state0
                + tg_k_cumdecay[t][k_idx + 1] * state1
                + tg_k_cumdecay[t][k_idx + 2] * state2
                + tg_k_cumdecay[t][k_idx + 3] * state3;

            attn_inter +=
                AccumT(query[q_row + k_idx]) * exp_g_t * state0
                + AccumT(query[q_row + k_idx + 1]) * exp_g_t * state1
                + AccumT(query[q_row + k_idx + 2]) * exp_g_t * state2
                + AccumT(query[q_row + k_idx + 3]) * exp_g_t * state3;
        }
        for (; k_idx < k_head_dim; ++k_idx) {
            v_prime += tg_k_cumdecay[t][k_idx] * state[k_idx];
            attn_inter += AccumT(query[q_row + k_idx]) * exp_g_t * state[k_idx];
        }

        v_new[t] = solved_v - v_prime;
        AccumT local = AccumT(0);
        for (size_t s = 0; s < t; ++s) {
            local += tg_local_attn[t][s] * v_new[s];
        }
        out[out_base + t * v_head_dim + v_idx] = T(attn_inter + local);
    }

    size_t k_idx = 0;
    for (; k_idx + 3 < k_head_dim; k_idx += 4) {
        AccumT update0 = AccumT(0);
        AccumT update1 = AccumT(0);
        AccumT update2 = AccumT(0);
        AccumT update3 = AccumT(0);
        for (size_t t = 0; t < chunk_size; ++t) {
            const AccumT scale = v_new[t] * (tg_state_decay[0] / tg_exp_g[t]);
            const size_t key_row = key_base + t * k_head_dim + k_idx;
            update0 += AccumT(key[key_row]) * scale;
            update1 += AccumT(key[key_row + 1]) * scale;
            update2 += AccumT(key[key_row + 2]) * scale;
            update3 += AccumT(key[key_row + 3]) * scale;
        }
        out[out_base + (chunk_size + k_idx) * v_head_dim + v_idx] =
            T(tg_state_decay[0] * state[k_idx] + update0);
        out[out_base + (chunk_size + k_idx + 1) * v_head_dim + v_idx] =
            T(tg_state_decay[0] * state[k_idx + 1] + update1);
        out[out_base + (chunk_size + k_idx + 2) * v_head_dim + v_idx] =
            T(tg_state_decay[0] * state[k_idx + 2] + update2);
        out[out_base + (chunk_size + k_idx + 3) * v_head_dim + v_idx] =
            T(tg_state_decay[0] * state[k_idx + 3] + update3);
    }
    for (; k_idx < k_head_dim; ++k_idx) {
        AccumT update = AccumT(0);
        for (size_t t = 0; t < chunk_size; ++t) {
            const AccumT weighted_key = AccumT(key[key_base + t * k_head_dim + k_idx])
                * (tg_state_decay[0] / tg_exp_g[t]);
            update += weighted_key * v_new[t];
        }
        out[out_base + (chunk_size + k_idx) * v_head_dim + v_idx] =
            T(tg_state_decay[0] * state[k_idx] + update);
    }
}

template [[host_name("delta_chunk_step_f16")]] [[kernel]] decltype(delta_chunk_step<half, float>) delta_chunk_step<half, float>;
template [[host_name("delta_chunk_step_f32")]] [[kernel]] decltype(delta_chunk_step<float, float>) delta_chunk_step<float, float>;
#if defined(__HAVE_BFLOAT__)
template [[host_name("delta_chunk_step_bf16")]] [[kernel]] decltype(delta_chunk_step<bfloat, float>) delta_chunk_step<bfloat, float>;
#endif


template <typename T, typename AccumT, ushort MAX_K = 256, ushort MAX_CHUNK = 24>
kernel void delta_chunk_step_windowed(
    constant size_t &batch_heads,
    constant size_t &chunk_size,
    constant size_t &k_head_dim,
    constant size_t &v_head_dim,
    constant size_t &prev_state_bh_stride,
    constant size_t &query_bh_stride,
    constant size_t &value_bh_stride,
    constant size_t &scalar_bh_stride,
    constant size_t &output_total_rows,
    constant size_t &output_token_row_offset,
    constant size_t &output_state_row_offset,
    device const T *prev_state,
    device const T *query,
    device const T *key,
    device const T *value,
    device const T *beta,
    device const T *g,
    device T *out,
    uint tid [[thread_index_in_threadgroup]],
    uint group_id [[threadgroup_position_in_grid]],
    uint threads_per_group [[threads_per_threadgroup]]
) {
    if (group_id >= batch_heads || chunk_size > MAX_CHUNK || k_head_dim > MAX_K) {
        return;
    }

    threadgroup AccumT tg_attn[MAX_CHUNK][MAX_CHUNK];
    threadgroup AccumT tg_local_attn[MAX_CHUNK][MAX_CHUNK];
    threadgroup AccumT tg_k_cumdecay[MAX_CHUNK][MAX_K];
    threadgroup AccumT tg_exp_g[MAX_CHUNK];
    threadgroup AccumT tg_row[MAX_CHUNK];
    threadgroup AccumT tg_state_decay[1];

    const size_t bh = group_id;
    const size_t query_base = bh * query_bh_stride;
    const size_t key_base = bh * query_bh_stride;
    const size_t value_base = bh * value_bh_stride;
    const size_t scalar_base = bh * scalar_bh_stride;
    const size_t prev_base = bh * prev_state_bh_stride;
    const size_t out_base = bh * output_total_rows * v_head_dim;

    if (tid == 0) {
        AccumT g_cum[MAX_CHUNK];
        AccumT exp_g_last = AccumT(1);
        for (size_t t = 0; t < chunk_size; ++t) {
            const AccumT g_value = AccumT(g[scalar_base + t]);
            g_cum[t] = t == 0 ? g_value : g_cum[t - 1] + g_value;
            tg_exp_g[t] = exp(g_cum[t]);
            exp_g_last = tg_exp_g[t];
        }
        tg_state_decay[0] = exp_g_last;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (size_t index = tid; index < chunk_size * chunk_size; index += threads_per_group) {
        const size_t t = index / chunk_size;
        const size_t s = index - t * chunk_size;
        tg_attn[t][s] = AccumT(0);
        tg_local_attn[t][s] = AccumT(0);
    }
    for (size_t index = tid; index < chunk_size * k_head_dim; index += threads_per_group) {
        const size_t t = index / k_head_dim;
        const size_t k_idx = index - t * k_head_dim;
        tg_k_cumdecay[t][k_idx] = AccumT(0);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        tg_attn[0][0] = AccumT(1);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (size_t t = 1; t < chunk_size; ++t) {
        if (tid < t) {
            const size_t s = tid;
            AccumT dot_k = AccumT(0);
            AccumT dot_q = AccumT(0);
            const AccumT decay = tg_exp_g[t] / tg_exp_g[s];
            for (size_t k_idx = 0; k_idx < k_head_dim; ++k_idx) {
                const AccumT key_t = AccumT(key[key_base + t * k_head_dim + k_idx]);
                const AccumT key_s = AccumT(key[key_base + s * k_head_dim + k_idx]);
                const AccumT query_t = AccumT(query[query_base + t * k_head_dim + k_idx]);
                dot_k += key_t * AccumT(beta[scalar_base + t]) * key_s;
                dot_q += query_t * key_s;
            }
            tg_row[s] = -dot_k * decay;
            tg_local_attn[t][s] = dot_q * decay;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid < t) {
            const size_t s = tid;
            AccumT correction = AccumT(0);
            for (size_t m = 0; m < t; ++m) {
                correction += tg_row[m] * tg_attn[m][s];
            }
            tg_attn[t][s] = tg_row[s] + correction;
        } else if (tid == t) {
            tg_attn[t][t] = AccumT(1);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (size_t index = tid; index < chunk_size * k_head_dim; index += threads_per_group) {
        const size_t t = index / k_head_dim;
        const size_t k_idx = index - t * k_head_dim;
        AccumT acc = AccumT(0);
        for (size_t s = 0; s <= t; ++s) {
            const AccumT key_s = AccumT(key[key_base + s * k_head_dim + k_idx]);
            const AccumT beta_s = AccumT(beta[scalar_base + s]);
            acc += tg_attn[t][s] * key_s * beta_s * tg_exp_g[s];
        }
        tg_k_cumdecay[t][k_idx] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid >= v_head_dim) {
        return;
    }

    const size_t v_idx = tid;
    AccumT state[MAX_K];
    for (size_t k_idx = 0; k_idx < k_head_dim; ++k_idx) {
        state[k_idx] = AccumT(prev_state[prev_base + k_idx * v_head_dim + v_idx]);
    }

    AccumT v_new[MAX_CHUNK];
    for (size_t t = 0; t < chunk_size; ++t) {
        AccumT solved_v = AccumT(0);
        for (size_t s = 0; s <= t; ++s) {
            solved_v += tg_attn[t][s]
                * AccumT(value[value_base + s * v_head_dim + v_idx])
                * AccumT(beta[scalar_base + s]);
        }

        AccumT v_prime = AccumT(0);
        AccumT attn_inter = AccumT(0);
        const size_t q_row = query_base + t * k_head_dim;
        const AccumT exp_g_t = tg_exp_g[t];
        size_t k_idx = 0;
        for (; k_idx + 3 < k_head_dim; k_idx += 4) {
            const AccumT state0 = state[k_idx];
            const AccumT state1 = state[k_idx + 1];
            const AccumT state2 = state[k_idx + 2];
            const AccumT state3 = state[k_idx + 3];

            v_prime += tg_k_cumdecay[t][k_idx] * state0
                + tg_k_cumdecay[t][k_idx + 1] * state1
                + tg_k_cumdecay[t][k_idx + 2] * state2
                + tg_k_cumdecay[t][k_idx + 3] * state3;

            attn_inter +=
                AccumT(query[q_row + k_idx]) * exp_g_t * state0
                + AccumT(query[q_row + k_idx + 1]) * exp_g_t * state1
                + AccumT(query[q_row + k_idx + 2]) * exp_g_t * state2
                + AccumT(query[q_row + k_idx + 3]) * exp_g_t * state3;
        }
        for (; k_idx < k_head_dim; ++k_idx) {
            v_prime += tg_k_cumdecay[t][k_idx] * state[k_idx];
            attn_inter += AccumT(query[q_row + k_idx]) * exp_g_t * state[k_idx];
        }

        v_new[t] = solved_v - v_prime;
        AccumT local = AccumT(0);
        for (size_t s = 0; s < t; ++s) {
            local += tg_local_attn[t][s] * v_new[s];
        }
        out[out_base + (output_token_row_offset + t) * v_head_dim + v_idx] = T(attn_inter + local);
    }

    size_t k_idx = 0;
    for (; k_idx + 3 < k_head_dim; k_idx += 4) {
        AccumT update0 = AccumT(0);
        AccumT update1 = AccumT(0);
        AccumT update2 = AccumT(0);
        AccumT update3 = AccumT(0);
        for (size_t t = 0; t < chunk_size; ++t) {
            const AccumT scale = v_new[t] * (tg_state_decay[0] / tg_exp_g[t]);
            const size_t key_row = key_base + t * k_head_dim + k_idx;
            update0 += AccumT(key[key_row]) * scale;
            update1 += AccumT(key[key_row + 1]) * scale;
            update2 += AccumT(key[key_row + 2]) * scale;
            update3 += AccumT(key[key_row + 3]) * scale;
        }
        out[out_base + (output_state_row_offset + k_idx) * v_head_dim + v_idx] =
            T(tg_state_decay[0] * state[k_idx] + update0);
        out[out_base + (output_state_row_offset + k_idx + 1) * v_head_dim + v_idx] =
            T(tg_state_decay[0] * state[k_idx + 1] + update1);
        out[out_base + (output_state_row_offset + k_idx + 2) * v_head_dim + v_idx] =
            T(tg_state_decay[0] * state[k_idx + 2] + update2);
        out[out_base + (output_state_row_offset + k_idx + 3) * v_head_dim + v_idx] =
            T(tg_state_decay[0] * state[k_idx + 3] + update3);
    }
    for (; k_idx < k_head_dim; ++k_idx) {
        AccumT update = AccumT(0);
        for (size_t t = 0; t < chunk_size; ++t) {
            const AccumT weighted_key = AccumT(key[key_base + t * k_head_dim + k_idx])
                * (tg_state_decay[0] / tg_exp_g[t]);
            update += weighted_key * v_new[t];
        }
        out[out_base + (output_state_row_offset + k_idx) * v_head_dim + v_idx] =
            T(tg_state_decay[0] * state[k_idx] + update);
    }
}

template [[host_name("delta_chunk_step_windowed_f16")]] [[kernel]] decltype(delta_chunk_step_windowed<half, float>) delta_chunk_step_windowed<half, float>;
template [[host_name("delta_chunk_step_windowed_f32")]] [[kernel]] decltype(delta_chunk_step_windowed<float, float>) delta_chunk_step_windowed<float, float>;
#if defined(__HAVE_BFLOAT__)
template [[host_name("delta_chunk_step_windowed_bf16")]] [[kernel]] decltype(delta_chunk_step_windowed<bfloat, float>) delta_chunk_step_windowed<bfloat, float>;
#endif

template <typename T, typename AccumT, ushort MAX_K = 256, ushort MAX_CHUNK = 24>
kernel void delta_chunk_readout(
    constant size_t &batch_heads,
    constant size_t &chunk_size,
    constant size_t &k_head_dim,
    constant size_t &v_head_dim,
    device const T *prev_state,
    device const T *query,
    device const T *key,
    device const T *value,
    device const T *beta,
    device const T *g,
    device T *out,
    uint tid [[thread_index_in_threadgroup]],
    uint group_id [[threadgroup_position_in_grid]],
    uint threads_per_group [[threads_per_threadgroup]]
) {
    if (group_id >= batch_heads || chunk_size > MAX_CHUNK || k_head_dim > MAX_K) {
        return;
    }

    threadgroup AccumT tg_attn[MAX_CHUNK][MAX_CHUNK];
    threadgroup AccumT tg_local_attn[MAX_CHUNK][MAX_CHUNK];
    threadgroup AccumT tg_k_cumdecay[MAX_CHUNK][MAX_K];
    threadgroup AccumT tg_exp_g[MAX_CHUNK];
    threadgroup AccumT tg_row[MAX_CHUNK];

    const size_t bh = group_id;
    const size_t state_stride = k_head_dim * v_head_dim;
    const size_t token_stride_k = chunk_size * k_head_dim;
    const size_t token_stride_v = chunk_size * v_head_dim;
    const size_t token_stride_s = chunk_size;
    const size_t query_base = bh * token_stride_k;
    const size_t key_base = bh * token_stride_k;
    const size_t value_base = bh * token_stride_v;
    const size_t scalar_base = bh * token_stride_s;
    const size_t out_base = bh * (2 * chunk_size) * v_head_dim;

    if (tid == 0) {
        AccumT g_cum[MAX_CHUNK];
        for (size_t t = 0; t < chunk_size; ++t) {
            const AccumT g_value = AccumT(g[scalar_base + t]);
            g_cum[t] = t == 0 ? g_value : g_cum[t - 1] + g_value;
            tg_exp_g[t] = exp(g_cum[t]);
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (size_t index = tid; index < chunk_size * chunk_size; index += threads_per_group) {
        const size_t t = index / chunk_size;
        const size_t s = index - t * chunk_size;
        tg_attn[t][s] = AccumT(0);
        tg_local_attn[t][s] = AccumT(0);
    }
    for (size_t index = tid; index < chunk_size * k_head_dim; index += threads_per_group) {
        const size_t t = index / k_head_dim;
        const size_t k_idx = index - t * k_head_dim;
        tg_k_cumdecay[t][k_idx] = AccumT(0);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        tg_attn[0][0] = AccumT(1);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (size_t t = 1; t < chunk_size; ++t) {
        if (tid < t) {
            const size_t s = tid;
            AccumT dot_k = AccumT(0);
            AccumT dot_q = AccumT(0);
            const AccumT decay = tg_exp_g[t] / tg_exp_g[s];
            for (size_t k_idx = 0; k_idx < k_head_dim; ++k_idx) {
                const AccumT key_t = AccumT(key[key_base + t * k_head_dim + k_idx]);
                const AccumT key_s = AccumT(key[key_base + s * k_head_dim + k_idx]);
                const AccumT query_t = AccumT(query[query_base + t * k_head_dim + k_idx]);
                dot_k += key_t * AccumT(beta[scalar_base + t]) * key_s;
                dot_q += query_t * key_s;
            }
            tg_row[s] = -dot_k * decay;
            tg_local_attn[t][s] = dot_q * decay;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid < t) {
            const size_t s = tid;
            AccumT correction = AccumT(0);
            for (size_t m = 0; m < t; ++m) {
                correction += tg_row[m] * tg_attn[m][s];
            }
            tg_attn[t][s] = tg_row[s] + correction;
        } else if (tid == t) {
            tg_attn[t][t] = AccumT(1);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (size_t index = tid; index < chunk_size * k_head_dim; index += threads_per_group) {
        const size_t t = index / k_head_dim;
        const size_t k_idx = index - t * k_head_dim;
        AccumT acc = AccumT(0);
        for (size_t s = 0; s <= t; ++s) {
            const AccumT key_s = AccumT(key[key_base + s * k_head_dim + k_idx]);
            const AccumT beta_s = AccumT(beta[scalar_base + s]);
            acc += tg_attn[t][s] * key_s * beta_s * tg_exp_g[s];
        }
        tg_k_cumdecay[t][k_idx] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid >= v_head_dim) {
        return;
    }

    const size_t v_idx = tid;
    AccumT state[MAX_K];
    for (size_t k_idx = 0; k_idx < k_head_dim; ++k_idx) {
        state[k_idx] = AccumT(prev_state[bh * state_stride + k_idx * v_head_dim + v_idx]);
    }

    AccumT v_new[MAX_CHUNK];
    for (size_t t = 0; t < chunk_size; ++t) {
        AccumT solved_v = AccumT(0);
        for (size_t s = 0; s <= t; ++s) {
            solved_v += tg_attn[t][s]
                * AccumT(value[value_base + s * v_head_dim + v_idx])
                * AccumT(beta[scalar_base + s]);
        }

        AccumT v_prime = AccumT(0);
        AccumT attn_inter = AccumT(0);
        const size_t q_row = query_base + t * k_head_dim;
        const AccumT exp_g_t = tg_exp_g[t];
        size_t k_idx = 0;
        for (; k_idx + 3 < k_head_dim; k_idx += 4) {
            const AccumT state0 = state[k_idx];
            const AccumT state1 = state[k_idx + 1];
            const AccumT state2 = state[k_idx + 2];
            const AccumT state3 = state[k_idx + 3];

            v_prime += tg_k_cumdecay[t][k_idx] * state0
                + tg_k_cumdecay[t][k_idx + 1] * state1
                + tg_k_cumdecay[t][k_idx + 2] * state2
                + tg_k_cumdecay[t][k_idx + 3] * state3;

            attn_inter +=
                AccumT(query[q_row + k_idx]) * exp_g_t * state0
                + AccumT(query[q_row + k_idx + 1]) * exp_g_t * state1
                + AccumT(query[q_row + k_idx + 2]) * exp_g_t * state2
                + AccumT(query[q_row + k_idx + 3]) * exp_g_t * state3;
        }
        for (; k_idx < k_head_dim; ++k_idx) {
            v_prime += tg_k_cumdecay[t][k_idx] * state[k_idx];
            attn_inter += AccumT(query[q_row + k_idx]) * exp_g_t * state[k_idx];
        }

        v_new[t] = solved_v - v_prime;
        AccumT local = AccumT(0);
        for (size_t s = 0; s < t; ++s) {
            local += tg_local_attn[t][s] * v_new[s];
        }
        out[out_base + t * v_head_dim + v_idx] = T(attn_inter + local);
        out[out_base + (chunk_size + t) * v_head_dim + v_idx] = T(v_new[t]);
    }
}

template [[host_name("delta_chunk_readout_f16")]] [[kernel]] decltype(delta_chunk_readout<half, float>) delta_chunk_readout<half, float>;
template [[host_name("delta_chunk_readout_f32")]] [[kernel]] decltype(delta_chunk_readout<float, float>) delta_chunk_readout<float, float>;
#if defined(__HAVE_BFLOAT__)
template [[host_name("delta_chunk_readout_bf16")]] [[kernel]] decltype(delta_chunk_readout<bfloat, float>) delta_chunk_readout<bfloat, float>;
#endif

template <typename T, typename AccumT, ushort MAX_K = 256, ushort MAX_CHUNK = 24>
kernel void delta_chunk_readout_split(
    constant size_t &batch_heads,
    constant size_t &chunk_size,
    constant size_t &k_head_dim,
    constant size_t &v_head_dim,
    device const T *prev_state,
    device const T *query,
    device const T *key,
    device const T *value,
    device const T *beta,
    device const T *g,
    device T *out,
    device T *v_new_out,
    uint tid [[thread_index_in_threadgroup]],
    uint group_id [[threadgroup_position_in_grid]],
    uint threads_per_group [[threads_per_threadgroup]]
) {
    if (group_id >= batch_heads || chunk_size > MAX_CHUNK || k_head_dim > MAX_K) {
        return;
    }

    threadgroup AccumT tg_attn[MAX_CHUNK][MAX_CHUNK];
    threadgroup AccumT tg_local_attn[MAX_CHUNK][MAX_CHUNK];
    threadgroup AccumT tg_k_cumdecay[MAX_CHUNK][MAX_K];
    threadgroup AccumT tg_exp_g[MAX_CHUNK];
    threadgroup AccumT tg_row[MAX_CHUNK];

    const size_t bh = group_id;
    const size_t state_stride = k_head_dim * v_head_dim;
    const size_t token_stride_k = chunk_size * k_head_dim;
    const size_t token_stride_v = chunk_size * v_head_dim;
    const size_t token_stride_s = chunk_size;
    const size_t query_base = bh * token_stride_k;
    const size_t key_base = bh * token_stride_k;
    const size_t value_base = bh * token_stride_v;
    const size_t scalar_base = bh * token_stride_s;
    const size_t out_base = bh * (chunk_size + k_head_dim) * v_head_dim;
    const size_t v_new_base = bh * chunk_size * v_head_dim;

    if (tid == 0) {
        AccumT g_cum[MAX_CHUNK];
        for (size_t t = 0; t < chunk_size; ++t) {
            const AccumT g_value = AccumT(g[scalar_base + t]);
            g_cum[t] = t == 0 ? g_value : g_cum[t - 1] + g_value;
            tg_exp_g[t] = exp(g_cum[t]);
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (size_t index = tid; index < chunk_size * chunk_size; index += threads_per_group) {
        const size_t t = index / chunk_size;
        const size_t s = index - t * chunk_size;
        tg_attn[t][s] = AccumT(0);
        tg_local_attn[t][s] = AccumT(0);
    }
    for (size_t index = tid; index < chunk_size * k_head_dim; index += threads_per_group) {
        const size_t t = index / k_head_dim;
        const size_t k_idx = index - t * k_head_dim;
        tg_k_cumdecay[t][k_idx] = AccumT(0);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        tg_attn[0][0] = AccumT(1);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (size_t t = 1; t < chunk_size; ++t) {
        if (tid < t) {
            const size_t s = tid;
            AccumT dot_k = AccumT(0);
            AccumT dot_q = AccumT(0);
            const AccumT decay = tg_exp_g[t] / tg_exp_g[s];
            for (size_t k_idx = 0; k_idx < k_head_dim; ++k_idx) {
                const AccumT key_t = AccumT(key[key_base + t * k_head_dim + k_idx]);
                const AccumT key_s = AccumT(key[key_base + s * k_head_dim + k_idx]);
                const AccumT query_t = AccumT(query[query_base + t * k_head_dim + k_idx]);
                dot_k += key_t * AccumT(beta[scalar_base + t]) * key_s;
                dot_q += query_t * key_s;
            }
            tg_row[s] = -dot_k * decay;
            tg_local_attn[t][s] = dot_q * decay;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid < t) {
            const size_t s = tid;
            AccumT correction = AccumT(0);
            for (size_t m = 0; m < t; ++m) {
                correction += tg_row[m] * tg_attn[m][s];
            }
            tg_attn[t][s] = tg_row[s] + correction;
        } else if (tid == t) {
            tg_attn[t][t] = AccumT(1);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (size_t index = tid; index < chunk_size * k_head_dim; index += threads_per_group) {
        const size_t t = index / k_head_dim;
        const size_t k_idx = index - t * k_head_dim;
        AccumT acc = AccumT(0);
        for (size_t s = 0; s <= t; ++s) {
            const AccumT key_s = AccumT(key[key_base + s * k_head_dim + k_idx]);
            const AccumT beta_s = AccumT(beta[scalar_base + s]);
            acc += tg_attn[t][s] * key_s * beta_s * tg_exp_g[s];
        }
        tg_k_cumdecay[t][k_idx] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid >= v_head_dim) {
        return;
    }

    const size_t v_idx = tid;
    AccumT state[MAX_K];
    for (size_t k_idx = 0; k_idx < k_head_dim; ++k_idx) {
        state[k_idx] = AccumT(prev_state[bh * state_stride + k_idx * v_head_dim + v_idx]);
    }

    AccumT v_new[MAX_CHUNK];
    for (size_t t = 0; t < chunk_size; ++t) {
        AccumT solved_v = AccumT(0);
        for (size_t s = 0; s <= t; ++s) {
            solved_v += tg_attn[t][s]
                * AccumT(value[value_base + s * v_head_dim + v_idx])
                * AccumT(beta[scalar_base + s]);
        }

        AccumT v_prime = AccumT(0);
        AccumT attn_inter = AccumT(0);
        const size_t q_row = query_base + t * k_head_dim;
        const AccumT exp_g_t = tg_exp_g[t];
        size_t k_idx = 0;
        for (; k_idx + 3 < k_head_dim; k_idx += 4) {
            const AccumT state0 = state[k_idx];
            const AccumT state1 = state[k_idx + 1];
            const AccumT state2 = state[k_idx + 2];
            const AccumT state3 = state[k_idx + 3];

            v_prime += tg_k_cumdecay[t][k_idx] * state0
                + tg_k_cumdecay[t][k_idx + 1] * state1
                + tg_k_cumdecay[t][k_idx + 2] * state2
                + tg_k_cumdecay[t][k_idx + 3] * state3;

            attn_inter +=
                AccumT(query[q_row + k_idx]) * exp_g_t * state0
                + AccumT(query[q_row + k_idx + 1]) * exp_g_t * state1
                + AccumT(query[q_row + k_idx + 2]) * exp_g_t * state2
                + AccumT(query[q_row + k_idx + 3]) * exp_g_t * state3;
        }
        for (; k_idx < k_head_dim; ++k_idx) {
            v_prime += tg_k_cumdecay[t][k_idx] * state[k_idx];
            attn_inter += AccumT(query[q_row + k_idx]) * exp_g_t * state[k_idx];
        }

        v_new[t] = solved_v - v_prime;
        AccumT local = AccumT(0);
        for (size_t s = 0; s < t; ++s) {
            local += tg_local_attn[t][s] * v_new[s];
        }
        out[out_base + t * v_head_dim + v_idx] = T(attn_inter + local);
        v_new_out[v_new_base + t * v_head_dim + v_idx] = T(v_new[t]);
    }
}

template [[host_name("delta_chunk_readout_split_f16")]] [[kernel]] decltype(delta_chunk_readout_split<half, float>) delta_chunk_readout_split<half, float>;
template [[host_name("delta_chunk_readout_split_f32")]] [[kernel]] decltype(delta_chunk_readout_split<float, float>) delta_chunk_readout_split<float, float>;
#if defined(__HAVE_BFLOAT__)
template [[host_name("delta_chunk_readout_split_bf16")]] [[kernel]] decltype(delta_chunk_readout_split<bfloat, float>) delta_chunk_readout_split<bfloat, float>;
#endif

template <typename T, typename AccumT, ushort MAX_CHUNK = 24>
kernel void delta_chunk_state_update_raw(
    constant size_t &batch_heads,
    constant size_t &chunk_size,
    constant size_t &k_head_dim,
    constant size_t &v_head_dim,
    device const T *prev_state,
    device const T *key,
    device const T *v_new,
    device const T *g,
    constant size_t &output_row_offset,
    device T *out,
    uint tid [[thread_position_in_grid]]
) {
    const size_t output_elems = batch_heads * k_head_dim * v_head_dim;
    if (tid >= output_elems || chunk_size > MAX_CHUNK) {
        return;
    }

    const size_t state_stride = k_head_dim * v_head_dim;
    const size_t bh = tid / state_stride;
    const size_t rem = tid - bh * state_stride;
    const size_t k_idx = rem / v_head_dim;
    const size_t v_idx = rem - k_idx * v_head_dim;
    const size_t key_base = bh * chunk_size * k_head_dim;
    const size_t v_new_base = bh * chunk_size * v_head_dim;
    const size_t scalar_base = bh * chunk_size;

    AccumT exp_g[MAX_CHUNK];
    AccumT g_cum = AccumT(0);
    for (size_t t = 0; t < chunk_size; ++t) {
        g_cum += AccumT(g[scalar_base + t]);
        exp_g[t] = exp(g_cum);
    }
    const AccumT state_decay = exp_g[chunk_size - 1];

    const size_t state_idx = bh * state_stride + k_idx * v_head_dim + v_idx;
    const size_t out_base = bh * (k_head_dim + output_row_offset) * v_head_dim;
    AccumT acc = state_decay * AccumT(prev_state[state_idx]);
    for (size_t t = 0; t < chunk_size; ++t) {
        const AccumT weighted_key =
            AccumT(key[key_base + t * k_head_dim + k_idx]) * (state_decay / exp_g[t]);
        const AccumT v_value = AccumT(v_new[v_new_base + t * v_head_dim + v_idx]);
        acc += weighted_key * v_value;
    }
    out[out_base + output_row_offset * v_head_dim + k_idx * v_head_dim + v_idx] = T(acc);
}

template [[host_name("delta_chunk_state_update_raw_f16")]] [[kernel]] decltype(delta_chunk_state_update_raw<half, float>) delta_chunk_state_update_raw<half, float>;
template [[host_name("delta_chunk_state_update_raw_f32")]] [[kernel]] decltype(delta_chunk_state_update_raw<float, float>) delta_chunk_state_update_raw<float, float>;
#if defined(__HAVE_BFLOAT__)
template [[host_name("delta_chunk_state_update_raw_bf16")]] [[kernel]] decltype(delta_chunk_state_update_raw<bfloat, float>) delta_chunk_state_update_raw<bfloat, float>;
#endif

template <typename T, typename AccumT, ushort MAX_K = 256, ushort MAX_CHUNK = 16, ushort MAX_V = 64, ushort MAX_K_WORKERS = 4>
kernel void delta_chunk_step_2d(
    constant size_t &batch_heads,
    constant size_t &chunk_size,
    constant size_t &k_head_dim,
    constant size_t &v_head_dim,
    device const T *prev_state,
    device const T *query,
    device const T *key,
    device const T *value,
    device const T *beta,
    device const T *g,
    device T *out,
    uint2 tid2 [[thread_position_in_threadgroup]],
    uint2 group_id [[threadgroup_position_in_grid]],
    uint2 threads_per_group [[threads_per_threadgroup]]
) {
    if (group_id.x >= batch_heads || chunk_size > MAX_CHUNK || k_head_dim > MAX_K || v_head_dim > MAX_V) {
        return;
    }

    const size_t linear_tid = tid2.y * threads_per_group.x + tid2.x;
    const size_t threads_in_group = threads_per_group.x * threads_per_group.y;
    const size_t bh = group_id.x;
    const size_t state_stride = k_head_dim * v_head_dim;
    const size_t token_stride_k = chunk_size * k_head_dim;
    const size_t token_stride_v = chunk_size * v_head_dim;
    const size_t token_stride_s = chunk_size;
    const size_t query_base = bh * token_stride_k;
    const size_t key_base = bh * token_stride_k;
    const size_t value_base = bh * token_stride_v;
    const size_t scalar_base = bh * token_stride_s;
    const size_t out_base = bh * (chunk_size + k_head_dim) * v_head_dim;
    const size_t k_workers = max((size_t)1, min((size_t)threads_per_group.y, (size_t)MAX_K_WORKERS));
    const size_t k_slice = (k_head_dim + k_workers - 1) / k_workers;
    const size_t k_begin = tid2.y * k_slice;
    const size_t k_end = min(k_head_dim, k_begin + k_slice);
    const bool active_v = tid2.x < v_head_dim;

    threadgroup AccumT tg_attn[MAX_CHUNK][MAX_CHUNK];
    threadgroup AccumT tg_local_attn[MAX_CHUNK][MAX_CHUNK];
    threadgroup AccumT tg_k_cumdecay[MAX_CHUNK][MAX_K];
    threadgroup AccumT tg_exp_g[MAX_CHUNK];
    threadgroup AccumT tg_row[MAX_CHUNK];
    threadgroup AccumT tg_state_decay[1];
    threadgroup AccumT tg_reduce_vprime[MAX_V][MAX_K_WORKERS];
    threadgroup AccumT tg_reduce_attn[MAX_V][MAX_K_WORKERS];
    threadgroup AccumT tg_v_new[MAX_CHUNK][MAX_V];

    if (tid2.x == 0 && tid2.y == 0) {
        AccumT g_cum[MAX_CHUNK];
        AccumT exp_g_last = AccumT(1);
        for (size_t t = 0; t < chunk_size; ++t) {
            const AccumT g_value = AccumT(g[scalar_base + t]);
            g_cum[t] = t == 0 ? g_value : g_cum[t - 1] + g_value;
            tg_exp_g[t] = exp(g_cum[t]);
            exp_g_last = tg_exp_g[t];
        }
        tg_state_decay[0] = exp_g_last;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (size_t index = linear_tid; index < chunk_size * chunk_size; index += threads_in_group) {
        const size_t t = index / chunk_size;
        const size_t s = index - t * chunk_size;
        tg_attn[t][s] = AccumT(0);
        tg_local_attn[t][s] = AccumT(0);
    }
    for (size_t index = linear_tid; index < chunk_size * k_head_dim; index += threads_in_group) {
        const size_t t = index / k_head_dim;
        const size_t k_idx = index - t * k_head_dim;
        tg_k_cumdecay[t][k_idx] = AccumT(0);
    }
    for (size_t index = linear_tid; index < chunk_size * v_head_dim; index += threads_in_group) {
        const size_t t = index / v_head_dim;
        const size_t v_idx = index - t * v_head_dim;
        tg_v_new[t][v_idx] = AccumT(0);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid2.x == 0 && tid2.y == 0) {
        tg_attn[0][0] = AccumT(1);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (size_t t = 1; t < chunk_size; ++t) {
        if (linear_tid < t) {
            const size_t s = linear_tid;
            AccumT dot_k = AccumT(0);
            AccumT dot_q = AccumT(0);
            const AccumT decay = tg_exp_g[t] / tg_exp_g[s];
            for (size_t k_idx = 0; k_idx < k_head_dim; ++k_idx) {
                const AccumT key_t = AccumT(key[key_base + t * k_head_dim + k_idx]);
                const AccumT key_s = AccumT(key[key_base + s * k_head_dim + k_idx]);
                const AccumT query_t = AccumT(query[query_base + t * k_head_dim + k_idx]);
                dot_k += key_t * AccumT(beta[scalar_base + t]) * key_s;
                dot_q += query_t * key_s;
            }
            tg_row[s] = -dot_k * decay;
            tg_local_attn[t][s] = dot_q * decay;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (linear_tid < t) {
            const size_t s = linear_tid;
            AccumT correction = AccumT(0);
            for (size_t m = 0; m < t; ++m) {
                correction += tg_row[m] * tg_attn[m][s];
            }
            tg_attn[t][s] = tg_row[s] + correction;
        } else if (linear_tid == t) {
            tg_attn[t][t] = AccumT(1);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (size_t index = linear_tid; index < chunk_size * k_head_dim; index += threads_in_group) {
        const size_t t = index / k_head_dim;
        const size_t k_idx = index - t * k_head_dim;
        AccumT acc = AccumT(0);
        for (size_t s = 0; s <= t; ++s) {
            const AccumT key_s = AccumT(key[key_base + s * k_head_dim + k_idx]);
            const AccumT beta_s = AccumT(beta[scalar_base + s]);
            acc += tg_attn[t][s] * key_s * beta_s * tg_exp_g[s];
        }
        tg_k_cumdecay[t][k_idx] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const size_t v_idx = tid2.x;
    const size_t local_k_count = k_end > k_begin ? (k_end - k_begin) : 0;
    AccumT state_local[MAX_K / MAX_K_WORKERS];
    if (active_v) {
        for (size_t local_k = 0; local_k < local_k_count; ++local_k) {
            const size_t k_idx = k_begin + local_k;
            state_local[local_k] =
                AccumT(prev_state[bh * state_stride + k_idx * v_head_dim + v_idx]);
        }
    }

    if (active_v) {
        for (size_t t = 0; t < chunk_size; ++t) {
            AccumT partial_v_prime = AccumT(0);
            AccumT partial_attn = AccumT(0);
            const size_t q_row = query_base + t * k_head_dim;
            const AccumT exp_g_t = tg_exp_g[t];
            for (size_t local_k = 0; local_k < local_k_count; ++local_k) {
                const size_t k_idx = k_begin + local_k;
                const AccumT state_value = state_local[local_k];
                partial_v_prime += tg_k_cumdecay[t][k_idx] * state_value;
                partial_attn += AccumT(query[q_row + k_idx]) * exp_g_t * state_value;
            }
            tg_reduce_vprime[v_idx][tid2.y] = partial_v_prime;
            tg_reduce_attn[v_idx][tid2.y] = partial_attn;
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (tid2.y == 0) {
                AccumT solved_v = AccumT(0);
                for (size_t s = 0; s <= t; ++s) {
                    solved_v += tg_attn[t][s]
                        * AccumT(value[value_base + s * v_head_dim + v_idx])
                        * AccumT(beta[scalar_base + s]);
                }

                AccumT v_prime = AccumT(0);
                AccumT attn_inter = AccumT(0);
                for (size_t worker = 0; worker < k_workers; ++worker) {
                    v_prime += tg_reduce_vprime[v_idx][worker];
                    attn_inter += tg_reduce_attn[v_idx][worker];
                }

                const AccumT v_new = solved_v - v_prime;
                tg_v_new[t][v_idx] = v_new;

                AccumT local = AccumT(0);
                for (size_t s = 0; s < t; ++s) {
                    local += tg_local_attn[t][s] * tg_v_new[s][v_idx];
                }
                out[out_base + t * v_head_dim + v_idx] = T(attn_inter + local);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        for (size_t local_k = 0; local_k < local_k_count; ++local_k) {
            const size_t k_idx = k_begin + local_k;
            AccumT update = AccumT(0);
            for (size_t t = 0; t < chunk_size; ++t) {
                const AccumT weighted_key = AccumT(key[key_base + t * k_head_dim + k_idx])
                    * (tg_state_decay[0] / tg_exp_g[t]);
                update += weighted_key * tg_v_new[t][v_idx];
            }
            out[out_base + (chunk_size + k_idx) * v_head_dim + v_idx] =
                T(tg_state_decay[0] * state_local[local_k] + update);
        }
    }
}

template [[host_name("delta_chunk_step_2d_f16")]] [[kernel]] decltype(delta_chunk_step_2d<half, float>) delta_chunk_step_2d<half, float>;
template [[host_name("delta_chunk_step_2d_f32")]] [[kernel]] decltype(delta_chunk_step_2d<float, float>) delta_chunk_step_2d<float, float>;
#if defined(__HAVE_BFLOAT__)
template [[host_name("delta_chunk_step_2d_bf16")]] [[kernel]] decltype(delta_chunk_step_2d<bfloat, float>) delta_chunk_step_2d<bfloat, float>;
#endif

template <typename T, typename AccumT, ushort MAX_K = 256, ushort MAX_CHUNK = 24, ushort MAX_V = 24, ushort MAX_K_WORKERS = 2>
kernel void delta_chunk_step_windowed_2d(
    constant size_t &batch_heads,
    constant size_t &chunk_size,
    constant size_t &k_head_dim,
    constant size_t &v_head_dim,
    constant size_t &prev_state_bh_stride,
    constant size_t &query_bh_stride,
    constant size_t &value_bh_stride,
    constant size_t &scalar_bh_stride,
    constant size_t &output_total_rows,
    constant size_t &output_token_row_offset,
    constant size_t &output_state_row_offset,
    device const T *prev_state,
    device const T *query,
    device const T *key,
    device const T *value,
    device const T *beta,
    device const T *g,
    device T *out,
    uint2 tid2 [[thread_position_in_threadgroup]],
    uint2 group_id [[threadgroup_position_in_grid]],
    uint2 threads_per_group [[threads_per_threadgroup]]
) {
    if (group_id.x >= batch_heads || chunk_size > MAX_CHUNK || k_head_dim > MAX_K) {
        return;
    }

    const size_t linear_tid = tid2.y * threads_per_group.x + tid2.x;
    const size_t threads_in_group = threads_per_group.x * threads_per_group.y;
    const size_t bh = group_id.x;
    const size_t v_tile_start = group_id.y * MAX_V;
    if (v_tile_start >= v_head_dim) {
        return;
    }
    const size_t v_tile_width = min((size_t)MAX_V, v_head_dim - v_tile_start);
    const size_t query_base = bh * query_bh_stride;
    const size_t key_base = bh * query_bh_stride;
    const size_t value_base = bh * value_bh_stride;
    const size_t scalar_base = bh * scalar_bh_stride;
    const size_t prev_base = bh * prev_state_bh_stride;
    const size_t out_base = bh * output_total_rows * v_head_dim;
    const size_t k_workers = max((size_t)1, min((size_t)threads_per_group.y, (size_t)MAX_K_WORKERS));
    const size_t k_slice = (k_head_dim + k_workers - 1) / k_workers;
    const size_t k_begin = tid2.y * k_slice;
    const size_t k_end = min(k_head_dim, k_begin + k_slice);
    const bool active_v = tid2.x < v_tile_width;

    threadgroup AccumT tg_attn[MAX_CHUNK][MAX_CHUNK];
    threadgroup AccumT tg_local_attn[MAX_CHUNK][MAX_CHUNK];
    threadgroup AccumT tg_k_cumdecay[MAX_CHUNK][MAX_K];
    threadgroup AccumT tg_exp_g[MAX_CHUNK];
    threadgroup AccumT tg_row[MAX_CHUNK];
    threadgroup AccumT tg_state_decay[1];
    threadgroup AccumT tg_reduce_vprime[MAX_V][MAX_K_WORKERS];
    threadgroup AccumT tg_reduce_attn[MAX_V][MAX_K_WORKERS];
    threadgroup AccumT tg_v_new[MAX_CHUNK][MAX_V];

    if (tid2.x == 0 && tid2.y == 0) {
        AccumT g_cum[MAX_CHUNK];
        AccumT exp_g_last = AccumT(1);
        for (size_t t = 0; t < chunk_size; ++t) {
            const AccumT g_value = AccumT(g[scalar_base + t]);
            g_cum[t] = t == 0 ? g_value : g_cum[t - 1] + g_value;
            tg_exp_g[t] = exp(g_cum[t]);
            exp_g_last = tg_exp_g[t];
        }
        tg_state_decay[0] = exp_g_last;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (size_t index = linear_tid; index < chunk_size * chunk_size; index += threads_in_group) {
        const size_t t = index / chunk_size;
        const size_t s = index - t * chunk_size;
        tg_attn[t][s] = AccumT(0);
        tg_local_attn[t][s] = AccumT(0);
    }
    for (size_t index = linear_tid; index < chunk_size * k_head_dim; index += threads_in_group) {
        const size_t t = index / k_head_dim;
        const size_t k_idx = index - t * k_head_dim;
        tg_k_cumdecay[t][k_idx] = AccumT(0);
    }
    for (size_t index = linear_tid; index < chunk_size * v_tile_width; index += threads_in_group) {
        const size_t t = index / v_tile_width;
        const size_t local_v_idx = index - t * v_tile_width;
        tg_v_new[t][local_v_idx] = AccumT(0);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid2.x == 0 && tid2.y == 0) {
        tg_attn[0][0] = AccumT(1);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (size_t t = 1; t < chunk_size; ++t) {
        if (linear_tid < t) {
            const size_t s = linear_tid;
            AccumT dot_k = AccumT(0);
            AccumT dot_q = AccumT(0);
            const AccumT decay = tg_exp_g[t] / tg_exp_g[s];
            for (size_t k_idx = 0; k_idx < k_head_dim; ++k_idx) {
                const AccumT key_t = AccumT(key[key_base + t * k_head_dim + k_idx]);
                const AccumT key_s = AccumT(key[key_base + s * k_head_dim + k_idx]);
                const AccumT query_t = AccumT(query[query_base + t * k_head_dim + k_idx]);
                dot_k += key_t * AccumT(beta[scalar_base + t]) * key_s;
                dot_q += query_t * key_s;
            }
            tg_row[s] = -dot_k * decay;
            tg_local_attn[t][s] = dot_q * decay;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (linear_tid < t) {
            const size_t s = linear_tid;
            AccumT correction = AccumT(0);
            for (size_t m = 0; m < t; ++m) {
                correction += tg_row[m] * tg_attn[m][s];
            }
            tg_attn[t][s] = tg_row[s] + correction;
        } else if (linear_tid == t) {
            tg_attn[t][t] = AccumT(1);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (size_t index = linear_tid; index < chunk_size * k_head_dim; index += threads_in_group) {
        const size_t t = index / k_head_dim;
        const size_t k_idx = index - t * k_head_dim;
        AccumT acc = AccumT(0);
        for (size_t s = 0; s <= t; ++s) {
            const AccumT key_s = AccumT(key[key_base + s * k_head_dim + k_idx]);
            const AccumT beta_s = AccumT(beta[scalar_base + s]);
            acc += tg_attn[t][s] * key_s * beta_s * tg_exp_g[s];
        }
        tg_k_cumdecay[t][k_idx] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const size_t local_v_idx = tid2.x;
    const size_t v_idx = v_tile_start + local_v_idx;
    const size_t local_k_count = k_end > k_begin ? (k_end - k_begin) : 0;
    AccumT state_local[MAX_K / MAX_K_WORKERS];
    if (active_v) {
        for (size_t local_k = 0; local_k < local_k_count; ++local_k) {
            const size_t k_idx = k_begin + local_k;
            state_local[local_k] = AccumT(prev_state[prev_base + k_idx * v_head_dim + v_idx]);
        }
    }

    if (active_v) {
        for (size_t t = 0; t < chunk_size; ++t) {
            AccumT partial_v_prime = AccumT(0);
            AccumT partial_attn = AccumT(0);
            const size_t q_row = query_base + t * k_head_dim;
            const AccumT exp_g_t = tg_exp_g[t];
            for (size_t local_k = 0; local_k < local_k_count; ++local_k) {
                const size_t k_idx = k_begin + local_k;
                const AccumT state_value = state_local[local_k];
                partial_v_prime += tg_k_cumdecay[t][k_idx] * state_value;
                partial_attn += AccumT(query[q_row + k_idx]) * exp_g_t * state_value;
            }
            tg_reduce_vprime[local_v_idx][tid2.y] = partial_v_prime;
            tg_reduce_attn[local_v_idx][tid2.y] = partial_attn;
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (tid2.y == 0) {
                AccumT solved_v = AccumT(0);
                for (size_t s = 0; s <= t; ++s) {
                    solved_v += tg_attn[t][s]
                        * AccumT(value[value_base + s * v_head_dim + v_idx])
                        * AccumT(beta[scalar_base + s]);
                }

                AccumT v_prime = AccumT(0);
                AccumT attn_inter = AccumT(0);
                for (size_t worker = 0; worker < k_workers; ++worker) {
                    v_prime += tg_reduce_vprime[local_v_idx][worker];
                    attn_inter += tg_reduce_attn[local_v_idx][worker];
                }

                const AccumT v_new = solved_v - v_prime;
                tg_v_new[t][local_v_idx] = v_new;

                AccumT local = AccumT(0);
                for (size_t s = 0; s < t; ++s) {
                    local += tg_local_attn[t][s] * tg_v_new[s][local_v_idx];
                }
                out[out_base + (output_token_row_offset + t) * v_head_dim + v_idx] =
                    T(attn_inter + local);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        for (size_t local_k = 0; local_k < local_k_count; ++local_k) {
            const size_t k_idx = k_begin + local_k;
            AccumT update = AccumT(0);
            for (size_t t = 0; t < chunk_size; ++t) {
                const AccumT weighted_key = AccumT(key[key_base + t * k_head_dim + k_idx])
                    * (tg_state_decay[0] / tg_exp_g[t]);
                update += weighted_key * tg_v_new[t][local_v_idx];
            }
            out[out_base + (output_state_row_offset + k_idx) * v_head_dim + v_idx] =
                T(tg_state_decay[0] * state_local[local_k] + update);
        }
    }
}

template [[host_name("delta_chunk_step_windowed_2d_f16")]] [[kernel]] decltype(delta_chunk_step_windowed_2d<half, float>) delta_chunk_step_windowed_2d<half, float>;
template [[host_name("delta_chunk_step_windowed_2d_f32")]] [[kernel]] decltype(delta_chunk_step_windowed_2d<float, float>) delta_chunk_step_windowed_2d<float, float>;
#if defined(__HAVE_BFLOAT__)
template [[host_name("delta_chunk_step_windowed_2d_bf16")]] [[kernel]] decltype(delta_chunk_step_windowed_2d<bfloat, float>) delta_chunk_step_windowed_2d<bfloat, float>;
#endif

template <typename T, typename AccumT, ushort MAX_K = 256, ushort MAX_CHUNK = 24>
kernel void delta_chunk_scan_raw(
    constant size_t &batch_heads,
    constant size_t &num_chunks,
    constant size_t &chunk_size,
    constant size_t &k_head_dim,
    constant size_t &v_head_dim,
    device const T *initial_state,
    device const T *query,
    device const T *key,
    device const T *value,
    device const T *beta,
    device const T *g,
    device T *out,
    uint tid [[thread_index_in_threadgroup]],
    uint group_id [[threadgroup_position_in_grid]],
    uint threads_per_group [[threads_per_threadgroup]]
) {
    if (group_id >= batch_heads || chunk_size > MAX_CHUNK || k_head_dim > MAX_K) {
        return;
    }

    threadgroup AccumT tg_attn[MAX_CHUNK][MAX_CHUNK];
    threadgroup AccumT tg_local_attn[MAX_CHUNK][MAX_CHUNK];
    threadgroup AccumT tg_k_cumdecay[MAX_CHUNK][MAX_K];
    threadgroup AccumT tg_exp_g[MAX_CHUNK];
    threadgroup AccumT tg_row[MAX_CHUNK];
    threadgroup AccumT tg_state_decay[1];

    const size_t bh = group_id;
    const size_t state_stride = k_head_dim * v_head_dim;
    const size_t chunk_stride_k = chunk_size * k_head_dim;
    const size_t chunk_stride_v = chunk_size * v_head_dim;
    const size_t chunk_stride_s = chunk_size;
    const size_t token_count = num_chunks * chunk_size;
    const size_t out_base = bh * (token_count + k_head_dim) * v_head_dim;

    AccumT state[MAX_K];
    if (tid < v_head_dim) {
        for (size_t k_idx = 0; k_idx < k_head_dim; ++k_idx) {
            state[k_idx] = AccumT(initial_state[bh * state_stride + k_idx * v_head_dim + tid]);
        }
    }

    for (size_t chunk = 0; chunk < num_chunks; ++chunk) {
        const size_t query_base = ((bh * num_chunks) + chunk) * chunk_stride_k;
        const size_t key_base = ((bh * num_chunks) + chunk) * chunk_stride_k;
        const size_t value_base = ((bh * num_chunks) + chunk) * chunk_stride_v;
        const size_t scalar_base = ((bh * num_chunks) + chunk) * chunk_stride_s;

        if (tid == 0) {
            AccumT g_cum[MAX_CHUNK];
            AccumT exp_g_last = AccumT(1);
            for (size_t t = 0; t < chunk_size; ++t) {
                const AccumT g_value = AccumT(g[scalar_base + t]);
                g_cum[t] = t == 0 ? g_value : g_cum[t - 1] + g_value;
                tg_exp_g[t] = exp(g_cum[t]);
                exp_g_last = tg_exp_g[t];
            }
            tg_state_decay[0] = exp_g_last;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (size_t index = tid; index < chunk_size * chunk_size; index += threads_per_group) {
            const size_t t = index / chunk_size;
            const size_t s = index - t * chunk_size;
            tg_attn[t][s] = AccumT(0);
            tg_local_attn[t][s] = AccumT(0);
        }
        for (size_t index = tid; index < chunk_size * k_head_dim; index += threads_per_group) {
            const size_t t = index / k_head_dim;
            const size_t k_idx = index - t * k_head_dim;
            tg_k_cumdecay[t][k_idx] = AccumT(0);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid == 0) {
            tg_attn[0][0] = AccumT(1);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (size_t t = 1; t < chunk_size; ++t) {
            if (tid < t) {
                const size_t s = tid;
                AccumT dot_k = AccumT(0);
                AccumT dot_q = AccumT(0);
                const AccumT decay = tg_exp_g[t] / tg_exp_g[s];
                for (size_t k_idx = 0; k_idx < k_head_dim; ++k_idx) {
                    const AccumT key_t = AccumT(key[key_base + t * k_head_dim + k_idx]);
                    const AccumT key_s = AccumT(key[key_base + s * k_head_dim + k_idx]);
                    const AccumT query_t = AccumT(query[query_base + t * k_head_dim + k_idx]);
                    dot_k += key_t * AccumT(beta[scalar_base + t]) * key_s;
                    dot_q += query_t * key_s;
                }
                tg_row[s] = -dot_k * decay;
                tg_local_attn[t][s] = dot_q * decay;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (tid < t) {
                const size_t s = tid;
                AccumT correction = AccumT(0);
                for (size_t m = 0; m < t; ++m) {
                    correction += tg_row[m] * tg_attn[m][s];
                }
                tg_attn[t][s] = tg_row[s] + correction;
            } else if (tid == t) {
                tg_attn[t][t] = AccumT(1);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        for (size_t index = tid; index < chunk_size * k_head_dim; index += threads_per_group) {
            const size_t t = index / k_head_dim;
            const size_t k_idx = index - t * k_head_dim;
            AccumT acc = AccumT(0);
            for (size_t s = 0; s <= t; ++s) {
                const AccumT key_s = AccumT(key[key_base + s * k_head_dim + k_idx]);
                const AccumT beta_s = AccumT(beta[scalar_base + s]);
                acc += tg_attn[t][s] * key_s * beta_s * tg_exp_g[s];
            }
            tg_k_cumdecay[t][k_idx] = acc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid < v_head_dim) {
            const size_t v_idx = tid;
            AccumT v_new[MAX_CHUNK];
            for (size_t t = 0; t < chunk_size; ++t) {
                AccumT solved_v = AccumT(0);
                for (size_t s = 0; s <= t; ++s) {
                    solved_v += tg_attn[t][s]
                        * AccumT(value[value_base + s * v_head_dim + v_idx])
                        * AccumT(beta[scalar_base + s]);
                }

                AccumT v_prime = AccumT(0);
                AccumT attn_inter = AccumT(0);
                for (size_t k_idx = 0; k_idx < k_head_dim; ++k_idx) {
                    v_prime += tg_k_cumdecay[t][k_idx] * state[k_idx];
                    attn_inter +=
                        AccumT(query[query_base + t * k_head_dim + k_idx]) * tg_exp_g[t] * state[k_idx];
                }

                v_new[t] = solved_v - v_prime;
                AccumT local = AccumT(0);
                for (size_t s = 0; s < t; ++s) {
                    local += tg_local_attn[t][s] * v_new[s];
                }
                out[out_base + (chunk * chunk_size + t) * v_head_dim + v_idx] = T(attn_inter + local);
            }

            for (size_t k_idx = 0; k_idx < k_head_dim; ++k_idx) {
                AccumT update = AccumT(0);
                for (size_t t = 0; t < chunk_size; ++t) {
                    const AccumT weighted_key =
                        AccumT(key[key_base + t * k_head_dim + k_idx]) * (tg_state_decay[0] / tg_exp_g[t]);
                    update += weighted_key * v_new[t];
                }
                state[k_idx] = tg_state_decay[0] * state[k_idx] + update;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid < v_head_dim) {
        const size_t v_idx = tid;
        const size_t state_out = out_base + token_count * v_head_dim;
        for (size_t k_idx = 0; k_idx < k_head_dim; ++k_idx) {
            out[state_out + k_idx * v_head_dim + v_idx] = T(state[k_idx]);
        }
    }
}

template [[host_name("delta_chunk_scan_raw_f16")]] [[kernel]] decltype(delta_chunk_scan_raw<half, float>) delta_chunk_scan_raw<half, float>;
template [[host_name("delta_chunk_scan_raw_f32")]] [[kernel]] decltype(delta_chunk_scan_raw<float, float>) delta_chunk_scan_raw<float, float>;
#if defined(__HAVE_BFLOAT__)
template [[host_name("delta_chunk_scan_raw_bf16")]] [[kernel]] decltype(delta_chunk_scan_raw<bfloat, float>) delta_chunk_scan_raw<bfloat, float>;
#endif

template <typename T, typename AccumT, ushort MAX_K = 256>
kernel void delta_state_scan(
    constant size_t &batch_heads,
    constant size_t &num_chunks,
    constant size_t &chunk_size,
    constant size_t &k_head_dim,
    constant size_t &v_head_dim,
    device const T *initial_state,
    device const T *packed_scan,
    device const T *value,
    device T *out,
    uint tid [[thread_position_in_grid]]
) {
    const size_t total_threads = batch_heads * v_head_dim;
    if (tid >= total_threads || k_head_dim > MAX_K) {
        return;
    }

    const size_t bh = tid / v_head_dim;
    const size_t v_idx = tid - bh * v_head_dim;
    const size_t state_stride = k_head_dim * v_head_dim;
    const size_t packed_width = 2 * k_head_dim + 1;

    AccumT state[MAX_K];
    for (size_t k = 0; k < k_head_dim; ++k) {
        const size_t idx = bh * state_stride + k * v_head_dim + v_idx;
        state[k] = AccumT(initial_state[idx]);
        out[idx] = T(state[k]);
    }

    for (size_t chunk = 0; chunk < num_chunks; ++chunk) {
        const size_t packed_chunk_base = ((bh * num_chunks) + chunk) * chunk_size * packed_width;
        const size_t value_chunk_base = ((bh * num_chunks) + chunk) * chunk_size * v_head_dim;
        const AccumT state_decay = AccumT(packed_scan[packed_chunk_base + 2 * k_head_dim]);
        AccumT update[MAX_K];
        for (size_t k = 0; k < k_head_dim; ++k) {
            update[k] = AccumT(0);
        }

        for (size_t t = 0; t < chunk_size; ++t) {
            const size_t packed_row = packed_chunk_base + t * packed_width;
            const size_t value_row = value_chunk_base + t * v_head_dim;
            AccumT v_prime = AccumT(0);
            for (size_t k = 0; k < k_head_dim; ++k) {
                v_prime += AccumT(packed_scan[packed_row + k_head_dim + k]) * state[k];
            }
            const AccumT v_new = AccumT(value[value_row + v_idx]) - v_prime;
            for (size_t k = 0; k < k_head_dim; ++k) {
                update[k] += AccumT(packed_scan[packed_row + k]) * v_new;
            }
        }

        const size_t out_chunk_base = ((bh * (num_chunks + 1)) + (chunk + 1)) * state_stride;
        for (size_t k = 0; k < k_head_dim; ++k) {
            state[k] = state_decay * state[k] + update[k];
            out[out_chunk_base + k * v_head_dim + v_idx] = T(state[k]);
        }
    }
}

template [[host_name("delta_state_scan_f16")]] [[kernel]] decltype(delta_state_scan<half, float>) delta_state_scan<half, float>;
template [[host_name("delta_state_scan_f32")]] [[kernel]] decltype(delta_state_scan<float, float>) delta_state_scan<float, float>;
#if defined(__HAVE_BFLOAT__)
template [[host_name("delta_state_scan_bf16")]] [[kernel]] decltype(delta_state_scan<bfloat, float>) delta_state_scan<bfloat, float>;
#endif

template <typename T, typename AccumT, ushort MAX_K = 256, ushort MAX_CHUNK = 64>
kernel void delta_chunk_fused(
    constant size_t &batch_heads,
    constant size_t &chunk_size,
    constant size_t &k_head_dim,
    constant size_t &v_head_dim,
    device const T *prev_state,
    device const T *packed_chunk,
    device const T *value,
    device T *out,
    uint tid [[thread_position_in_grid]]
) {
    const size_t total_threads = batch_heads * v_head_dim;
    if (tid >= total_threads || k_head_dim > MAX_K || chunk_size > MAX_CHUNK) {
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

    AccumT state[MAX_K];
    for (size_t k = 0; k < k_head_dim; ++k) {
        state[k] = AccumT(prev_state[bh * state_stride + k * v_head_dim + v_idx]);
    }

    AccumT v_new[MAX_CHUNK];
    AccumT attn_inter[MAX_CHUNK];
    for (size_t t = 0; t < chunk_size; ++t) {
        const size_t packed_row = packed_base + t * packed_width;
        AccumT vp = AccumT(0);
        AccumT ai = AccumT(0);
        for (size_t k = 0; k < k_head_dim; ++k) {
            vp += AccumT(packed_chunk[packed_row + k_head_dim + k]) * state[k];
            ai += AccumT(packed_chunk[packed_row + 2 * k_head_dim + k]) * state[k];
        }
        v_new[t] = AccumT(value[value_base + t * v_head_dim + v_idx]) - vp;
        attn_inter[t] = ai;
        out[out_base + t * v_head_dim + v_idx] = T(v_new[t]);
        out[out_base + (chunk_size + t) * v_head_dim + v_idx] = T(attn_inter[t]);
    }

    const AccumT state_decay = AccumT(packed_chunk[packed_base + 3 * k_head_dim]);
    for (size_t k = 0; k < k_head_dim; ++k) {
        AccumT update = AccumT(0);
        for (size_t t = 0; t < chunk_size; ++t) {
            const size_t packed_row = packed_base + t * packed_width;
            update += AccumT(packed_chunk[packed_row + k]) * v_new[t];
        }
        out[out_base + (2 * chunk_size + k) * v_head_dim + v_idx] = T(state_decay * state[k] + update);
    }
}

template [[host_name("delta_chunk_fused_f16")]] [[kernel]] decltype(delta_chunk_fused<half, float>) delta_chunk_fused<half, float>;
template [[host_name("delta_chunk_fused_f32")]] [[kernel]] decltype(delta_chunk_fused<float, float>) delta_chunk_fused<float, float>;
#if defined(__HAVE_BFLOAT__)
template [[host_name("delta_chunk_fused_bf16")]] [[kernel]] decltype(delta_chunk_fused<bfloat, float>) delta_chunk_fused<bfloat, float>;
#endif

template <typename T, typename AccumT, ushort MAX_K = 256, ushort MAX_CHUNK = 64>
kernel void delta_full_scan(
    constant size_t &batch_heads,
    constant size_t &num_chunks,
    constant size_t &chunk_size,
    constant size_t &k_head_dim,
    constant size_t &v_head_dim,
    device const T *initial_state,
    device const T *weighted_key_scan,
    device const T *k_cumdecay_scan,
    device const T *q_state_scan,
    device const T *local_attn_scan,
    device const T *state_decay_scan,
    device const T *value,
    device T *out,
    uint tid [[thread_position_in_grid]]
) {
    const size_t total_threads = batch_heads * v_head_dim;
    if (tid >= total_threads || k_head_dim > MAX_K || chunk_size > MAX_CHUNK) {
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

    AccumT state[MAX_K];
    for (size_t k = 0; k < k_head_dim; ++k) {
        state[k] = AccumT(initial_state[bh * state_stride + k * v_head_dim + v_idx]);
    }

    for (size_t chunk = 0; chunk < num_chunks; ++chunk) {
        const size_t chunk_scan = scan_base + chunk * chunk_size * k_head_dim;
        const size_t chunk_local = local_base + chunk * chunk_size * chunk_size;
        const size_t chunk_value = value_base + chunk * chunk_size * v_head_dim;
        AccumT v_new[MAX_CHUNK];
        AccumT attn_inter[MAX_CHUNK];
        for (size_t t = 0; t < chunk_size; ++t) {
            AccumT vp = AccumT(0);
            AccumT ai = AccumT(0);
            const size_t row = chunk_scan + t * k_head_dim;
            for (size_t k = 0; k < k_head_dim; ++k) {
                vp += AccumT(k_cumdecay_scan[row + k]) * state[k];
                ai += AccumT(q_state_scan[row + k]) * state[k];
            }
            v_new[t] = AccumT(value[chunk_value + t * v_head_dim + v_idx]) - vp;
            attn_inter[t] = ai;
        }
        for (size_t t = 0; t < chunk_size; ++t) {
            AccumT local = AccumT(0);
            for (size_t s = 0; s < chunk_size; ++s) {
                const size_t row = chunk_local + t * chunk_size;
                local += AccumT(local_attn_scan[row + s]) * v_new[s];
            }
            out[out_base + (chunk * chunk_size + t) * v_head_dim + v_idx] = T(attn_inter[t] + local);
        }

        const AccumT state_decay = AccumT(state_decay_scan[decay_base + chunk]);
        for (size_t k = 0; k < k_head_dim; ++k) {
            AccumT update = AccumT(0);
            for (size_t t = 0; t < chunk_size; ++t) {
                const size_t row = chunk_scan + t * k_head_dim;
                update += AccumT(weighted_key_scan[row + k]) * v_new[t];
            }
            state[k] = state_decay * state[k] + update;
        }
    }

    const size_t state_out = out_base + token_count * v_head_dim;
    for (size_t k = 0; k < k_head_dim; ++k) {
        out[state_out + k * v_head_dim + v_idx] = T(state[k]);
    }
}

template [[host_name("delta_full_scan_f16")]] [[kernel]] decltype(delta_full_scan<half, float>) delta_full_scan<half, float>;
template [[host_name("delta_full_scan_f32")]] [[kernel]] decltype(delta_full_scan<float, float>) delta_full_scan<float, float>;
#if defined(__HAVE_BFLOAT__)
template [[host_name("delta_full_scan_bf16")]] [[kernel]] decltype(delta_full_scan<bfloat, float>) delta_full_scan<bfloat, float>;
#endif
