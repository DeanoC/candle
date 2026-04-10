#include <hip/hip_runtime.h>
#include <stddef.h>
#include <stdint.h>

#include "qwen35_delta.hip"

namespace {

template <typename T, typename Kernel>
int linear_prefill_conv_pack_host(
    size_t batch_size,
    size_t conv_dim,
    size_t total_len,
    size_t seq_len,
    size_t kernel_size,
    const void* mixed_qkv_host,
    const void* weights_host,
    void* out_host,
    Kernel kernel
) {
    const size_t mixed_elems = batch_size * conv_dim * total_len;
    const size_t weight_elems = conv_dim * kernel_size;
    const size_t out_elems = batch_size * seq_len * conv_dim;
    const size_t mixed_bytes = mixed_elems * sizeof(T);
    const size_t weight_bytes = weight_elems * sizeof(T);
    const size_t out_bytes = out_elems * sizeof(T);

    T* d_mixed_qkv = nullptr;
    T* d_weights = nullptr;
    T* d_out = nullptr;
    hipError_t err = hipSuccess;

    err = hipMalloc(reinterpret_cast<void**>(&d_mixed_qkv), mixed_bytes);
    if(err != hipSuccess) goto cleanup;
    err = hipMalloc(reinterpret_cast<void**>(&d_weights), weight_bytes);
    if(err != hipSuccess) goto cleanup;
    err = hipMalloc(reinterpret_cast<void**>(&d_out), out_bytes);
    if(err != hipSuccess) goto cleanup;

    err = hipMemcpy(d_mixed_qkv, mixed_qkv_host, mixed_bytes, hipMemcpyHostToDevice);
    if(err != hipSuccess) goto cleanup;
    err = hipMemcpy(d_weights, weights_host, weight_bytes, hipMemcpyHostToDevice);
    if(err != hipSuccess) goto cleanup;

    {
        const int block = 256;
        const int grid = static_cast<int>((out_elems + block - 1) / block);
        hipLaunchKernelGGL(
            kernel,
            dim3(grid),
            dim3(block),
            0,
            0,
            batch_size,
            conv_dim,
            total_len,
            seq_len,
            kernel_size,
            d_mixed_qkv,
            d_weights,
            d_out
        );
    }
    err = hipGetLastError();
    if(err != hipSuccess) goto cleanup;
    err = hipDeviceSynchronize();
    if(err != hipSuccess) goto cleanup;
    err = hipMemcpy(out_host, d_out, out_bytes, hipMemcpyDeviceToHost);

cleanup:
    if(d_out != nullptr) {
        hipFree(d_out);
    }
    if(d_weights != nullptr) {
        hipFree(d_weights);
    }
    if(d_mixed_qkv != nullptr) {
        hipFree(d_mixed_qkv);
    }
    return static_cast<int>(err);
}

template <typename T, typename Kernel>
int full_attention_prefill_host(
    size_t batch_size,
    size_t q_heads,
    size_t kv_heads,
    size_t q_len,
    size_t kv_len,
    size_t head_dim,
    size_t num_kv_groups,
    float scale,
    size_t seqlen_offset,
    const void* query_host,
    const void* key_host,
    const void* value_host,
    void* out_host,
    Kernel kernel
) {
    const size_t query_elems = batch_size * q_heads * q_len * head_dim;
    const size_t kv_elems = batch_size * kv_heads * kv_len * head_dim;
    const size_t out_elems = query_elems;
    const size_t query_bytes = query_elems * sizeof(T);
    const size_t kv_bytes = kv_elems * sizeof(T);
    const size_t out_bytes = out_elems * sizeof(T);
    const size_t total_rows = batch_size * q_heads * q_len;

    T* d_query = nullptr;
    T* d_key = nullptr;
    T* d_value = nullptr;
    T* d_out = nullptr;
    hipError_t err = hipSuccess;

    err = hipMalloc(reinterpret_cast<void**>(&d_query), query_bytes);
    if(err != hipSuccess) goto cleanup;
    err = hipMalloc(reinterpret_cast<void**>(&d_key), kv_bytes);
    if(err != hipSuccess) goto cleanup;
    err = hipMalloc(reinterpret_cast<void**>(&d_value), kv_bytes);
    if(err != hipSuccess) goto cleanup;
    err = hipMalloc(reinterpret_cast<void**>(&d_out), out_bytes);
    if(err != hipSuccess) goto cleanup;

    err = hipMemcpy(d_query, query_host, query_bytes, hipMemcpyHostToDevice);
    if(err != hipSuccess) goto cleanup;
    err = hipMemcpy(d_key, key_host, kv_bytes, hipMemcpyHostToDevice);
    if(err != hipSuccess) goto cleanup;
    err = hipMemcpy(d_value, value_host, kv_bytes, hipMemcpyHostToDevice);
    if(err != hipSuccess) goto cleanup;

    {
        const int block = 256;
        const int grid = static_cast<int>((total_rows + block - 1) / block);
        hipLaunchKernelGGL(
            kernel,
            dim3(grid),
            dim3(block),
            0,
            0,
            batch_size,
            q_heads,
            kv_heads,
            q_len,
            kv_len,
            head_dim,
            num_kv_groups,
            scale,
            seqlen_offset,
            d_query,
            d_key,
            d_value,
            d_out
        );
    }
    err = hipGetLastError();
    if(err != hipSuccess) goto cleanup;
    err = hipDeviceSynchronize();
    if(err != hipSuccess) goto cleanup;
    err = hipMemcpy(out_host, d_out, out_bytes, hipMemcpyDeviceToHost);

cleanup:
    if(d_out != nullptr) {
        hipFree(d_out);
    }
    if(d_value != nullptr) {
        hipFree(d_value);
    }
    if(d_key != nullptr) {
        hipFree(d_key);
    }
    if(d_query != nullptr) {
        hipFree(d_query);
    }
    return static_cast<int>(err);
}

}  // namespace

extern "C" int qwen35_hip_linear_prefill_conv_pack(
    int dtype,
    size_t batch_size,
    size_t conv_dim,
    size_t total_len,
    size_t seq_len,
    size_t kernel_size,
    const void* mixed_qkv,
    const void* weights,
    void* out
) {
    switch(dtype) {
        case 0:
            return linear_prefill_conv_pack_host<half>(
                batch_size,
                conv_dim,
                total_len,
                seq_len,
                kernel_size,
                mixed_qkv,
                weights,
                out,
                linear_prefill_conv_pack_f16
            );
        case 1:
            return linear_prefill_conv_pack_host<float>(
                batch_size,
                conv_dim,
                total_len,
                seq_len,
                kernel_size,
                mixed_qkv,
                weights,
                out,
                linear_prefill_conv_pack_f32
            );
        case 2:
            return linear_prefill_conv_pack_host<hip_bfloat16>(
                batch_size,
                conv_dim,
                total_len,
                seq_len,
                kernel_size,
                mixed_qkv,
                weights,
                out,
                linear_prefill_conv_pack_bf16
            );
        default:
            return static_cast<int>(hipErrorInvalidValue);
    }
}

extern "C" int qwen35_hip_full_attention_prefill(
    int dtype,
    size_t batch_size,
    size_t q_heads,
    size_t kv_heads,
    size_t q_len,
    size_t kv_len,
    size_t head_dim,
    size_t num_kv_groups,
    float scale,
    size_t seqlen_offset,
    const void* query,
    const void* key,
    const void* value,
    void* out
) {
    switch(dtype) {
        case 0:
            return full_attention_prefill_host<half>(
                batch_size,
                q_heads,
                kv_heads,
                q_len,
                kv_len,
                head_dim,
                num_kv_groups,
                scale,
                seqlen_offset,
                query,
                key,
                value,
                out,
                full_attention_prefill_f16
            );
        case 1:
            return full_attention_prefill_host<float>(
                batch_size,
                q_heads,
                kv_heads,
                q_len,
                kv_len,
                head_dim,
                num_kv_groups,
                scale,
                seqlen_offset,
                query,
                key,
                value,
                out,
                full_attention_prefill_f32
            );
        case 2:
            return full_attention_prefill_host<hip_bfloat16>(
                batch_size,
                q_heads,
                kv_heads,
                q_len,
                kv_len,
                head_dim,
                num_kv_groups,
                scale,
                seqlen_offset,
                query,
                key,
                value,
                out,
                full_attention_prefill_bf16
            );
        default:
            return static_cast<int>(hipErrorInvalidValue);
    }
}

extern "C" const char* qwen35_hip_error_string(int code) {
    return hipGetErrorString(static_cast<hipError_t>(code));
}
