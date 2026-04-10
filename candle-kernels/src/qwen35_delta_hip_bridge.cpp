#include <hip/hip_runtime.h>
#include <limits.h>
#include <stddef.h>
#include <stdint.h>

#include "qwen35_delta.hip"

namespace {

class ScopedHipDevice {
   public:
    explicit ScopedHipDevice(size_t device_ordinal)
        : previous_device_(-1), changed_(false), status_(hipSuccess) {
        if(device_ordinal > static_cast<size_t>(INT_MAX)) {
            status_ = hipErrorInvalidDevice;
            return;
        }
        const int target_device = static_cast<int>(device_ordinal);
        status_ = hipGetDevice(&previous_device_);
        if(status_ != hipSuccess) {
            return;
        }
        if(previous_device_ != target_device) {
            status_ = hipSetDevice(target_device);
            if(status_ != hipSuccess) {
                return;
            }
            changed_ = true;
        }
    }

    ~ScopedHipDevice() {
        if(changed_) {
            hipSetDevice(previous_device_);
        }
    }

    hipError_t status() const {
        return status_;
    }

   private:
    int previous_device_;
    bool changed_;
    hipError_t status_;
};

template <typename T, typename Kernel>
int linear_prefill_conv_pack_host(
    size_t device_ordinal,
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
    ScopedHipDevice scoped_device(device_ordinal);
    hipError_t err = scoped_device.status();
    if(err != hipSuccess) return static_cast<int>(err);

    const size_t mixed_elems = batch_size * conv_dim * total_len;
    const size_t weight_elems = conv_dim * kernel_size;
    const size_t out_elems = batch_size * seq_len * conv_dim;
    const size_t mixed_bytes = mixed_elems * sizeof(T);
    const size_t weight_bytes = weight_elems * sizeof(T);
    const size_t out_bytes = out_elems * sizeof(T);

    T* d_mixed_qkv = nullptr;
    T* d_weights = nullptr;
    T* d_out = nullptr;
    err = hipSuccess;

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
    size_t device_ordinal,
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
    ScopedHipDevice scoped_device(device_ordinal);
    hipError_t err = scoped_device.status();
    if(err != hipSuccess) return static_cast<int>(err);

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
    err = hipSuccess;

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

template <typename T, typename Kernel>
int delta_recurrent_prefill_host(
    size_t device_ordinal,
    size_t batch_heads,
    size_t seq_len,
    size_t k_head_dim,
    size_t v_head_dim,
    const void* initial_state_host,
    const void* query_host,
    const void* key_host,
    const void* value_host,
    const void* beta_host,
    const void* g_host,
    void* out_host,
    Kernel kernel
) {
    ScopedHipDevice scoped_device(device_ordinal);
    hipError_t err = scoped_device.status();
    if(err != hipSuccess) return static_cast<int>(err);

    const size_t state_elems = batch_heads * k_head_dim * v_head_dim;
    const size_t qk_elems = batch_heads * seq_len * k_head_dim;
    const size_t value_elems = batch_heads * seq_len * v_head_dim;
    const size_t scalar_elems = batch_heads * seq_len;
    const size_t out_elems = batch_heads * (seq_len + k_head_dim) * v_head_dim;
    const size_t state_bytes = state_elems * sizeof(T);
    const size_t qk_bytes = qk_elems * sizeof(T);
    const size_t value_bytes = value_elems * sizeof(T);
    const size_t scalar_bytes = scalar_elems * sizeof(T);
    const size_t out_bytes = out_elems * sizeof(T);
    const size_t total_threads = batch_heads * v_head_dim;

    T* d_initial_state = nullptr;
    T* d_query = nullptr;
    T* d_key = nullptr;
    T* d_value = nullptr;
    T* d_beta = nullptr;
    T* d_g = nullptr;
    T* d_out = nullptr;
    err = hipSuccess;

    err = hipMalloc(reinterpret_cast<void**>(&d_initial_state), state_bytes);
    if(err != hipSuccess) goto cleanup;
    err = hipMalloc(reinterpret_cast<void**>(&d_query), qk_bytes);
    if(err != hipSuccess) goto cleanup;
    err = hipMalloc(reinterpret_cast<void**>(&d_key), qk_bytes);
    if(err != hipSuccess) goto cleanup;
    err = hipMalloc(reinterpret_cast<void**>(&d_value), value_bytes);
    if(err != hipSuccess) goto cleanup;
    err = hipMalloc(reinterpret_cast<void**>(&d_beta), scalar_bytes);
    if(err != hipSuccess) goto cleanup;
    err = hipMalloc(reinterpret_cast<void**>(&d_g), scalar_bytes);
    if(err != hipSuccess) goto cleanup;
    err = hipMalloc(reinterpret_cast<void**>(&d_out), out_bytes);
    if(err != hipSuccess) goto cleanup;

    err = hipMemcpy(d_initial_state, initial_state_host, state_bytes, hipMemcpyHostToDevice);
    if(err != hipSuccess) goto cleanup;
    err = hipMemcpy(d_query, query_host, qk_bytes, hipMemcpyHostToDevice);
    if(err != hipSuccess) goto cleanup;
    err = hipMemcpy(d_key, key_host, qk_bytes, hipMemcpyHostToDevice);
    if(err != hipSuccess) goto cleanup;
    err = hipMemcpy(d_value, value_host, value_bytes, hipMemcpyHostToDevice);
    if(err != hipSuccess) goto cleanup;
    err = hipMemcpy(d_beta, beta_host, scalar_bytes, hipMemcpyHostToDevice);
    if(err != hipSuccess) goto cleanup;
    err = hipMemcpy(d_g, g_host, scalar_bytes, hipMemcpyHostToDevice);
    if(err != hipSuccess) goto cleanup;

    {
        const int block = 256;
        const int grid = static_cast<int>((total_threads + block - 1) / block);
        hipLaunchKernelGGL(
            kernel,
            dim3(grid),
            dim3(block),
            0,
            0,
            batch_heads,
            seq_len,
            k_head_dim,
            v_head_dim,
            d_initial_state,
            d_query,
            d_key,
            d_value,
            d_beta,
            d_g,
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
    if(d_g != nullptr) {
        hipFree(d_g);
    }
    if(d_beta != nullptr) {
        hipFree(d_beta);
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
    if(d_initial_state != nullptr) {
        hipFree(d_initial_state);
    }
    return static_cast<int>(err);
}

template <typename T, typename Kernel>
int delta_chunk_step_host(
    size_t device_ordinal,
    size_t batch_heads,
    size_t chunk_size,
    size_t k_head_dim,
    size_t v_head_dim,
    const void* prev_state_host,
    const void* query_host,
    const void* key_host,
    const void* value_host,
    const void* beta_host,
    const void* g_host,
    void* out_host,
    Kernel kernel
) {
    ScopedHipDevice scoped_device(device_ordinal);
    hipError_t err = scoped_device.status();
    if(err != hipSuccess) return static_cast<int>(err);

    const size_t state_elems = batch_heads * k_head_dim * v_head_dim;
    const size_t qk_elems = batch_heads * chunk_size * k_head_dim;
    const size_t value_elems = batch_heads * chunk_size * v_head_dim;
    const size_t scalar_elems = batch_heads * chunk_size;
    const size_t out_elems = batch_heads * (chunk_size + k_head_dim) * v_head_dim;
    const size_t state_bytes = state_elems * sizeof(T);
    const size_t qk_bytes = qk_elems * sizeof(T);
    const size_t value_bytes = value_elems * sizeof(T);
    const size_t scalar_bytes = scalar_elems * sizeof(T);
    const size_t out_bytes = out_elems * sizeof(T);
    const size_t total_threads = batch_heads * v_head_dim;

    T* d_prev_state = nullptr;
    T* d_query = nullptr;
    T* d_key = nullptr;
    T* d_value = nullptr;
    T* d_beta = nullptr;
    T* d_g = nullptr;
    T* d_out = nullptr;
    err = hipSuccess;

    err = hipMalloc(reinterpret_cast<void**>(&d_prev_state), state_bytes);
    if(err != hipSuccess) goto cleanup;
    err = hipMalloc(reinterpret_cast<void**>(&d_query), qk_bytes);
    if(err != hipSuccess) goto cleanup;
    err = hipMalloc(reinterpret_cast<void**>(&d_key), qk_bytes);
    if(err != hipSuccess) goto cleanup;
    err = hipMalloc(reinterpret_cast<void**>(&d_value), value_bytes);
    if(err != hipSuccess) goto cleanup;
    err = hipMalloc(reinterpret_cast<void**>(&d_beta), scalar_bytes);
    if(err != hipSuccess) goto cleanup;
    err = hipMalloc(reinterpret_cast<void**>(&d_g), scalar_bytes);
    if(err != hipSuccess) goto cleanup;
    err = hipMalloc(reinterpret_cast<void**>(&d_out), out_bytes);
    if(err != hipSuccess) goto cleanup;

    err = hipMemcpy(d_prev_state, prev_state_host, state_bytes, hipMemcpyHostToDevice);
    if(err != hipSuccess) goto cleanup;
    err = hipMemcpy(d_query, query_host, qk_bytes, hipMemcpyHostToDevice);
    if(err != hipSuccess) goto cleanup;
    err = hipMemcpy(d_key, key_host, qk_bytes, hipMemcpyHostToDevice);
    if(err != hipSuccess) goto cleanup;
    err = hipMemcpy(d_value, value_host, value_bytes, hipMemcpyHostToDevice);
    if(err != hipSuccess) goto cleanup;
    err = hipMemcpy(d_beta, beta_host, scalar_bytes, hipMemcpyHostToDevice);
    if(err != hipSuccess) goto cleanup;
    err = hipMemcpy(d_g, g_host, scalar_bytes, hipMemcpyHostToDevice);
    if(err != hipSuccess) goto cleanup;

    {
        const int block = 256;
        const int grid = static_cast<int>((total_threads + block - 1) / block);
        hipLaunchKernelGGL(
            kernel,
            dim3(grid),
            dim3(block),
            0,
            0,
            batch_heads,
            chunk_size,
            k_head_dim,
            v_head_dim,
            d_prev_state,
            d_query,
            d_key,
            d_value,
            d_beta,
            d_g,
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
    if(d_g != nullptr) {
        hipFree(d_g);
    }
    if(d_beta != nullptr) {
        hipFree(d_beta);
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
    if(d_prev_state != nullptr) {
        hipFree(d_prev_state);
    }
    return static_cast<int>(err);
}

template <typename T, typename Kernel>
int delta_chunk_windowed_host(
    size_t device_ordinal,
    size_t batch_heads,
    size_t num_chunks,
    size_t chunk_size,
    size_t k_head_dim,
    size_t v_head_dim,
    const void* prev_state_host,
    const void* query_host,
    const void* key_host,
    const void* value_host,
    const void* beta_host,
    const void* g_host,
    void* out_host,
    Kernel kernel
) {
    ScopedHipDevice scoped_device(device_ordinal);
    hipError_t err = scoped_device.status();
    if(err != hipSuccess) return static_cast<int>(err);

    const size_t total_tokens = num_chunks * chunk_size;
    const size_t state_elems = batch_heads * k_head_dim * v_head_dim;
    const size_t qk_elems = batch_heads * total_tokens * k_head_dim;
    const size_t value_elems = batch_heads * total_tokens * v_head_dim;
    const size_t scalar_elems = batch_heads * total_tokens;
    const size_t out_elems = batch_heads * (total_tokens + k_head_dim) * v_head_dim;
    const size_t state_bytes = state_elems * sizeof(T);
    const size_t qk_bytes = qk_elems * sizeof(T);
    const size_t value_bytes = value_elems * sizeof(T);
    const size_t scalar_bytes = scalar_elems * sizeof(T);
    const size_t out_bytes = out_elems * sizeof(T);
    const size_t total_threads = batch_heads * v_head_dim;

    T* d_prev_state = nullptr;
    T* d_query = nullptr;
    T* d_key = nullptr;
    T* d_value = nullptr;
    T* d_beta = nullptr;
    T* d_g = nullptr;
    T* d_out = nullptr;
    err = hipSuccess;

    err = hipMalloc(reinterpret_cast<void**>(&d_prev_state), state_bytes);
    if(err != hipSuccess) goto cleanup;
    err = hipMalloc(reinterpret_cast<void**>(&d_query), qk_bytes);
    if(err != hipSuccess) goto cleanup;
    err = hipMalloc(reinterpret_cast<void**>(&d_key), qk_bytes);
    if(err != hipSuccess) goto cleanup;
    err = hipMalloc(reinterpret_cast<void**>(&d_value), value_bytes);
    if(err != hipSuccess) goto cleanup;
    err = hipMalloc(reinterpret_cast<void**>(&d_beta), scalar_bytes);
    if(err != hipSuccess) goto cleanup;
    err = hipMalloc(reinterpret_cast<void**>(&d_g), scalar_bytes);
    if(err != hipSuccess) goto cleanup;
    err = hipMalloc(reinterpret_cast<void**>(&d_out), out_bytes);
    if(err != hipSuccess) goto cleanup;

    err = hipMemcpy(d_prev_state, prev_state_host, state_bytes, hipMemcpyHostToDevice);
    if(err != hipSuccess) goto cleanup;
    err = hipMemcpy(d_query, query_host, qk_bytes, hipMemcpyHostToDevice);
    if(err != hipSuccess) goto cleanup;
    err = hipMemcpy(d_key, key_host, qk_bytes, hipMemcpyHostToDevice);
    if(err != hipSuccess) goto cleanup;
    err = hipMemcpy(d_value, value_host, value_bytes, hipMemcpyHostToDevice);
    if(err != hipSuccess) goto cleanup;
    err = hipMemcpy(d_beta, beta_host, scalar_bytes, hipMemcpyHostToDevice);
    if(err != hipSuccess) goto cleanup;
    err = hipMemcpy(d_g, g_host, scalar_bytes, hipMemcpyHostToDevice);
    if(err != hipSuccess) goto cleanup;

    {
        const int block = 256;
        const int grid = static_cast<int>((total_threads + block - 1) / block);
        hipLaunchKernelGGL(
            kernel,
            dim3(grid),
            dim3(block),
            0,
            0,
            batch_heads,
            num_chunks,
            chunk_size,
            k_head_dim,
            v_head_dim,
            d_prev_state,
            d_query,
            d_key,
            d_value,
            d_beta,
            d_g,
            d_out
        );
    }
    err = hipGetLastError();
    if(err != hipSuccess) goto cleanup;
    err = hipDeviceSynchronize();
    if(err != hipSuccess) goto cleanup;
    err = hipMemcpy(out_host, d_out, out_bytes, hipMemcpyDeviceToHost);

cleanup:
    if(d_out != nullptr) hipFree(d_out);
    if(d_g != nullptr) hipFree(d_g);
    if(d_beta != nullptr) hipFree(d_beta);
    if(d_value != nullptr) hipFree(d_value);
    if(d_key != nullptr) hipFree(d_key);
    if(d_query != nullptr) hipFree(d_query);
    if(d_prev_state != nullptr) hipFree(d_prev_state);
    return static_cast<int>(err);
}

template <typename T, typename Kernel>
int delta_state_scan_host(
    size_t device_ordinal,
    size_t batch_heads,
    size_t num_chunks,
    size_t chunk_size,
    size_t k_head_dim,
    size_t v_head_dim,
    const void* initial_state_host,
    const void* packed_scan_host,
    const void* value_host,
    void* out_host,
    Kernel kernel
) {
    ScopedHipDevice scoped_device(device_ordinal);
    hipError_t err = scoped_device.status();
    if(err != hipSuccess) return static_cast<int>(err);

    const size_t state_elems = batch_heads * k_head_dim * v_head_dim;
    const size_t packed_width = 2 * k_head_dim + 1;
    const size_t packed_elems = batch_heads * num_chunks * chunk_size * packed_width;
    const size_t value_elems = batch_heads * num_chunks * chunk_size * v_head_dim;
    const size_t out_elems = batch_heads * (num_chunks + 1) * k_head_dim * v_head_dim;
    const size_t state_bytes = state_elems * sizeof(T);
    const size_t packed_bytes = packed_elems * sizeof(T);
    const size_t value_bytes = value_elems * sizeof(T);
    const size_t out_bytes = out_elems * sizeof(T);
    const size_t total_threads = batch_heads * v_head_dim;

    T* d_initial_state = nullptr;
    T* d_packed_scan = nullptr;
    T* d_value = nullptr;
    T* d_out = nullptr;
    err = hipSuccess;

    err = hipMalloc(reinterpret_cast<void**>(&d_initial_state), state_bytes);
    if(err != hipSuccess) goto cleanup;
    err = hipMalloc(reinterpret_cast<void**>(&d_packed_scan), packed_bytes);
    if(err != hipSuccess) goto cleanup;
    err = hipMalloc(reinterpret_cast<void**>(&d_value), value_bytes);
    if(err != hipSuccess) goto cleanup;
    err = hipMalloc(reinterpret_cast<void**>(&d_out), out_bytes);
    if(err != hipSuccess) goto cleanup;

    err = hipMemcpy(d_initial_state, initial_state_host, state_bytes, hipMemcpyHostToDevice);
    if(err != hipSuccess) goto cleanup;
    err = hipMemcpy(d_packed_scan, packed_scan_host, packed_bytes, hipMemcpyHostToDevice);
    if(err != hipSuccess) goto cleanup;
    err = hipMemcpy(d_value, value_host, value_bytes, hipMemcpyHostToDevice);
    if(err != hipSuccess) goto cleanup;

    {
        const int block = 256;
        const int grid = static_cast<int>((total_threads + block - 1) / block);
        hipLaunchKernelGGL(
            kernel,
            dim3(grid),
            dim3(block),
            0,
            0,
            batch_heads,
            num_chunks,
            chunk_size,
            k_head_dim,
            v_head_dim,
            d_initial_state,
            d_packed_scan,
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
    if(d_out != nullptr) hipFree(d_out);
    if(d_value != nullptr) hipFree(d_value);
    if(d_packed_scan != nullptr) hipFree(d_packed_scan);
    if(d_initial_state != nullptr) hipFree(d_initial_state);
    return static_cast<int>(err);
}

template <typename T, typename Kernel>
int delta_chunk_fused_host(
    size_t device_ordinal,
    size_t batch_heads,
    size_t chunk_size,
    size_t k_head_dim,
    size_t v_head_dim,
    const void* prev_state_host,
    const void* packed_chunk_host,
    const void* value_host,
    void* out_host,
    Kernel kernel
) {
    ScopedHipDevice scoped_device(device_ordinal);
    hipError_t err = scoped_device.status();
    if(err != hipSuccess) return static_cast<int>(err);

    const size_t state_elems = batch_heads * k_head_dim * v_head_dim;
    const size_t packed_width = 3 * k_head_dim + 1;
    const size_t packed_elems = batch_heads * chunk_size * packed_width;
    const size_t value_elems = batch_heads * chunk_size * v_head_dim;
    const size_t out_elems = batch_heads * (2 * chunk_size + k_head_dim) * v_head_dim;
    const size_t state_bytes = state_elems * sizeof(T);
    const size_t packed_bytes = packed_elems * sizeof(T);
    const size_t value_bytes = value_elems * sizeof(T);
    const size_t out_bytes = out_elems * sizeof(T);
    const size_t total_threads = batch_heads * v_head_dim;

    T* d_prev_state = nullptr;
    T* d_packed_chunk = nullptr;
    T* d_value = nullptr;
    T* d_out = nullptr;
    err = hipSuccess;

    err = hipMalloc(reinterpret_cast<void**>(&d_prev_state), state_bytes);
    if(err != hipSuccess) goto cleanup;
    err = hipMalloc(reinterpret_cast<void**>(&d_packed_chunk), packed_bytes);
    if(err != hipSuccess) goto cleanup;
    err = hipMalloc(reinterpret_cast<void**>(&d_value), value_bytes);
    if(err != hipSuccess) goto cleanup;
    err = hipMalloc(reinterpret_cast<void**>(&d_out), out_bytes);
    if(err != hipSuccess) goto cleanup;

    err = hipMemcpy(d_prev_state, prev_state_host, state_bytes, hipMemcpyHostToDevice);
    if(err != hipSuccess) goto cleanup;
    err = hipMemcpy(d_packed_chunk, packed_chunk_host, packed_bytes, hipMemcpyHostToDevice);
    if(err != hipSuccess) goto cleanup;
    err = hipMemcpy(d_value, value_host, value_bytes, hipMemcpyHostToDevice);
    if(err != hipSuccess) goto cleanup;

    {
        const int block = 256;
        const int grid = static_cast<int>((total_threads + block - 1) / block);
        hipLaunchKernelGGL(
            kernel,
            dim3(grid),
            dim3(block),
            0,
            0,
            batch_heads,
            chunk_size,
            k_head_dim,
            v_head_dim,
            d_prev_state,
            d_packed_chunk,
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
    if(d_out != nullptr) hipFree(d_out);
    if(d_value != nullptr) hipFree(d_value);
    if(d_packed_chunk != nullptr) hipFree(d_packed_chunk);
    if(d_prev_state != nullptr) hipFree(d_prev_state);
    return static_cast<int>(err);
}

template <typename T, typename Kernel>
int delta_full_scan_host(
    size_t device_ordinal,
    size_t batch_heads,
    size_t num_chunks,
    size_t chunk_size,
    size_t k_head_dim,
    size_t v_head_dim,
    const void* initial_state_host,
    const void* weighted_key_scan_host,
    const void* k_cumdecay_scan_host,
    const void* q_state_scan_host,
    const void* local_attn_scan_host,
    const void* state_decay_scan_host,
    const void* value_host,
    void* out_host,
    Kernel kernel
) {
    ScopedHipDevice scoped_device(device_ordinal);
    hipError_t err = scoped_device.status();
    if(err != hipSuccess) return static_cast<int>(err);

    const size_t state_elems = batch_heads * k_head_dim * v_head_dim;
    const size_t token_count = num_chunks * chunk_size;
    const size_t scan_elems = batch_heads * num_chunks * chunk_size * k_head_dim;
    const size_t local_elems = batch_heads * num_chunks * chunk_size * chunk_size;
    const size_t decay_elems = batch_heads * num_chunks;
    const size_t value_elems = batch_heads * token_count * v_head_dim;
    const size_t out_elems = batch_heads * (token_count + k_head_dim) * v_head_dim;
    const size_t state_bytes = state_elems * sizeof(T);
    const size_t scan_bytes = scan_elems * sizeof(T);
    const size_t local_bytes = local_elems * sizeof(T);
    const size_t decay_bytes = decay_elems * sizeof(T);
    const size_t value_bytes = value_elems * sizeof(T);
    const size_t out_bytes = out_elems * sizeof(T);
    const size_t total_threads = batch_heads * v_head_dim;

    T* d_initial_state = nullptr;
    T* d_weighted_key_scan = nullptr;
    T* d_k_cumdecay_scan = nullptr;
    T* d_q_state_scan = nullptr;
    T* d_local_attn_scan = nullptr;
    T* d_state_decay_scan = nullptr;
    T* d_value = nullptr;
    T* d_out = nullptr;
    err = hipSuccess;

    err = hipMalloc(reinterpret_cast<void**>(&d_initial_state), state_bytes);
    if(err != hipSuccess) goto cleanup;
    err = hipMalloc(reinterpret_cast<void**>(&d_weighted_key_scan), scan_bytes);
    if(err != hipSuccess) goto cleanup;
    err = hipMalloc(reinterpret_cast<void**>(&d_k_cumdecay_scan), scan_bytes);
    if(err != hipSuccess) goto cleanup;
    err = hipMalloc(reinterpret_cast<void**>(&d_q_state_scan), scan_bytes);
    if(err != hipSuccess) goto cleanup;
    err = hipMalloc(reinterpret_cast<void**>(&d_local_attn_scan), local_bytes);
    if(err != hipSuccess) goto cleanup;
    err = hipMalloc(reinterpret_cast<void**>(&d_state_decay_scan), decay_bytes);
    if(err != hipSuccess) goto cleanup;
    err = hipMalloc(reinterpret_cast<void**>(&d_value), value_bytes);
    if(err != hipSuccess) goto cleanup;
    err = hipMalloc(reinterpret_cast<void**>(&d_out), out_bytes);
    if(err != hipSuccess) goto cleanup;

    err = hipMemcpy(d_initial_state, initial_state_host, state_bytes, hipMemcpyHostToDevice);
    if(err != hipSuccess) goto cleanup;
    err = hipMemcpy(d_weighted_key_scan, weighted_key_scan_host, scan_bytes, hipMemcpyHostToDevice);
    if(err != hipSuccess) goto cleanup;
    err = hipMemcpy(d_k_cumdecay_scan, k_cumdecay_scan_host, scan_bytes, hipMemcpyHostToDevice);
    if(err != hipSuccess) goto cleanup;
    err = hipMemcpy(d_q_state_scan, q_state_scan_host, scan_bytes, hipMemcpyHostToDevice);
    if(err != hipSuccess) goto cleanup;
    err = hipMemcpy(d_local_attn_scan, local_attn_scan_host, local_bytes, hipMemcpyHostToDevice);
    if(err != hipSuccess) goto cleanup;
    err = hipMemcpy(d_state_decay_scan, state_decay_scan_host, decay_bytes, hipMemcpyHostToDevice);
    if(err != hipSuccess) goto cleanup;
    err = hipMemcpy(d_value, value_host, value_bytes, hipMemcpyHostToDevice);
    if(err != hipSuccess) goto cleanup;

    {
        const int block = 256;
        const int grid = static_cast<int>((total_threads + block - 1) / block);
        hipLaunchKernelGGL(
            kernel,
            dim3(grid),
            dim3(block),
            0,
            0,
            batch_heads,
            num_chunks,
            chunk_size,
            k_head_dim,
            v_head_dim,
            d_initial_state,
            d_weighted_key_scan,
            d_k_cumdecay_scan,
            d_q_state_scan,
            d_local_attn_scan,
            d_state_decay_scan,
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
    if(d_out != nullptr) hipFree(d_out);
    if(d_value != nullptr) hipFree(d_value);
    if(d_state_decay_scan != nullptr) hipFree(d_state_decay_scan);
    if(d_local_attn_scan != nullptr) hipFree(d_local_attn_scan);
    if(d_q_state_scan != nullptr) hipFree(d_q_state_scan);
    if(d_k_cumdecay_scan != nullptr) hipFree(d_k_cumdecay_scan);
    if(d_weighted_key_scan != nullptr) hipFree(d_weighted_key_scan);
    if(d_initial_state != nullptr) hipFree(d_initial_state);
    return static_cast<int>(err);
}

}  // namespace

extern "C" int qwen35_hip_linear_prefill_conv_pack(
    int dtype,
    size_t device_ordinal,
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
                device_ordinal,
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
                device_ordinal,
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
                device_ordinal,
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
    size_t device_ordinal,
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
                device_ordinal,
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
                device_ordinal,
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
                device_ordinal,
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

extern "C" int qwen35_hip_delta_recurrent_prefill(
    int dtype,
    size_t device_ordinal,
    size_t batch_heads,
    size_t seq_len,
    size_t k_head_dim,
    size_t v_head_dim,
    const void* initial_state,
    const void* query,
    const void* key,
    const void* value,
    const void* beta,
    const void* g,
    void* out
) {
    switch(dtype) {
        case 0:
            return delta_recurrent_prefill_host<half>(
                device_ordinal,
                batch_heads,
                seq_len,
                k_head_dim,
                v_head_dim,
                initial_state,
                query,
                key,
                value,
                beta,
                g,
                out,
                delta_recurrent_prefill_f16
            );
        case 1:
            return delta_recurrent_prefill_host<float>(
                device_ordinal,
                batch_heads,
                seq_len,
                k_head_dim,
                v_head_dim,
                initial_state,
                query,
                key,
                value,
                beta,
                g,
                out,
                delta_recurrent_prefill_f32
            );
        case 2:
            return delta_recurrent_prefill_host<hip_bfloat16>(
                device_ordinal,
                batch_heads,
                seq_len,
                k_head_dim,
                v_head_dim,
                initial_state,
                query,
                key,
                value,
                beta,
                g,
                out,
                delta_recurrent_prefill_bf16
            );
        default:
            return static_cast<int>(hipErrorInvalidValue);
    }
}

extern "C" int qwen35_hip_delta_chunk_step(
    int dtype,
    size_t device_ordinal,
    size_t batch_heads,
    size_t chunk_size,
    size_t k_head_dim,
    size_t v_head_dim,
    const void* prev_state,
    const void* query,
    const void* key,
    const void* value,
    const void* beta,
    const void* g,
    void* out
) {
    switch(dtype) {
        case 0:
            return delta_chunk_step_host<half>(
                device_ordinal,
                batch_heads,
                chunk_size,
                k_head_dim,
                v_head_dim,
                prev_state,
                query,
                key,
                value,
                beta,
                g,
                out,
                delta_chunk_step_f16
            );
        case 1:
            return delta_chunk_step_host<float>(
                device_ordinal,
                batch_heads,
                chunk_size,
                k_head_dim,
                v_head_dim,
                prev_state,
                query,
                key,
                value,
                beta,
                g,
                out,
                delta_chunk_step_f32
            );
        case 2:
            return delta_chunk_step_host<hip_bfloat16>(
                device_ordinal,
                batch_heads,
                chunk_size,
                k_head_dim,
                v_head_dim,
                prev_state,
                query,
                key,
                value,
                beta,
                g,
                out,
                delta_chunk_step_bf16
            );
        default:
            return static_cast<int>(hipErrorInvalidValue);
    }
}

extern "C" int qwen35_hip_delta_chunk_windowed(
    int dtype,
    size_t device_ordinal,
    size_t batch_heads,
    size_t num_chunks,
    size_t chunk_size,
    size_t k_head_dim,
    size_t v_head_dim,
    const void* prev_state,
    const void* query,
    const void* key,
    const void* value,
    const void* beta,
    const void* g,
    void* out
) {
    switch(dtype) {
        case 0:
            return delta_chunk_windowed_host<half>(
                device_ordinal,
                batch_heads, num_chunks, chunk_size, k_head_dim, v_head_dim, prev_state, query,
                key, value, beta, g, out, delta_chunk_step_windowed_f16);
        case 1:
            return delta_chunk_windowed_host<float>(
                device_ordinal,
                batch_heads, num_chunks, chunk_size, k_head_dim, v_head_dim, prev_state, query,
                key, value, beta, g, out, delta_chunk_step_windowed_f32);
        case 2:
            return delta_chunk_windowed_host<hip_bfloat16>(
                device_ordinal,
                batch_heads, num_chunks, chunk_size, k_head_dim, v_head_dim, prev_state, query,
                key, value, beta, g, out, delta_chunk_step_windowed_bf16);
        default:
            return static_cast<int>(hipErrorInvalidValue);
    }
}

extern "C" int qwen35_hip_delta_chunk_scan_raw(
    int dtype,
    size_t device_ordinal,
    size_t batch_heads,
    size_t num_chunks,
    size_t chunk_size,
    size_t k_head_dim,
    size_t v_head_dim,
    const void* initial_state,
    const void* query,
    const void* key,
    const void* value,
    const void* beta,
    const void* g,
    void* out
) {
    switch(dtype) {
        case 0:
            return delta_chunk_windowed_host<half>(
                device_ordinal,
                batch_heads, num_chunks, chunk_size, k_head_dim, v_head_dim, initial_state, query,
                key, value, beta, g, out, delta_chunk_scan_raw_f16);
        case 1:
            return delta_chunk_windowed_host<float>(
                device_ordinal,
                batch_heads, num_chunks, chunk_size, k_head_dim, v_head_dim, initial_state, query,
                key, value, beta, g, out, delta_chunk_scan_raw_f32);
        case 2:
            return delta_chunk_windowed_host<hip_bfloat16>(
                device_ordinal,
                batch_heads, num_chunks, chunk_size, k_head_dim, v_head_dim, initial_state, query,
                key, value, beta, g, out, delta_chunk_scan_raw_bf16);
        default:
            return static_cast<int>(hipErrorInvalidValue);
    }
}

extern "C" int qwen35_hip_delta_state_scan(
    int dtype,
    size_t device_ordinal,
    size_t batch_heads,
    size_t num_chunks,
    size_t chunk_size,
    size_t k_head_dim,
    size_t v_head_dim,
    const void* initial_state,
    const void* packed_scan,
    const void* value,
    void* out
) {
    switch(dtype) {
        case 0:
            return delta_state_scan_host<half>(
                device_ordinal,
                batch_heads, num_chunks, chunk_size, k_head_dim, v_head_dim, initial_state,
                packed_scan, value, out, delta_state_scan_f16);
        case 1:
            return delta_state_scan_host<float>(
                device_ordinal,
                batch_heads, num_chunks, chunk_size, k_head_dim, v_head_dim, initial_state,
                packed_scan, value, out, delta_state_scan_f32);
        case 2:
            return delta_state_scan_host<hip_bfloat16>(
                device_ordinal,
                batch_heads, num_chunks, chunk_size, k_head_dim, v_head_dim, initial_state,
                packed_scan, value, out, delta_state_scan_bf16);
        default:
            return static_cast<int>(hipErrorInvalidValue);
    }
}

extern "C" int qwen35_hip_delta_chunk_fused(
    int dtype,
    size_t device_ordinal,
    size_t batch_heads,
    size_t chunk_size,
    size_t k_head_dim,
    size_t v_head_dim,
    const void* prev_state,
    const void* packed_chunk,
    const void* value,
    void* out
) {
    switch(dtype) {
        case 0:
            return delta_chunk_fused_host<half>(
                device_ordinal,
                batch_heads, chunk_size, k_head_dim, v_head_dim, prev_state, packed_chunk, value,
                out, delta_chunk_fused_f16);
        case 1:
            return delta_chunk_fused_host<float>(
                device_ordinal,
                batch_heads, chunk_size, k_head_dim, v_head_dim, prev_state, packed_chunk, value,
                out, delta_chunk_fused_f32);
        case 2:
            return delta_chunk_fused_host<hip_bfloat16>(
                device_ordinal,
                batch_heads, chunk_size, k_head_dim, v_head_dim, prev_state, packed_chunk, value,
                out, delta_chunk_fused_bf16);
        default:
            return static_cast<int>(hipErrorInvalidValue);
    }
}

extern "C" int qwen35_hip_delta_full_scan(
    int dtype,
    size_t device_ordinal,
    size_t batch_heads,
    size_t num_chunks,
    size_t chunk_size,
    size_t k_head_dim,
    size_t v_head_dim,
    const void* initial_state,
    const void* weighted_key_scan,
    const void* k_cumdecay_scan,
    const void* q_state_scan,
    const void* local_attn_scan,
    const void* state_decay_scan,
    const void* value,
    void* out
) {
    switch(dtype) {
        case 0:
            return delta_full_scan_host<half>(
                device_ordinal,
                batch_heads, num_chunks, chunk_size, k_head_dim, v_head_dim, initial_state,
                weighted_key_scan, k_cumdecay_scan, q_state_scan, local_attn_scan,
                state_decay_scan, value, out, delta_full_scan_f16);
        case 1:
            return delta_full_scan_host<float>(
                device_ordinal,
                batch_heads, num_chunks, chunk_size, k_head_dim, v_head_dim, initial_state,
                weighted_key_scan, k_cumdecay_scan, q_state_scan, local_attn_scan,
                state_decay_scan, value, out, delta_full_scan_f32);
        case 2:
            return delta_full_scan_host<hip_bfloat16>(
                device_ordinal,
                batch_heads, num_chunks, chunk_size, k_head_dim, v_head_dim, initial_state,
                weighted_key_scan, k_cumdecay_scan, q_state_scan, local_attn_scan,
                state_decay_scan, value, out, delta_full_scan_bf16);
        default:
            return static_cast<int>(hipErrorInvalidValue);
    }
}

extern "C" const char* qwen35_hip_error_string(int code) {
    return hipGetErrorString(static_cast<hipError_t>(code));
}
