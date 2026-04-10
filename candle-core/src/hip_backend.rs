use crate::backend::{BackendDevice, BackendStorage};
use crate::cpu_backend::CpuDevice;
use crate::op::{BinaryOpT, CmpOp, ReduceOp, UnaryOpT};
use crate::{CpuStorage, DType, Error, Layout, Result, Shape};
use std::ffi::{c_char, c_int, c_void, CStr};

#[derive(Debug, Clone)]
pub struct HipDevice {
    ordinal: usize,
}

#[derive(Debug, Clone)]
pub struct HipStorage {
    storage: CpuStorage,
    device: HipDevice,
}

impl HipStorage {
    pub fn wrap_cpu_storage(storage: CpuStorage, device: HipDevice) -> Self {
        Self { storage, device }
    }

    pub fn cpu_storage(&self) -> &CpuStorage {
        &self.storage
    }

    pub fn transfer_to_device(&self, dst: &HipDevice) -> Result<Self> {
        Ok(Self {
            storage: self.storage.clone(),
            device: dst.clone(),
        })
    }
}

fn wrap(storage: CpuStorage, device: &HipDevice) -> HipStorage {
    HipStorage {
        storage,
        device: device.clone(),
    }
}

impl BackendStorage for HipStorage {
    type Device = HipDevice;

    fn try_clone(&self, layout: &Layout) -> Result<Self> {
        Ok(wrap(self.storage.try_clone(layout)?, &self.device))
    }

    fn dtype(&self) -> DType {
        self.storage.dtype()
    }

    fn device(&self) -> &Self::Device {
        &self.device
    }

    fn const_set(&mut self, value: crate::scalar::Scalar, layout: &Layout) -> Result<()> {
        self.storage.const_set(value, layout)
    }

    fn to_cpu_storage(&self) -> Result<CpuStorage> {
        Ok(self.storage.clone())
    }

    fn affine(&self, layout: &Layout, mul: f64, add: f64) -> Result<Self> {
        Ok(wrap(self.storage.affine(layout, mul, add)?, &self.device))
    }

    fn powf(&self, layout: &Layout, alpha: f64) -> Result<Self> {
        Ok(wrap(self.storage.powf(layout, alpha)?, &self.device))
    }

    fn elu(&self, layout: &Layout, alpha: f64) -> Result<Self> {
        Ok(wrap(self.storage.elu(layout, alpha)?, &self.device))
    }

    fn reduce_op(&self, op: ReduceOp, layout: &Layout, dims: &[usize]) -> Result<Self> {
        Ok(wrap(
            self.storage.reduce_op(op, layout, dims)?,
            &self.device,
        ))
    }

    fn cmp(&self, op: CmpOp, rhs: &Self, lhs_layout: &Layout, rhs_layout: &Layout) -> Result<Self> {
        Ok(wrap(
            self.storage.cmp(op, &rhs.storage, lhs_layout, rhs_layout)?,
            &self.device,
        ))
    }

    fn to_dtype(&self, layout: &Layout, dtype: DType) -> Result<Self> {
        Ok(wrap(self.storage.to_dtype(layout, dtype)?, &self.device))
    }

    fn unary_impl<B: UnaryOpT>(&self, layout: &Layout) -> Result<Self> {
        Ok(wrap(self.storage.unary_impl::<B>(layout)?, &self.device))
    }

    fn binary_impl<B: BinaryOpT>(
        &self,
        rhs: &Self,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<Self> {
        Ok(wrap(
            self.storage
                .binary_impl::<B>(&rhs.storage, lhs_layout, rhs_layout)?,
            &self.device,
        ))
    }

    fn where_cond(
        &self,
        layout: &Layout,
        true_tensor: &Self,
        true_layout: &Layout,
        false_tensor: &Self,
        false_layout: &Layout,
    ) -> Result<Self> {
        Ok(wrap(
            self.storage.where_cond(
                layout,
                &true_tensor.storage,
                true_layout,
                &false_tensor.storage,
                false_layout,
            )?,
            &self.device,
        ))
    }

    fn conv1d(
        &self,
        layout: &Layout,
        kernel: &Self,
        kernel_layout: &Layout,
        params: &crate::conv::ParamsConv1D,
    ) -> Result<Self> {
        Ok(wrap(
            self.storage
                .conv1d(layout, &kernel.storage, kernel_layout, params)?,
            &self.device,
        ))
    }

    fn conv_transpose1d(
        &self,
        layout: &Layout,
        kernel: &Self,
        kernel_layout: &Layout,
        params: &crate::conv::ParamsConvTranspose1D,
    ) -> Result<Self> {
        Ok(wrap(
            self.storage
                .conv_transpose1d(layout, &kernel.storage, kernel_layout, params)?,
            &self.device,
        ))
    }

    fn conv2d(
        &self,
        layout: &Layout,
        kernel: &Self,
        kernel_layout: &Layout,
        params: &crate::conv::ParamsConv2D,
    ) -> Result<Self> {
        Ok(wrap(
            self.storage
                .conv2d(layout, &kernel.storage, kernel_layout, params)?,
            &self.device,
        ))
    }

    fn conv_transpose2d(
        &self,
        layout: &Layout,
        kernel: &Self,
        kernel_layout: &Layout,
        params: &crate::conv::ParamsConvTranspose2D,
    ) -> Result<Self> {
        Ok(wrap(
            self.storage
                .conv_transpose2d(layout, &kernel.storage, kernel_layout, params)?,
            &self.device,
        ))
    }

    fn avg_pool2d(
        &self,
        layout: &Layout,
        kernel_size: (usize, usize),
        stride: (usize, usize),
    ) -> Result<Self> {
        Ok(wrap(
            self.storage.avg_pool2d(layout, kernel_size, stride)?,
            &self.device,
        ))
    }

    fn max_pool2d(
        &self,
        layout: &Layout,
        kernel_size: (usize, usize),
        stride: (usize, usize),
    ) -> Result<Self> {
        Ok(wrap(
            self.storage.max_pool2d(layout, kernel_size, stride)?,
            &self.device,
        ))
    }

    fn upsample_nearest1d(&self, layout: &Layout, size: usize) -> Result<Self> {
        Ok(wrap(
            self.storage.upsample_nearest1d(layout, size)?,
            &self.device,
        ))
    }

    fn upsample_nearest2d(&self, layout: &Layout, h: usize, w: usize) -> Result<Self> {
        Ok(wrap(
            self.storage.upsample_nearest2d(layout, h, w)?,
            &self.device,
        ))
    }

    fn upsample_bilinear2d(
        &self,
        layout: &Layout,
        h: usize,
        w: usize,
        align_corners: bool,
        scale_h: Option<f64>,
        scale_w: Option<f64>,
    ) -> Result<Self> {
        Ok(wrap(
            self.storage
                .upsample_bilinear2d(layout, h, w, align_corners, scale_h, scale_w)?,
            &self.device,
        ))
    }

    fn gather(
        &self,
        layout: &Layout,
        indexes: &Self,
        indexes_layout: &Layout,
        dim: usize,
    ) -> Result<Self> {
        Ok(wrap(
            self.storage
                .gather(layout, &indexes.storage, indexes_layout, dim)?,
            &self.device,
        ))
    }

    fn scatter_set(
        &mut self,
        layout: &Layout,
        indexes: &Self,
        indexes_layout: &Layout,
        source: &Self,
        source_layout: &Layout,
        dim: usize,
    ) -> Result<()> {
        self.storage.scatter_set(
            layout,
            &indexes.storage,
            indexes_layout,
            &source.storage,
            source_layout,
            dim,
        )
    }

    fn scatter_add_set(
        &mut self,
        layout: &Layout,
        indexes: &Self,
        indexes_layout: &Layout,
        source: &Self,
        source_layout: &Layout,
        dim: usize,
    ) -> Result<()> {
        self.storage.scatter_add_set(
            layout,
            &indexes.storage,
            indexes_layout,
            &source.storage,
            source_layout,
            dim,
        )
    }

    fn index_select(
        &self,
        indexes: &Self,
        layout: &Layout,
        indexes_layout: &Layout,
        dim: usize,
    ) -> Result<Self> {
        Ok(wrap(
            self.storage
                .index_select(&indexes.storage, layout, indexes_layout, dim)?,
            &self.device,
        ))
    }

    fn index_add(
        &self,
        layout: &Layout,
        indexes: &Self,
        indexes_layout: &Layout,
        source: &Self,
        source_layout: &Layout,
        dim: usize,
    ) -> Result<Self> {
        Ok(wrap(
            self.storage.index_add(
                layout,
                &indexes.storage,
                indexes_layout,
                &source.storage,
                source_layout,
                dim,
            )?,
            &self.device,
        ))
    }

    fn matmul(
        &self,
        rhs: &Self,
        bmnk: (usize, usize, usize, usize),
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<Self> {
        Ok(wrap(
            self.storage
                .matmul(&rhs.storage, bmnk, lhs_layout, rhs_layout)?,
            &self.device,
        ))
    }

    fn copy_strided_src(&self, dst: &mut Self, src_offset: usize, layout: &Layout) -> Result<()> {
        self.storage
            .copy_strided_src(&mut dst.storage, src_offset, layout)
    }

    fn copy2d(
        &self,
        dst: &mut Self,
        d1: usize,
        d2: usize,
        src_stride1: usize,
        dst_stride1: usize,
        src_offset: usize,
        dst_offset: usize,
    ) -> Result<()> {
        self.storage.copy2d(
            &mut dst.storage,
            d1,
            d2,
            src_stride1,
            dst_stride1,
            src_offset,
            dst_offset,
        )
    }
}

impl BackendDevice for HipDevice {
    type Storage = HipStorage;

    fn new(ordinal: usize) -> Result<Self> {
        Ok(Self { ordinal })
    }

    fn location(&self) -> crate::DeviceLocation {
        crate::DeviceLocation::Hip {
            gpu_id: self.ordinal,
        }
    }

    fn same_device(&self, rhs: &Self) -> bool {
        self.ordinal == rhs.ordinal
    }

    fn zeros_impl(&self, shape: &Shape, dtype: DType) -> Result<Self::Storage> {
        Ok(wrap(CpuDevice.zeros_impl(shape, dtype)?, self))
    }

    unsafe fn alloc_uninit(&self, shape: &Shape, dtype: DType) -> Result<Self::Storage> {
        Ok(wrap(CpuDevice.alloc_uninit(shape, dtype)?, self))
    }

    fn storage_from_slice<T: crate::WithDType>(&self, data: &[T]) -> Result<Self::Storage> {
        Ok(wrap(CpuDevice.storage_from_slice(data)?, self))
    }

    fn storage_from_cpu_storage(&self, storage: &CpuStorage) -> Result<Self::Storage> {
        Ok(wrap(CpuDevice.storage_from_cpu_storage(storage)?, self))
    }

    fn storage_from_cpu_storage_owned(&self, storage: CpuStorage) -> Result<Self::Storage> {
        Ok(wrap(
            CpuDevice.storage_from_cpu_storage_owned(storage)?,
            self,
        ))
    }

    fn rand_uniform(&self, shape: &Shape, dtype: DType, lo: f64, up: f64) -> Result<Self::Storage> {
        Ok(wrap(CpuDevice.rand_uniform(shape, dtype, lo, up)?, self))
    }

    fn rand_normal(
        &self,
        shape: &Shape,
        dtype: DType,
        mean: f64,
        std: f64,
    ) -> Result<Self::Storage> {
        Ok(wrap(CpuDevice.rand_normal(shape, dtype, mean, std)?, self))
    }

    fn set_seed(&self, seed: u64) -> Result<()> {
        CpuDevice.set_seed(seed)
    }

    fn get_current_seed(&self) -> Result<u64> {
        CpuDevice.get_current_seed()
    }

    fn synchronize(&self) -> Result<()> {
        Ok(())
    }
}

pub mod ffi {
    use super::*;

    unsafe extern "C" {
        pub fn qwen35_hip_linear_prefill_conv_pack(
            dtype: c_int,
            batch_size: usize,
            conv_dim: usize,
            total_len: usize,
            seq_len: usize,
            kernel_size: usize,
            mixed_qkv: *const c_void,
            weights: *const c_void,
            out: *mut c_void,
        ) -> c_int;

        pub fn qwen35_hip_full_attention_prefill(
            dtype: c_int,
            batch_size: usize,
            q_heads: usize,
            kv_heads: usize,
            q_len: usize,
            kv_len: usize,
            head_dim: usize,
            num_kv_groups: usize,
            scale: f32,
            seqlen_offset: usize,
            query: *const c_void,
            key: *const c_void,
            value: *const c_void,
            out: *mut c_void,
        ) -> c_int;

        pub fn qwen35_hip_delta_recurrent_prefill(
            dtype: c_int,
            batch_heads: usize,
            seq_len: usize,
            k_head_dim: usize,
            v_head_dim: usize,
            initial_state: *const c_void,
            query: *const c_void,
            key: *const c_void,
            value: *const c_void,
            beta: *const c_void,
            g: *const c_void,
            out: *mut c_void,
        ) -> c_int;

        pub fn qwen35_hip_delta_chunk_step(
            dtype: c_int,
            batch_heads: usize,
            chunk_size: usize,
            k_head_dim: usize,
            v_head_dim: usize,
            prev_state: *const c_void,
            query: *const c_void,
            key: *const c_void,
            value: *const c_void,
            beta: *const c_void,
            g: *const c_void,
            out: *mut c_void,
        ) -> c_int;

        pub fn qwen35_hip_delta_chunk_windowed(
            dtype: c_int,
            batch_heads: usize,
            num_chunks: usize,
            chunk_size: usize,
            k_head_dim: usize,
            v_head_dim: usize,
            prev_state: *const c_void,
            query: *const c_void,
            key: *const c_void,
            value: *const c_void,
            beta: *const c_void,
            g: *const c_void,
            out: *mut c_void,
        ) -> c_int;

        pub fn qwen35_hip_delta_chunk_scan_raw(
            dtype: c_int,
            batch_heads: usize,
            num_chunks: usize,
            chunk_size: usize,
            k_head_dim: usize,
            v_head_dim: usize,
            initial_state: *const c_void,
            query: *const c_void,
            key: *const c_void,
            value: *const c_void,
            beta: *const c_void,
            g: *const c_void,
            out: *mut c_void,
        ) -> c_int;

        pub fn qwen35_hip_delta_state_scan(
            dtype: c_int,
            batch_heads: usize,
            num_chunks: usize,
            chunk_size: usize,
            k_head_dim: usize,
            v_head_dim: usize,
            initial_state: *const c_void,
            packed_scan: *const c_void,
            value: *const c_void,
            out: *mut c_void,
        ) -> c_int;

        pub fn qwen35_hip_delta_chunk_fused(
            dtype: c_int,
            batch_heads: usize,
            chunk_size: usize,
            k_head_dim: usize,
            v_head_dim: usize,
            prev_state: *const c_void,
            packed_chunk: *const c_void,
            value: *const c_void,
            out: *mut c_void,
        ) -> c_int;

        pub fn qwen35_hip_delta_full_scan(
            dtype: c_int,
            batch_heads: usize,
            num_chunks: usize,
            chunk_size: usize,
            k_head_dim: usize,
            v_head_dim: usize,
            initial_state: *const c_void,
            weighted_key_scan: *const c_void,
            k_cumdecay_scan: *const c_void,
            q_state_scan: *const c_void,
            local_attn_scan: *const c_void,
            state_decay_scan: *const c_void,
            value: *const c_void,
            out: *mut c_void,
        ) -> c_int;

        pub fn qwen35_hip_error_string(code: c_int) -> *const c_char;
    }
}

pub fn qwen35_dtype_code(dtype: DType) -> Result<c_int> {
    match dtype {
        DType::F16 => Ok(0),
        DType::F32 => Ok(1),
        DType::BF16 => Ok(2),
        other => crate::bail!("unsupported HIP dtype for qwen3.5 kernel: {other:?}"),
    }
}

pub fn qwen35_error(op: &'static str, code: c_int) -> Error {
    let detail = unsafe {
        let ptr = ffi::qwen35_hip_error_string(code);
        if ptr.is_null() {
            format!("{op} failed with HIP status {code}")
        } else {
            let text = CStr::from_ptr(ptr).to_string_lossy();
            format!("{op} failed with HIP status {code}: {text}")
        }
    };
    Error::Hip(Box::new(std::io::Error::other(detail))).bt()
}
