use crate::backend::{BackendDevice, BackendStorage};
use crate::cpu_backend::CpuDevice;
use crate::op::{BinaryOpT, CmpOp, ReduceOp, UnaryOpT};
use crate::{CpuStorage, DType, Error, Layout, Result, Shape};
use float8::F8E4M3;
use half::{bf16, f16};
use std::ffi::{c_char, c_int, c_void, CStr};
use std::ptr::{self, NonNull};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::sync::OnceLock;

const HIP_MEMCPY_HOST_TO_DEVICE: c_int = 1;
const HIP_MEMCPY_DEVICE_TO_HOST: c_int = 2;
const HIP_MEMCPY_DEVICE_TO_DEVICE: c_int = 3;

static HIP_H2D_BYTES: AtomicU64 = AtomicU64::new(0);
static HIP_D2H_BYTES: AtomicU64 = AtomicU64::new(0);
static HIP_D2D_BYTES: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct HipTransferCounters {
    pub host_to_device_bytes: u64,
    pub device_to_host_bytes: u64,
    pub device_to_device_bytes: u64,
}

pub fn reset_transfer_counters() {
    HIP_H2D_BYTES.store(0, Ordering::Relaxed);
    HIP_D2H_BYTES.store(0, Ordering::Relaxed);
    HIP_D2D_BYTES.store(0, Ordering::Relaxed);
}

pub fn transfer_counters() -> HipTransferCounters {
    HipTransferCounters {
        host_to_device_bytes: HIP_H2D_BYTES.load(Ordering::Relaxed),
        device_to_host_bytes: HIP_D2H_BYTES.load(Ordering::Relaxed),
        device_to_device_bytes: HIP_D2D_BYTES.load(Ordering::Relaxed),
    }
}

#[derive(Debug, Clone)]
pub struct HipDevice {
    ordinal: usize,
}

impl HipDevice {
    pub fn ordinal(&self) -> usize {
        self.ordinal
    }
}

#[derive(Debug)]
struct HipAllocation {
    ptr: Option<NonNull<c_void>>,
    len_bytes: usize,
    device_ordinal: usize,
}

unsafe impl Send for HipAllocation {}
unsafe impl Sync for HipAllocation {}

impl Drop for HipAllocation {
    fn drop(&mut self) {
        let Some(ptr) = self.ptr else {
            return;
        };
        let _scoped = ScopedHipDevice::new(self.device_ordinal);
        unsafe {
            let _ = hipFree(ptr.as_ptr());
        }
    }
}

#[derive(Debug, Clone)]
pub struct HipStorage {
    buffer: Arc<HipAllocation>,
    dtype: DType,
    elem_count: usize,
    device: HipDevice,
    cpu_cache: Arc<OnceLock<CpuStorage>>,
}

impl HipStorage {
    pub fn wrap_cpu_storage(storage: CpuStorage, device: HipDevice) -> Self {
        Self::from_cpu_storage(storage, &device).expect("failed to upload CpuStorage to HIP")
    }

    pub fn cpu_storage(&self) -> &CpuStorage {
        self.cpu_cache.get_or_init(|| {
            self.to_cpu_storage()
                .expect("failed to materialize HipStorage as CpuStorage")
        })
    }

    pub fn raw_device_ptr(&self) -> *mut c_void {
        self.buffer
            .ptr
            .map_or(ptr::null_mut(), |ptr| ptr.as_ptr())
    }

    pub fn raw_device_ptr_with_offset(&self, elem_offset: usize) -> Result<*mut c_void> {
        if elem_offset > self.elem_count {
            crate::bail!(
                "hip device pointer offset out of bounds: offset={} len={}",
                elem_offset,
                self.elem_count
            )
        }
        let elem_size = dtype_elem_size(self.dtype)?;
        let byte_offset = elem_offset
            .checked_mul(elem_size)
            .ok_or_else(|| Error::Msg("hip pointer offset overflow".into()).bt())?;
        Ok((self.raw_device_ptr() as *mut u8).wrapping_add(byte_offset) as *mut c_void)
    }

    pub fn elem_count(&self) -> usize {
        self.elem_count
    }

    pub fn len_bytes(&self) -> usize {
        self.buffer.len_bytes
    }

    pub fn transfer_to_device(&self, dst: &HipDevice) -> Result<Self> {
        if self.device.same_device(dst) {
            if self.len_bytes() == 0 {
                return Ok(Self {
                    buffer: Arc::new(HipAllocation {
                        ptr: None,
                        len_bytes: 0,
                        device_ordinal: dst.ordinal,
                    }),
                    dtype: self.dtype,
                    elem_count: self.elem_count,
                    device: dst.clone(),
                    cpu_cache: Arc::new(OnceLock::new()),
                });
            }
            let buffer = allocate_hip_buffer(dst.ordinal, self.len_bytes())?;
            hip_memcpy_device_to_device(
                dst.ordinal,
                buffer.ptr.map_or(ptr::null_mut(), |ptr| ptr.as_ptr()),
                self.raw_device_ptr(),
                self.len_bytes(),
            )?;
            Ok(Self {
                buffer,
                dtype: self.dtype,
                elem_count: self.elem_count,
                device: dst.clone(),
                cpu_cache: Arc::new(OnceLock::new()),
            })
        } else {
            let cpu = self.to_cpu_storage()?;
            Self::from_cpu_storage(cpu, dst)
        }
    }

    fn from_cpu_storage(storage: CpuStorage, device: &HipDevice) -> Result<Self> {
        let dtype = storage.dtype();
        let elem_count = cpu_storage_elem_count(&storage);
        let len_bytes = cpu_storage_len_bytes(&storage)?;
        let buffer = allocate_hip_buffer(device.ordinal, len_bytes)?;
        if len_bytes != 0 {
            hip_memcpy_host_to_device(
                device.ordinal,
                buffer.ptr,
                cpu_storage_as_ptr(&storage),
                len_bytes,
            )?;
        }
        Ok(Self {
            buffer,
            dtype,
            elem_count,
            device: device.clone(),
            cpu_cache: Arc::new(OnceLock::new()),
        })
    }

    fn alloc_uninit(device: &HipDevice, elem_count: usize, dtype: DType) -> Result<Self> {
        let len_bytes = elem_count
            .checked_mul(dtype_elem_size(dtype)?)
            .ok_or_else(|| Error::Msg("hip allocation size overflow".into()).bt())?;
        Ok(Self {
            buffer: allocate_hip_buffer(device.ordinal, len_bytes)?,
            dtype,
            elem_count,
            device: device.clone(),
            cpu_cache: Arc::new(OnceLock::new()),
        })
    }

    fn zeros(device: &HipDevice, elem_count: usize, dtype: DType) -> Result<Self> {
        let storage = Self::alloc_uninit(device, elem_count, dtype)?;
        if storage.len_bytes() != 0 {
            hip_memset(device.ordinal, storage.raw_device_ptr(), 0, storage.len_bytes())?;
        }
        Ok(storage)
    }

    fn replace_from_cpu_storage(&mut self, storage: CpuStorage) -> Result<()> {
        *self = Self::from_cpu_storage(storage, &self.device)?;
        Ok(())
    }

    fn cpu_fallback_map<F>(&self, f: F) -> Result<Self>
    where
        F: FnOnce(&CpuStorage) -> Result<CpuStorage>,
    {
        let cpu = self.to_cpu_storage()?;
        Self::from_cpu_storage(f(&cpu)?, &self.device)
    }

    fn cpu_fallback_map2<F>(&self, rhs: &Self, f: F) -> Result<Self>
    where
        F: FnOnce(&CpuStorage, &CpuStorage) -> Result<CpuStorage>,
    {
        let lhs = self.to_cpu_storage()?;
        let rhs = rhs.to_cpu_storage()?;
        Self::from_cpu_storage(f(&lhs, &rhs)?, &self.device)
    }
}

#[derive(Debug, Clone, Copy)]
struct HipGemmConfig {
    transa: c_int,
    transb: c_int,
    lda: c_int,
    ldb: c_int,
    ldc: c_int,
    stride_a: i64,
    stride_b: i64,
    stride_c: i64,
}

fn hip_gemm_config(
    (_b, m, n, k): (usize, usize, usize, usize),
    lhs_l: &Layout,
    rhs_l: &Layout,
) -> Result<HipGemmConfig> {
    let lhs_stride = lhs_l.stride();
    let rhs_stride = rhs_l.stride();
    let rhs_m1 = rhs_stride[rhs_stride.len() - 1];
    let rhs_m2 = rhs_stride[rhs_stride.len() - 2];
    let lhs_m1 = lhs_stride[lhs_stride.len() - 1];
    let lhs_m2 = lhs_stride[lhs_stride.len() - 2];

    let (lda, transa) = if (rhs_m1 == 1 || n == 1) && (rhs_m2 == n || k == 1) {
        (n as c_int, 0)
    } else if (rhs_m1 == k || n == 1) && (rhs_m2 == 1 || k == 1) {
        (k as c_int, 1)
    } else {
        crate::bail!(
            "hip matmul non-contiguous rhs layout: lhs={lhs_l:?} rhs={rhs_l:?} mnk=({m}, {n}, {k})"
        )
    };
    let (ldb, transb) = if (lhs_m1 == 1 || k == 1) && (lhs_m2 == k || m == 1) {
        (k as c_int, 0)
    } else if (lhs_m1 == m || k == 1) && (lhs_m2 == 1 || m == 1) {
        (m as c_int, 1)
    } else {
        crate::bail!(
            "hip matmul non-contiguous lhs layout: lhs={lhs_l:?} rhs={rhs_l:?} mnk=({m}, {n}, {k})"
        )
    };

    let stride_b: usize = match lhs_stride[..lhs_stride.len() - 2] {
        [s1, stride] if s1 == stride * lhs_l.dims()[1] => stride,
        [_, stride] if lhs_l.dims()[0] == 1 => stride,
        [stride, _] if lhs_l.dims()[1] == 1 => stride,
        [stride] => stride,
        [] => m * k,
        _ => {
            crate::bail!(
                "hip matmul unsupported lhs batch stride: lhs={lhs_l:?} rhs={rhs_l:?} mnk=({m}, {n}, {k})"
            )
        }
    };
    let stride_a: usize = match rhs_stride[..rhs_stride.len() - 2] {
        [s1, stride] if s1 == stride * rhs_l.dims()[1] => stride,
        [_, stride] if rhs_l.dims()[0] == 1 => stride,
        [stride, _] if rhs_l.dims()[1] == 1 => stride,
        [stride] => stride,
        [] => n * k,
        _ => {
            crate::bail!(
                "hip matmul unsupported rhs batch stride: lhs={lhs_l:?} rhs={rhs_l:?} mnk=({m}, {n}, {k})"
            )
        }
    };
    Ok(HipGemmConfig {
        transa,
        transb,
        lda,
        ldb,
        ldc: n as c_int,
        stride_a: stride_a as i64,
        stride_b: stride_b as i64,
        stride_c: (m * n) as i64,
    })
}

impl BackendStorage for HipStorage {
    type Device = HipDevice;

    fn try_clone(&self, layout: &Layout) -> Result<Self> {
        if let Some((start, _end)) = layout.contiguous_offsets() {
            let elem_count = layout.shape().elem_count();
            let storage = Self::alloc_uninit(&self.device, elem_count, self.dtype)?;
            let byte_len = elem_count
                .checked_mul(dtype_elem_size(self.dtype)?)
                .ok_or_else(|| Error::Msg("hip clone size overflow".into()).bt())?;
            if byte_len != 0 {
                hip_memcpy_device_to_device(
                    self.device.ordinal,
                    storage.raw_device_ptr(),
                    self.raw_device_ptr_with_offset(start)?,
                    byte_len,
                )?;
            }
            Ok(storage)
        } else {
            self.cpu_fallback_map(|cpu| cpu.try_clone(layout))
        }
    }

    fn dtype(&self) -> DType {
        self.dtype
    }

    fn device(&self) -> &Self::Device {
        &self.device
    }

    fn const_set(&mut self, value: crate::scalar::Scalar, layout: &Layout) -> Result<()> {
        let mut cpu = self.to_cpu_storage()?;
        cpu.const_set(value, layout)?;
        self.replace_from_cpu_storage(cpu)
    }

    fn to_cpu_storage(&self) -> Result<CpuStorage> {
        if let Some(cpu) = self.cpu_cache.get() {
            return Ok(cpu.clone());
        }
        let mut storage = zeroed_cpu_storage(self.dtype, self.elem_count)?;
        if self.len_bytes() != 0 {
            hip_memcpy_device_to_host(
                self.device.ordinal,
                cpu_storage_as_mut_ptr(&mut storage),
                self.raw_device_ptr(),
                self.len_bytes(),
            )?;
        }
        Ok(storage)
    }

    fn affine(&self, layout: &Layout, mul: f64, add: f64) -> Result<Self> {
        let Some((start, _end)) = layout.contiguous_offsets() else {
            return self.cpu_fallback_map(|cpu| cpu.affine(layout, mul, add));
        };
        let dtype = match hip_dtype_code(self.dtype) {
            Ok(code) => code,
            Err(_) => return self.cpu_fallback_map(|cpu| cpu.affine(layout, mul, add)),
        };
        let output = Self::alloc_uninit(&self.device, layout.shape().elem_count(), self.dtype)?;
        let status = unsafe {
            ffi::candle_hip_affine_contiguous(
                dtype,
                self.device.ordinal,
                layout.shape().elem_count(),
                self.raw_device_ptr_with_offset(start)?,
                mul as f32,
                add as f32,
                output.raw_device_ptr(),
            )
        };
        if status != 0 {
            return Err(qwen35_error("hip-affine", status));
        }
        Ok(output)
    }

    fn powf(&self, layout: &Layout, alpha: f64) -> Result<Self> {
        self.cpu_fallback_map(|cpu| cpu.powf(layout, alpha))
    }

    fn elu(&self, layout: &Layout, alpha: f64) -> Result<Self> {
        self.cpu_fallback_map(|cpu| cpu.elu(layout, alpha))
    }

    fn reduce_op(&self, op: ReduceOp, layout: &Layout, dims: &[usize]) -> Result<Self> {
        if dims.is_empty() || !layout.is_contiguous() {
            return self.cpu_fallback_map(|cpu| cpu.reduce_op(op, layout, dims));
        }
        let src_dims = layout.shape().dims();
        let reduce_start = src_dims
            .len()
            .checked_sub(dims.len())
            .ok_or_else(|| Error::Msg("invalid HIP reduce dims".into()).bt())?;
        if !dims
            .iter()
            .enumerate()
            .all(|(idx, dim)| *dim == reduce_start + idx)
        {
            return self.cpu_fallback_map(|cpu| cpu.reduce_op(op, layout, dims));
        }
        if layout.shape().elem_count() == 0 && matches!(op, ReduceOp::Min | ReduceOp::Max) {
            return self.cpu_fallback_map(|cpu| cpu.reduce_op(op, layout, dims));
        }
        let dtype = match hip_dtype_code(self.dtype) {
            Ok(code) => code,
            Err(_) => return self.cpu_fallback_map(|cpu| cpu.reduce_op(op, layout, dims)),
        };
        let op_code = match op {
            ReduceOp::Sum => 0,
            ReduceOp::Min => 1,
            ReduceOp::Max => 2,
            _ => return self.cpu_fallback_map(|cpu| cpu.reduce_op(op, layout, dims)),
        };
        let n_rows = src_dims[..reduce_start]
            .iter()
            .copied()
            .product::<usize>()
            .max(1);
        let n_cols = src_dims[reduce_start..]
            .iter()
            .copied()
            .product::<usize>()
            .max(1);
        let output = Self::alloc_uninit(&self.device, n_rows, self.dtype)?;
        let status = unsafe {
            ffi::candle_hip_reduce_contiguous(
                op_code,
                dtype,
                self.device.ordinal,
                n_rows,
                n_cols,
                self.raw_device_ptr_with_offset(layout.start_offset())?,
                output.raw_device_ptr(),
            )
        };
        if status != 0 {
            return Err(qwen35_error("hip-reduce", status));
        }
        Ok(output)
    }

    fn cmp(&self, op: CmpOp, rhs: &Self, lhs_layout: &Layout, rhs_layout: &Layout) -> Result<Self> {
        self.cpu_fallback_map2(rhs, |lhs, rhs| lhs.cmp(op, rhs, lhs_layout, rhs_layout))
    }

    fn to_dtype(&self, layout: &Layout, dtype: DType) -> Result<Self> {
        if dtype == self.dtype {
            return self.try_clone(layout);
        }
        let Some((start, _end)) = layout.contiguous_offsets() else {
            return self.cpu_fallback_map(|cpu| cpu.to_dtype(layout, dtype));
        };
        let src_dtype = match hip_dtype_code(self.dtype) {
            Ok(code) => code,
            Err(_) => return self.cpu_fallback_map(|cpu| cpu.to_dtype(layout, dtype)),
        };
        let dst_dtype = match hip_dtype_code(dtype) {
            Ok(code) => code,
            Err(_) => return self.cpu_fallback_map(|cpu| cpu.to_dtype(layout, dtype)),
        };
        let output = Self::alloc_uninit(&self.device, layout.shape().elem_count(), dtype)?;
        let status = unsafe {
            ffi::candle_hip_cast_contiguous(
                src_dtype,
                dst_dtype,
                self.device.ordinal,
                layout.shape().elem_count(),
                self.raw_device_ptr_with_offset(start)?,
                output.raw_device_ptr(),
            )
        };
        if status != 0 {
            return Err(qwen35_error("hip-cast", status));
        }
        Ok(output)
    }

    fn unary_impl<B: UnaryOpT>(&self, layout: &Layout) -> Result<Self> {
        self.cpu_fallback_map(|cpu| cpu.unary_impl::<B>(layout))
    }

    fn binary_impl<B: BinaryOpT>(
        &self,
        rhs: &Self,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<Self> {
        self.cpu_fallback_map2(rhs, |lhs, rhs| lhs.binary_impl::<B>(rhs, lhs_layout, rhs_layout))
    }

    fn where_cond(
        &self,
        layout: &Layout,
        true_tensor: &Self,
        true_layout: &Layout,
        false_tensor: &Self,
        false_layout: &Layout,
    ) -> Result<Self> {
        if true_tensor.dtype != false_tensor.dtype {
            return self.cpu_fallback_map2(true_tensor, |_cond, _true_tensor| {
                let cond = self.to_cpu_storage()?;
                let t = true_tensor.to_cpu_storage()?;
                let f = false_tensor.to_cpu_storage()?;
                cond.where_cond(layout, &t, true_layout, &f, false_layout)
            });
        }
        let (
            Some((cond_start, _)),
            Some((true_start, _)),
            Some((false_start, _)),
        ) = (
            layout.contiguous_offsets(),
            true_layout.contiguous_offsets(),
            false_layout.contiguous_offsets(),
        )
        else {
            let cond = self.to_cpu_storage()?;
            let t = true_tensor.to_cpu_storage()?;
            let f = false_tensor.to_cpu_storage()?;
            return Self::from_cpu_storage(
                cond.where_cond(layout, &t, true_layout, &f, false_layout)?,
                &self.device,
            );
        };
        if layout.shape() != true_layout.shape() || layout.shape() != false_layout.shape() {
            let cond = self.to_cpu_storage()?;
            let t = true_tensor.to_cpu_storage()?;
            let f = false_tensor.to_cpu_storage()?;
            return Self::from_cpu_storage(
                cond.where_cond(layout, &t, true_layout, &f, false_layout)?,
                &self.device,
            );
        }
        let pred_dtype = match self.dtype {
            DType::U8 => 0,
            DType::U32 => 1,
            _ => {
                let cond = self.to_cpu_storage()?;
                let t = true_tensor.to_cpu_storage()?;
                let f = false_tensor.to_cpu_storage()?;
                return Self::from_cpu_storage(
                    cond.where_cond(layout, &t, true_layout, &f, false_layout)?,
                    &self.device,
                );
            }
        };
        let value_dtype = match true_tensor.dtype {
            DType::F16 => 0,
            DType::F32 => 1,
            DType::BF16 => 2,
            DType::U8 => 3,
            DType::U32 => 4,
            DType::I64 => 5,
            _ => {
                let cond = self.to_cpu_storage()?;
                let t = true_tensor.to_cpu_storage()?;
                let f = false_tensor.to_cpu_storage()?;
                return Self::from_cpu_storage(
                    cond.where_cond(layout, &t, true_layout, &f, false_layout)?,
                    &self.device,
                );
            }
        };
        let output = Self::alloc_uninit(&self.device, true_layout.shape().elem_count(), true_tensor.dtype)?;
        let status = unsafe {
            ffi::candle_hip_where_cond_contiguous(
                pred_dtype,
                value_dtype,
                self.device.ordinal,
                true_layout.shape().elem_count(),
                self.raw_device_ptr_with_offset(cond_start)?,
                true_tensor.raw_device_ptr_with_offset(true_start)?,
                false_tensor.raw_device_ptr_with_offset(false_start)?,
                output.raw_device_ptr(),
            )
        };
        if status != 0 {
            return Err(qwen35_error("hip-where-cond", status));
        }
        Ok(output)
    }

    fn conv1d(
        &self,
        layout: &Layout,
        kernel: &Self,
        kernel_layout: &Layout,
        params: &crate::conv::ParamsConv1D,
    ) -> Result<Self> {
        self.cpu_fallback_map2(kernel, |inp, kernel| inp.conv1d(layout, kernel, kernel_layout, params))
    }

    fn conv_transpose1d(
        &self,
        layout: &Layout,
        kernel: &Self,
        kernel_layout: &Layout,
        params: &crate::conv::ParamsConvTranspose1D,
    ) -> Result<Self> {
        self.cpu_fallback_map2(kernel, |inp, kernel| {
            inp.conv_transpose1d(layout, kernel, kernel_layout, params)
        })
    }

    fn conv2d(
        &self,
        layout: &Layout,
        kernel: &Self,
        kernel_layout: &Layout,
        params: &crate::conv::ParamsConv2D,
    ) -> Result<Self> {
        self.cpu_fallback_map2(kernel, |inp, kernel| inp.conv2d(layout, kernel, kernel_layout, params))
    }

    fn conv_transpose2d(
        &self,
        layout: &Layout,
        kernel: &Self,
        kernel_layout: &Layout,
        params: &crate::conv::ParamsConvTranspose2D,
    ) -> Result<Self> {
        self.cpu_fallback_map2(kernel, |inp, kernel| {
            inp.conv_transpose2d(layout, kernel, kernel_layout, params)
        })
    }

    fn avg_pool2d(
        &self,
        layout: &Layout,
        kernel_size: (usize, usize),
        stride: (usize, usize),
    ) -> Result<Self> {
        self.cpu_fallback_map(|cpu| cpu.avg_pool2d(layout, kernel_size, stride))
    }

    fn max_pool2d(
        &self,
        layout: &Layout,
        kernel_size: (usize, usize),
        stride: (usize, usize),
    ) -> Result<Self> {
        self.cpu_fallback_map(|cpu| cpu.max_pool2d(layout, kernel_size, stride))
    }

    fn upsample_nearest1d(&self, layout: &Layout, size: usize) -> Result<Self> {
        self.cpu_fallback_map(|cpu| cpu.upsample_nearest1d(layout, size))
    }

    fn upsample_nearest2d(&self, layout: &Layout, h: usize, w: usize) -> Result<Self> {
        self.cpu_fallback_map(|cpu| cpu.upsample_nearest2d(layout, h, w))
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
        self.cpu_fallback_map(|cpu| {
            cpu.upsample_bilinear2d(layout, h, w, align_corners, scale_h, scale_w)
        })
    }

    fn gather(
        &self,
        layout: &Layout,
        indexes: &Self,
        indexes_layout: &Layout,
        dim: usize,
    ) -> Result<Self> {
        self.cpu_fallback_map2(indexes, |src, indexes| src.gather(layout, indexes, indexes_layout, dim))
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
        let mut dst = self.to_cpu_storage()?;
        let indexes = indexes.to_cpu_storage()?;
        let source = source.to_cpu_storage()?;
        dst.scatter_set(layout, &indexes, indexes_layout, &source, source_layout, dim)?;
        self.replace_from_cpu_storage(dst)
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
        let mut dst = self.to_cpu_storage()?;
        let indexes = indexes.to_cpu_storage()?;
        let source = source.to_cpu_storage()?;
        dst.scatter_add_set(layout, &indexes, indexes_layout, &source, source_layout, dim)?;
        self.replace_from_cpu_storage(dst)
    }

    fn index_select(
        &self,
        indexes: &Self,
        layout: &Layout,
        indexes_layout: &Layout,
        dim: usize,
    ) -> Result<Self> {
        self.cpu_fallback_map2(indexes, |src, indexes| src.index_select(indexes, layout, indexes_layout, dim))
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
        let src = self.to_cpu_storage()?;
        let indexes = indexes.to_cpu_storage()?;
        let add = source.to_cpu_storage()?;
        Self::from_cpu_storage(
            src.index_add(layout, &indexes, indexes_layout, &add, source_layout, dim)?,
            &self.device,
        )
    }

    fn matmul(
        &self,
        rhs: &Self,
        bmnk: (usize, usize, usize, usize),
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<Self> {
        if self.dtype != rhs.dtype {
            return self.cpu_fallback_map2(rhs, |lhs, rhs| lhs.matmul(rhs, bmnk, lhs_layout, rhs_layout));
        }
        let dtype = match hip_dtype_code(self.dtype) {
            Ok(code @ (0..=2)) => code,
            _ => return self.cpu_fallback_map2(rhs, |lhs, rhs| lhs.matmul(rhs, bmnk, lhs_layout, rhs_layout)),
        };
        let cfg = hip_gemm_config(bmnk, lhs_layout, rhs_layout)?;
        let (b, m, n, _k) = bmnk;
        let out = Self::alloc_uninit(&self.device, b * m * n, self.dtype)?;
        let status = unsafe {
            ffi::candle_hip_matmul_strided_batched(
                dtype,
                self.device.ordinal,
                b,
                m,
                n,
                bmnk.3,
                cfg.transa,
                cfg.transb,
                cfg.lda,
                cfg.ldb,
                cfg.ldc,
                cfg.stride_a,
                cfg.stride_b,
                cfg.stride_c,
                rhs.raw_device_ptr_with_offset(rhs_layout.start_offset())?,
                self.raw_device_ptr_with_offset(lhs_layout.start_offset())?,
                out.raw_device_ptr(),
            )
        };
        if status != 0 {
            return Err(qwen35_error("hip-matmul", status));
        }
        Ok(out)
    }

    fn copy_strided_src(&self, dst: &mut Self, dst_offset: usize, layout: &Layout) -> Result<()> {
        if !self.device.same_device(&dst.device) {
            return Err(Error::DeviceMismatchBinaryOp {
                lhs: self.device.location(),
                rhs: dst.device.location(),
                op: "copy",
            }
            .bt());
        }
        if self.dtype != dst.dtype {
            return Err(Error::DTypeMismatchBinaryOp {
                lhs: self.dtype,
                rhs: dst.dtype,
                op: "copy_strided",
            }
            .bt());
        }
        let elem_size = dtype_elem_size(self.dtype)?;
        match layout.strided_blocks() {
            crate::StridedBlocks::SingleBlock { start_offset, len } => {
                let to_copy = (dst.elem_count.saturating_sub(dst_offset)).min(len);
                if to_copy != 0 {
                    hip_memcpy_device_to_device(
                        self.device.ordinal,
                        dst.raw_device_ptr_with_offset(dst_offset)?,
                        self.raw_device_ptr_with_offset(start_offset)?,
                        to_copy * elem_size,
                    )?;
                }
            }
            crate::StridedBlocks::MultipleBlocks {
                block_start_index,
                block_len,
            } => {
                let mut dst_index = dst_offset;
                for src_index in block_start_index {
                    if dst_index >= dst.elem_count {
                        break;
                    }
                    let to_copy = block_len.min(dst.elem_count - dst_index);
                    if to_copy != 0 {
                        hip_memcpy_device_to_device(
                            self.device.ordinal,
                            dst.raw_device_ptr_with_offset(dst_index)?,
                            self.raw_device_ptr_with_offset(src_index)?,
                            to_copy * elem_size,
                        )?;
                    }
                    dst_index = dst_index.saturating_add(block_len);
                }
            }
        }
        Ok(())
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
        if !self.device.same_device(&dst.device) {
            return Err(Error::DeviceMismatchBinaryOp {
                lhs: self.device.location(),
                rhs: dst.device.location(),
                op: "copy2d",
            }
            .bt());
        }
        if self.dtype != dst.dtype {
            return Err(Error::DTypeMismatchBinaryOp {
                lhs: self.dtype,
                rhs: dst.dtype,
                op: "copy2d",
            }
            .bt());
        }
        let elem_size = dtype_elem_size(self.dtype)?;
        let width_bytes = d2
            .checked_mul(elem_size)
            .ok_or_else(|| Error::Msg("hip copy2d width overflow".into()).bt())?;
        let src_pitch = src_stride1
            .checked_mul(elem_size)
            .ok_or_else(|| Error::Msg("hip copy2d src pitch overflow".into()).bt())?;
        let dst_pitch = dst_stride1
            .checked_mul(elem_size)
            .ok_or_else(|| Error::Msg("hip copy2d dst pitch overflow".into()).bt())?;
        let src_ptr = self.raw_device_ptr_with_offset(src_offset)?;
        let dst_ptr = dst.raw_device_ptr_with_offset(dst_offset)?;
        hip_memcpy_2d_device_to_device(
            self.device.ordinal,
            dst_ptr,
            dst_pitch,
            src_ptr,
            src_pitch,
            width_bytes,
            d1,
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
        HipStorage::zeros(self, shape.elem_count(), dtype)
    }

    unsafe fn alloc_uninit(&self, shape: &Shape, dtype: DType) -> Result<Self::Storage> {
        HipStorage::alloc_uninit(self, shape.elem_count(), dtype)
    }

    fn storage_from_slice<T: crate::WithDType>(&self, data: &[T]) -> Result<Self::Storage> {
        Self::storage_from_cpu_storage(self, &T::to_cpu_storage(data))
    }

    fn storage_from_cpu_storage(&self, storage: &CpuStorage) -> Result<Self::Storage> {
        HipStorage::from_cpu_storage(storage.clone(), self)
    }

    fn storage_from_cpu_storage_owned(&self, storage: CpuStorage) -> Result<Self::Storage> {
        HipStorage::from_cpu_storage(storage, self)
    }

    fn rand_uniform(&self, shape: &Shape, dtype: DType, lo: f64, up: f64) -> Result<Self::Storage> {
        HipStorage::from_cpu_storage(CpuDevice.rand_uniform(shape, dtype, lo, up)?, self)
    }

    fn rand_normal(
        &self,
        shape: &Shape,
        dtype: DType,
        mean: f64,
        std: f64,
    ) -> Result<Self::Storage> {
        HipStorage::from_cpu_storage(CpuDevice.rand_normal(shape, dtype, mean, std)?, self)
    }

    fn set_seed(&self, seed: u64) -> Result<()> {
        CpuDevice.set_seed(seed)
    }

    fn get_current_seed(&self) -> Result<u64> {
        CpuDevice.get_current_seed()
    }

    fn synchronize(&self) -> Result<()> {
        hip_synchronize(self.ordinal)
    }
}

#[derive(Debug)]
struct ScopedHipDevice {
    previous: c_int,
    changed: bool,
}

impl ScopedHipDevice {
    fn new(target: usize) -> Result<Self> {
        let mut previous = 0;
        hip_status(
            unsafe { hipGetDevice(&mut previous as *mut c_int) },
            "hipGetDevice",
        )?;
        if previous != target as c_int {
            hip_status(unsafe { hipSetDevice(target as c_int) }, "hipSetDevice")?;
            Ok(Self {
                previous,
                changed: true,
            })
        } else {
            Ok(Self {
                previous,
                changed: false,
            })
        }
    }
}

impl Drop for ScopedHipDevice {
    fn drop(&mut self) {
        if self.changed {
            unsafe {
                let _ = hipSetDevice(self.previous);
            }
        }
    }
}

fn hip_status(code: c_int, op: &'static str) -> Result<()> {
    if code == 0 {
        Ok(())
    } else {
        Err(qwen35_error(op, code))
    }
}

fn hip_synchronize(device_ordinal: usize) -> Result<()> {
    let _scoped = ScopedHipDevice::new(device_ordinal)?;
    hip_status(unsafe { hipDeviceSynchronize() }, "hipDeviceSynchronize")
}

fn allocate_hip_buffer(device_ordinal: usize, len_bytes: usize) -> Result<Arc<HipAllocation>> {
    if len_bytes == 0 {
        return Ok(Arc::new(HipAllocation {
            ptr: None,
            len_bytes: 0,
            device_ordinal,
        }));
    }
    let _scoped = ScopedHipDevice::new(device_ordinal)?;
    let mut ptr = ptr::null_mut();
    hip_status(
        unsafe { hipMalloc(&mut ptr as *mut *mut c_void, len_bytes) },
        "hipMalloc",
    )?;
    Ok(Arc::new(HipAllocation {
        ptr: NonNull::new(ptr),
        len_bytes,
        device_ordinal,
    }))
}

fn hip_memcpy_host_to_device(
    device_ordinal: usize,
    dst: Option<NonNull<c_void>>,
    src: *const c_void,
    len_bytes: usize,
) -> Result<()> {
    if len_bytes == 0 {
        return Ok(());
    }
    let _scoped = ScopedHipDevice::new(device_ordinal)?;
    let dst = dst
        .map(NonNull::as_ptr)
        .ok_or_else(|| Error::Msg("missing HIP destination pointer".into()).bt())?;
    hip_status(
        unsafe { hipMemcpy(dst, src, len_bytes, HIP_MEMCPY_HOST_TO_DEVICE) },
        "hipMemcpy H2D",
    )?;
    HIP_H2D_BYTES.fetch_add(len_bytes as u64, Ordering::Relaxed);
    Ok(())
}

fn hip_memcpy_device_to_host(
    device_ordinal: usize,
    dst: *mut c_void,
    src: *mut c_void,
    len_bytes: usize,
) -> Result<()> {
    if len_bytes == 0 {
        return Ok(());
    }
    let _scoped = ScopedHipDevice::new(device_ordinal)?;
    hip_status(
        unsafe { hipMemcpy(dst, src as *const c_void, len_bytes, HIP_MEMCPY_DEVICE_TO_HOST) },
        "hipMemcpy D2H",
    )?;
    HIP_D2H_BYTES.fetch_add(len_bytes as u64, Ordering::Relaxed);
    Ok(())
}

fn hip_memcpy_device_to_device(
    device_ordinal: usize,
    dst: *mut c_void,
    src: *mut c_void,
    len_bytes: usize,
) -> Result<()> {
    if len_bytes == 0 {
        return Ok(());
    }
    let _scoped = ScopedHipDevice::new(device_ordinal)?;
    hip_status(
        unsafe { hipMemcpy(dst, src as *const c_void, len_bytes, HIP_MEMCPY_DEVICE_TO_DEVICE) },
        "hipMemcpy D2D",
    )?;
    HIP_D2D_BYTES.fetch_add(len_bytes as u64, Ordering::Relaxed);
    Ok(())
}

fn hip_memcpy_2d_device_to_device(
    device_ordinal: usize,
    dst: *mut c_void,
    dst_pitch: usize,
    src: *mut c_void,
    src_pitch: usize,
    width_bytes: usize,
    height: usize,
) -> Result<()> {
    if width_bytes == 0 || height == 0 {
        return Ok(());
    }
    let _scoped = ScopedHipDevice::new(device_ordinal)?;
    hip_status(
        unsafe {
            hipMemcpy2D(
                dst,
                dst_pitch,
                src as *const c_void,
                src_pitch,
                width_bytes,
                height,
                HIP_MEMCPY_DEVICE_TO_DEVICE,
            )
        },
        "hipMemcpy2D D2D",
    )?;
    HIP_D2D_BYTES.fetch_add((width_bytes * height) as u64, Ordering::Relaxed);
    Ok(())
}

fn hip_memset(device_ordinal: usize, dst: *mut c_void, value: c_int, len_bytes: usize) -> Result<()> {
    if len_bytes == 0 {
        return Ok(());
    }
    let _scoped = ScopedHipDevice::new(device_ordinal)?;
    hip_status(unsafe { hipMemset(dst, value, len_bytes) }, "hipMemset")
}

fn cpu_storage_elem_count(storage: &CpuStorage) -> usize {
    match storage {
        CpuStorage::U8(v) => v.len(),
        CpuStorage::U32(v) => v.len(),
        CpuStorage::I16(v) => v.len(),
        CpuStorage::I32(v) => v.len(),
        CpuStorage::I64(v) => v.len(),
        CpuStorage::BF16(v) => v.len(),
        CpuStorage::F16(v) => v.len(),
        CpuStorage::F32(v) => v.len(),
        CpuStorage::F64(v) => v.len(),
        CpuStorage::F8E4M3(v) => v.len(),
        CpuStorage::F6E2M3(v) => v.len(),
        CpuStorage::F6E3M2(v) => v.len(),
        CpuStorage::F4(v) => v.len(),
        CpuStorage::F8E8M0(v) => v.len(),
    }
}

fn cpu_storage_len_bytes(storage: &CpuStorage) -> Result<usize> {
    Ok(match storage {
        CpuStorage::U8(v) => v.len(),
        CpuStorage::U32(v) => v.len() * std::mem::size_of::<u32>(),
        CpuStorage::I16(v) => v.len() * std::mem::size_of::<i16>(),
        CpuStorage::I32(v) => v.len() * std::mem::size_of::<i32>(),
        CpuStorage::I64(v) => v.len() * std::mem::size_of::<i64>(),
        CpuStorage::BF16(v) => v.len() * std::mem::size_of::<bf16>(),
        CpuStorage::F16(v) => v.len() * std::mem::size_of::<f16>(),
        CpuStorage::F32(v) => v.len() * std::mem::size_of::<f32>(),
        CpuStorage::F64(v) => v.len() * std::mem::size_of::<f64>(),
        CpuStorage::F8E4M3(v) => v.len() * std::mem::size_of::<F8E4M3>(),
        CpuStorage::F6E2M3(v) => v.len(),
        CpuStorage::F6E3M2(v) => v.len(),
        CpuStorage::F4(v) => v.len(),
        CpuStorage::F8E8M0(v) => v.len(),
    })
}

fn cpu_storage_as_ptr(storage: &CpuStorage) -> *const c_void {
    match storage {
        CpuStorage::U8(v) => v.as_ptr() as *const c_void,
        CpuStorage::U32(v) => v.as_ptr() as *const c_void,
        CpuStorage::I16(v) => v.as_ptr() as *const c_void,
        CpuStorage::I32(v) => v.as_ptr() as *const c_void,
        CpuStorage::I64(v) => v.as_ptr() as *const c_void,
        CpuStorage::BF16(v) => v.as_ptr() as *const c_void,
        CpuStorage::F16(v) => v.as_ptr() as *const c_void,
        CpuStorage::F32(v) => v.as_ptr() as *const c_void,
        CpuStorage::F64(v) => v.as_ptr() as *const c_void,
        CpuStorage::F8E4M3(v) => v.as_ptr() as *const c_void,
        CpuStorage::F6E2M3(v) => v.as_ptr() as *const c_void,
        CpuStorage::F6E3M2(v) => v.as_ptr() as *const c_void,
        CpuStorage::F4(v) => v.as_ptr() as *const c_void,
        CpuStorage::F8E8M0(v) => v.as_ptr() as *const c_void,
    }
}

fn cpu_storage_as_mut_ptr(storage: &mut CpuStorage) -> *mut c_void {
    match storage {
        CpuStorage::U8(v) => v.as_mut_ptr() as *mut c_void,
        CpuStorage::U32(v) => v.as_mut_ptr() as *mut c_void,
        CpuStorage::I16(v) => v.as_mut_ptr() as *mut c_void,
        CpuStorage::I32(v) => v.as_mut_ptr() as *mut c_void,
        CpuStorage::I64(v) => v.as_mut_ptr() as *mut c_void,
        CpuStorage::BF16(v) => v.as_mut_ptr() as *mut c_void,
        CpuStorage::F16(v) => v.as_mut_ptr() as *mut c_void,
        CpuStorage::F32(v) => v.as_mut_ptr() as *mut c_void,
        CpuStorage::F64(v) => v.as_mut_ptr() as *mut c_void,
        CpuStorage::F8E4M3(v) => v.as_mut_ptr() as *mut c_void,
        CpuStorage::F6E2M3(v) => v.as_mut_ptr() as *mut c_void,
        CpuStorage::F6E3M2(v) => v.as_mut_ptr() as *mut c_void,
        CpuStorage::F4(v) => v.as_mut_ptr() as *mut c_void,
        CpuStorage::F8E8M0(v) => v.as_mut_ptr() as *mut c_void,
    }
}

fn zeroed_cpu_storage(dtype: DType, elem_count: usize) -> Result<CpuStorage> {
    Ok(match dtype {
        DType::U8 => CpuStorage::U8(vec![0; elem_count]),
        DType::U32 => CpuStorage::U32(vec![0; elem_count]),
        DType::I16 => CpuStorage::I16(vec![0; elem_count]),
        DType::I32 => CpuStorage::I32(vec![0; elem_count]),
        DType::I64 => CpuStorage::I64(vec![0; elem_count]),
        DType::BF16 => CpuStorage::BF16(vec![bf16::from_f32(0.0); elem_count]),
        DType::F16 => CpuStorage::F16(vec![f16::from_f32(0.0); elem_count]),
        DType::F32 => CpuStorage::F32(vec![0.0; elem_count]),
        DType::F64 => CpuStorage::F64(vec![0.0; elem_count]),
        DType::F8E4M3 => CpuStorage::F8E4M3(vec![F8E4M3::from_f32(0.0); elem_count]),
        DType::F6E2M3 => CpuStorage::F6E2M3(vec![0; elem_count]),
        DType::F6E3M2 => CpuStorage::F6E3M2(vec![0; elem_count]),
        DType::F4 => CpuStorage::F4(vec![0; elem_count]),
        DType::F8E8M0 => CpuStorage::F8E8M0(vec![0; elem_count]),
    })
}

fn dtype_elem_size(dtype: DType) -> Result<usize> {
    let size = dtype.size_in_bytes();
    if size == 0 {
        crate::bail!("HIP backend does not support packed dtype {dtype:?}")
    } else {
        Ok(size)
    }
}

#[link(name = "amdhip64")]
unsafe extern "C" {
    fn hipMalloc(ptr: *mut *mut c_void, size: usize) -> c_int;
    fn hipFree(ptr: *mut c_void) -> c_int;
    fn hipMemcpy(dst: *mut c_void, src: *const c_void, size: usize, kind: c_int) -> c_int;
    fn hipMemcpy2D(
        dst: *mut c_void,
        dpitch: usize,
        src: *const c_void,
        spitch: usize,
        width: usize,
        height: usize,
        kind: c_int,
    ) -> c_int;
    fn hipMemset(dst: *mut c_void, value: c_int, size: usize) -> c_int;
    fn hipGetDevice(device: *mut c_int) -> c_int;
    fn hipSetDevice(device: c_int) -> c_int;
    fn hipDeviceSynchronize() -> c_int;
}

pub mod ffi {
    use super::*;

    unsafe extern "C" {
        pub fn qwen35_hip_linear_prefill_conv_pack(
            dtype: c_int,
            device_ordinal: usize,
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
            device_ordinal: usize,
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

        pub fn qwen35_hip_full_attention_prefill_persistent(
            dtype: c_int,
            device_ordinal: usize,
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
            device_ordinal: usize,
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
            device_ordinal: usize,
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
            device_ordinal: usize,
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
            device_ordinal: usize,
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
            device_ordinal: usize,
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
            device_ordinal: usize,
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
            device_ordinal: usize,
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

        pub fn candle_hip_cast_contiguous(
            src_dtype: c_int,
            dst_dtype: c_int,
            device_ordinal: usize,
            elem_count: usize,
            src: *const c_void,
            dst: *mut c_void,
        ) -> c_int;

        pub fn candle_hip_affine_contiguous(
            dtype: c_int,
            device_ordinal: usize,
            elem_count: usize,
            src: *const c_void,
            mul: f32,
            add: f32,
            dst: *mut c_void,
        ) -> c_int;

        pub fn candle_hip_sigmoid_contiguous(
            dtype: c_int,
            device_ordinal: usize,
            elem_count: usize,
            src: *const c_void,
            dst: *mut c_void,
        ) -> c_int;

        pub fn candle_hip_where_cond_contiguous(
            pred_dtype: c_int,
            value_dtype: c_int,
            device_ordinal: usize,
            elem_count: usize,
            pred: *const c_void,
            on_true: *const c_void,
            on_false: *const c_void,
            dst: *mut c_void,
        ) -> c_int;

        pub fn candle_hip_reduce_contiguous(
            op: c_int,
            dtype: c_int,
            device_ordinal: usize,
            n_rows: usize,
            n_cols: usize,
            src: *const c_void,
            dst: *mut c_void,
        ) -> c_int;

        pub fn candle_hip_softmax_last_dim_contiguous(
            dtype: c_int,
            device_ordinal: usize,
            n_rows: usize,
            n_cols: usize,
            src: *const c_void,
            dst: *mut c_void,
        ) -> c_int;

        pub fn candle_hip_rope_i_contiguous(
            dtype: c_int,
            device_ordinal: usize,
            bh: u32,
            td: u32,
            stride_b: u32,
            src: *const c_void,
            cos: *const c_void,
            sin: *const c_void,
            dst: *mut c_void,
        ) -> c_int;

        pub fn candle_hip_rope_contiguous(
            dtype: c_int,
            device_ordinal: usize,
            bh: u32,
            td: u32,
            d: u32,
            stride_b: u32,
            src: *const c_void,
            cos: *const c_void,
            sin: *const c_void,
            dst: *mut c_void,
        ) -> c_int;

        pub fn candle_hip_rope_thd_contiguous(
            dtype: c_int,
            device_ordinal: usize,
            b: u32,
            t: u32,
            h: u32,
            d: u32,
            stride_b: u32,
            src: *const c_void,
            cos: *const c_void,
            sin: *const c_void,
            dst: *mut c_void,
        ) -> c_int;

        pub fn candle_hip_matmul_strided_batched(
            dtype: c_int,
            device_ordinal: usize,
            batch_size: usize,
            m: usize,
            n: usize,
            k: usize,
            transa: c_int,
            transb: c_int,
            lda: c_int,
            ldb: c_int,
            ldc: c_int,
            stride_a: i64,
            stride_b: i64,
            stride_c: i64,
            a: *const c_void,
            b: *const c_void,
            c: *mut c_void,
        ) -> c_int;
    }
}

pub fn hip_dtype_code(dtype: DType) -> Result<c_int> {
    match dtype {
        DType::F16 => Ok(0),
        DType::F32 => Ok(1),
        DType::BF16 => Ok(2),
        DType::U8 => Ok(3),
        other => crate::bail!("unsupported HIP dtype: {other:?}"),
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
