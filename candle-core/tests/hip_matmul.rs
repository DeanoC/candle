#[cfg(feature = "hip")]
use candle_core::{Device, Result, Tensor};
#[cfg(feature = "hip")]
use std::sync::Mutex;

#[cfg(feature = "hip")]
static HIP_TEST_LOCK: Mutex<()> = Mutex::new(());

#[cfg(feature = "hip")]
fn max_diff(lhs: &Tensor, rhs: &Tensor) -> Result<f32> {
    let lhs = lhs.to_dtype(candle_core::DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    let rhs = rhs.to_dtype(candle_core::DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    let mut max = 0.0f32;
    for (lhs, rhs) in lhs.iter().zip(rhs.iter()) {
        max = max.max((lhs - rhs).abs());
    }
    Ok(max)
}

#[cfg(feature = "hip")]
#[test]
fn hip_matmul_matches_cpu_without_host_staging() -> Result<()> {
    let _guard = HIP_TEST_LOCK.lock().unwrap();
    let cpu = Device::Cpu;
    let hip = Device::new_hip(0)?;

    let lhs = Tensor::from_vec((0..24).map(|v| v as f32 / 8.0).collect::<Vec<_>>(), (2, 3, 4), &cpu)?;
    let rhs =
        Tensor::from_vec((0..24).map(|v| (v as f32 - 7.0) / 5.0).collect::<Vec<_>>(), (2, 4, 3), &cpu)?;
    let cpu_out = lhs.matmul(&rhs)?;

    let lhs_hip = lhs.to_device(&hip)?;
    let rhs_hip = rhs.to_device(&hip)?;
    candle_core::hip::reset_transfer_counters();
    let hip_out = lhs_hip.matmul(&rhs_hip)?;
    let counters = candle_core::hip::transfer_counters();

    assert_eq!(counters.host_to_device_bytes, 0);
    assert_eq!(counters.device_to_host_bytes, 0);
    assert!(max_diff(&cpu_out, &hip_out)? < 1e-5);
    Ok(())
}

#[cfg(feature = "hip")]
#[test]
fn hip_matmul_bf16_matches_cpu_without_host_staging() -> Result<()> {
    let _guard = HIP_TEST_LOCK.lock().unwrap();
    let cpu = Device::Cpu;
    let hip = Device::new_hip(0)?;
    if !hip.supports_bf16() {
        return Ok(());
    }

    let lhs_f32 =
        Tensor::from_vec((0..24).map(|v| v as f32 / 11.0).collect::<Vec<_>>(), (2, 3, 4), &cpu)?;
    let rhs_f32 = Tensor::from_vec(
        (0..24).map(|v| (v as f32 - 5.0) / 9.0).collect::<Vec<_>>(),
        (2, 4, 3),
        &cpu,
    )?;
    let cpu_out = lhs_f32.matmul(&rhs_f32)?;
    let lhs = lhs_f32.to_dtype(candle_core::DType::BF16)?;
    let rhs = rhs_f32.to_dtype(candle_core::DType::BF16)?;

    let lhs_hip = lhs.to_device(&hip)?;
    let rhs_hip = rhs.to_device(&hip)?;
    candle_core::hip::reset_transfer_counters();
    let hip_out = lhs_hip.matmul(&rhs_hip)?.to_dtype(candle_core::DType::F32)?;
    let counters = candle_core::hip::transfer_counters();

    assert_eq!(counters.host_to_device_bytes, 0);
    assert_eq!(counters.device_to_host_bytes, 0);
    assert!(max_diff(&cpu_out, &hip_out)? < 5e-2);
    Ok(())
}
