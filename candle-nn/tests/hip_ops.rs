#[cfg(feature = "hip")]
use candle::{test_utils::to_vec1_round, DType, Device, Result, Tensor};
#[cfg(feature = "hip")]
use std::sync::Mutex;

#[cfg(feature = "hip")]
static HIP_TEST_LOCK: Mutex<()> = Mutex::new(());

#[cfg(feature = "hip")]
fn max_diff(lhs: &Tensor, rhs: &Tensor) -> Result<f32> {
    let lhs = lhs.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    let rhs = rhs.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    let mut max = 0.0f32;
    for (lhs, rhs) in lhs.iter().zip(rhs.iter()) {
        max = max.max((lhs - rhs).abs());
    }
    Ok(max)
}

#[cfg(feature = "hip")]
#[test]
fn hip_to_dtype_contiguous_avoids_host_staging() -> Result<()> {
    let _guard = HIP_TEST_LOCK.lock().unwrap();
    let dev = Device::new_hip(0)?;
    let src = Tensor::new(&[[1.0f32, -2.5, 3.25], [0.0, 7.0, -1.0]], &dev)?.to_dtype(DType::F16)?;
    candle::hip::reset_transfer_counters();
    let cast = src.to_dtype(DType::F32)?;
    let counters = candle::hip::transfer_counters();
    assert_eq!(counters.host_to_device_bytes, 0);
    assert_eq!(counters.device_to_host_bytes, 0);
    assert_eq!(to_vec1_round(&cast.flatten_all()?, 4)?, vec![1.0, -2.5, 3.25, 0.0, 7.0, -1.0]);

    let mask = Tensor::from_vec(vec![0u8, 1, 1, 0], (2, 2), &dev)?;
    candle::hip::reset_transfer_counters();
    let cast = mask.to_dtype(DType::F16)?;
    let counters = candle::hip::transfer_counters();
    assert_eq!(counters.host_to_device_bytes, 0);
    assert_eq!(counters.device_to_host_bytes, 0);
    assert_eq!(to_vec1_round(&cast.flatten_all()?.to_dtype(DType::F32)?, 4)?, vec![0.0, 1.0, 1.0, 0.0]);
    Ok(())
}

#[cfg(feature = "hip")]
#[test]
fn hip_sigmoid_matches_cpu_without_host_staging() -> Result<()> {
    let _guard = HIP_TEST_LOCK.lock().unwrap();
    let cpu = Device::Cpu;
    let hip = Device::new_hip(0)?;
    let data = &[[[3f32, 1., 4.], [1., 5., 9.]], [[2., 1., 7.], [8., 2., 8.]]];
    let cpu_tensor = Tensor::new(data, &cpu)?;
    let hip_tensor = Tensor::new(data, &hip)?;
    let cpu_out = candle_nn::ops::sigmoid(&cpu_tensor)?;
    candle::hip::reset_transfer_counters();
    let hip_out = candle_nn::ops::sigmoid(&hip_tensor)?;
    let counters = candle::hip::transfer_counters();
    assert_eq!(counters.host_to_device_bytes, 0);
    assert_eq!(counters.device_to_host_bytes, 0);
    assert_eq!(max_diff(&cpu_out, &hip_out)?, 0.0);
    Ok(())
}

#[cfg(feature = "hip")]
#[test]
fn hip_softmax_last_dim_matches_cpu_without_host_staging() -> Result<()> {
    let _guard = HIP_TEST_LOCK.lock().unwrap();
    let cpu = Device::Cpu;
    let hip = Device::new_hip(0)?;
    let data = &[[[3f32, 1., 4.], [1., 5., 9.]], [[2., 1., 7.], [8., 2., 8.]]];
    let cpu_tensor = Tensor::new(data, &cpu)?.log()?;
    let hip_tensor = Tensor::new(data, &hip)?.log()?;
    let cpu_out = candle_nn::ops::softmax_last_dim(&cpu_tensor)?;
    candle::hip::reset_transfer_counters();
    let hip_out = candle_nn::ops::softmax_last_dim(&hip_tensor)?;
    let counters = candle::hip::transfer_counters();
    assert_eq!(counters.host_to_device_bytes, 0);
    assert_eq!(counters.device_to_host_bytes, 0);
    assert!(max_diff(&cpu_out, &hip_out)? < 1e-5);
    Ok(())
}

#[cfg(feature = "hip")]
#[test]
fn hip_rotary_variants_match_cpu_without_host_staging() -> Result<()> {
    let _guard = HIP_TEST_LOCK.lock().unwrap();
    let cpu = Device::Cpu;
    let hip = Device::new_hip(0)?;
    let (b_size, num_head, seq_len, head_dim) = (2, 3, 4, 8);
    let el_count = b_size * num_head * seq_len * head_dim;
    let src: Vec<f32> = (0..el_count).map(|idx| (idx as f32) / 17.0).collect();
    let cos: Vec<f32> = (0..seq_len * head_dim / 2)
        .map(|idx| ((idx as f32) / 11.0).cos())
        .collect();
    let sin: Vec<f32> = (0..seq_len * head_dim / 2)
        .map(|idx| ((idx as f32) / 11.0).sin())
        .collect();

    let cpu_src = Tensor::from_vec(src.clone(), (b_size, num_head, seq_len, head_dim), &cpu)?;
    let hip_src = Tensor::from_vec(src, (b_size, num_head, seq_len, head_dim), &hip)?;
    let cpu_cos = Tensor::from_vec(cos.clone(), (seq_len, head_dim / 2), &cpu)?;
    let hip_cos = Tensor::from_vec(cos, (seq_len, head_dim / 2), &hip)?;
    let cpu_sin = Tensor::from_vec(sin.clone(), (seq_len, head_dim / 2), &cpu)?;
    let hip_sin = Tensor::from_vec(sin, (seq_len, head_dim / 2), &hip)?;

    let cpu_ropei = candle_nn::rotary_emb::rope_i(&cpu_src, &cpu_cos, &cpu_sin)?;
    candle::hip::reset_transfer_counters();
    let hip_ropei = candle_nn::rotary_emb::rope_i(&hip_src, &hip_cos, &hip_sin)?;
    let counters = candle::hip::transfer_counters();
    assert_eq!(counters.host_to_device_bytes, 0);
    assert_eq!(counters.device_to_host_bytes, 0);
    assert!(max_diff(&cpu_ropei, &hip_ropei)? < 1e-4);

    let cpu_rope = candle_nn::rotary_emb::rope(&cpu_src, &cpu_cos, &cpu_sin)?;
    candle::hip::reset_transfer_counters();
    let hip_rope = candle_nn::rotary_emb::rope(&hip_src, &hip_cos, &hip_sin)?;
    let counters = candle::hip::transfer_counters();
    assert_eq!(counters.host_to_device_bytes, 0);
    assert_eq!(counters.device_to_host_bytes, 0);
    assert!(max_diff(&cpu_rope, &hip_rope)? < 1e-4);

    let cpu_src_thd = cpu_src.transpose(1, 2)?.contiguous()?;
    let hip_src_thd = hip_src.transpose(1, 2)?.contiguous()?;
    let cpu_rope_thd = candle_nn::rotary_emb::rope_thd(&cpu_src_thd, &cpu_cos, &cpu_sin)?;
    candle::hip::reset_transfer_counters();
    let hip_rope_thd = candle_nn::rotary_emb::rope_thd(&hip_src_thd, &hip_cos, &hip_sin)?;
    let counters = candle::hip::transfer_counters();
    assert_eq!(counters.host_to_device_bytes, 0);
    assert_eq!(counters.device_to_host_bytes, 0);
    assert!(max_diff(&cpu_rope_thd, &hip_rope_thd)? < 1e-4);
    Ok(())
}
