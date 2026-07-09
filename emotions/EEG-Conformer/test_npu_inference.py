"""
NPU (RK3588) vs CPU 推理对比测试
"""
import sys
sys.path.insert(0, '.')
import time
import numpy as np
import torch
from inference_model import load_model
from rknnlite.api import RKNNLite


def test_cpu(model, data, n_runs=100):
    """PyTorch CPU 推理基准"""
    model.eval()
    inputs = torch.from_numpy(data).float()
    
    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _, out = model(inputs)
    
    # Timed
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        with torch.no_grad():
            _, out = model(inputs)
        times.append(time.perf_counter() - t0)
    
    avg = np.mean(times) * 1000
    std = np.std(times) * 1000
    return avg, std, out.numpy()


def test_npu(rknn_model_path, data, n_runs=100):
    """RK3588 NPU 推理基准"""
    rknn = RKNNLite()
    
    ret = rknn.load_rknn(rknn_model_path)
    if ret != 0:
        raise RuntimeError(f'Load RKNN failed: {ret}')
    
    # Use core 0
    ret = rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0)
    if ret != 0:
        raise RuntimeError(f'Init runtime failed: {ret}')
    
    # Warmup
    for _ in range(10):
        rknn.inference(inputs=[data])
    
    # Timed
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        outputs = rknn.inference(inputs=[data])
        times.append(time.perf_counter() - t0)
    
    avg = np.mean(times) * 1000
    std = np.std(times) * 1000
    rknn.release()
    return avg, std, outputs


def main():
    data = np.random.randn(1, 1, 8, 200).astype(np.float32)
    
    print('=' * 60)
    print('EEG-Conformer 8-Channel Inference Benchmark')
    print('=' * 60)
    
    # CPU
    print('\n[1/2] PyTorch CPU baseline...')
    model = load_model('/home/elf/Projects/emotions/best_fold1.pth', 'cpu')
    cpu_avg, cpu_std, cpu_out = test_cpu(model, data)
    cpu_class = cpu_out.argmax(axis=1)[0]
    print(f'  Class: {cpu_class} | {cpu_avg:.3f} ± {cpu_std:.3f} ms (avg over 100 runs)')
    
    # NPU
    print('\n[2/2] RK3588 NPU inference...')
    npu_avg, npu_std, npu_outs = test_npu(
        '/home/elf/Projects/emotions/EEG-Conformer/eeg_conformer.rknn',
        data
    )
    npu_class = npu_outs[1].argmax(axis=1)[0] if len(npu_outs) > 1 else npu_outs[0].argmax(axis=1)[0]
    print(f'  Class: {npu_class} | {npu_avg:.3f} ± {npu_std:.3f} ms (avg over 100 runs)')
    
    # Summary
    print('\n' + '=' * 60)
    print('RESULTS')
    print('=' * 60)
    speedup = cpu_avg / npu_avg if npu_avg > 0 else 0
    print(f'  CPU:  {cpu_avg:.3f} ms/sample')
    print(f'  NPU:  {npu_avg:.3f} ms/sample')
    print(f'  Speedup: {speedup:.2f}x')
    print(f'  Model size: 143K params, ~676KB on NPU')
    print(f'  NPU Memory: 364KB internal + 313KB weights')


if __name__ == '__main__':
    main()
