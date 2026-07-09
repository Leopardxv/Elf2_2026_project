"""
全量数据集 NPU/CPU 推理测试 + NPU 负载监控
"""
import sys
sys.path.insert(0, '.')
import time
import numpy as np
import torch
import threading
from inference_model import load_model
from rknnlite.api import RKNNLite

DATA_PATH = '/home/elf/Projects/emotions/S1_session1.npy'
LABEL_PATH = '/home/elf/Projects/emotions/S1_session1_label.npy'
RKNN_MODEL = '/home/elf/Projects/emotions/EEG-Conformer/eeg_conformer.rknn'
WEIGHT_PATH = '/home/elf/Projects/emotions/best_fold1.pth'
NPU_LOAD_PATH = '/sys/class/devfreq/fdab0000.npu/load'


class NPULoadMonitor:
    def __init__(self):
        self.samples = []
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        self._stop.clear()
        self._thread = threading.Thread(target=self._monitor, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join()

    def _monitor(self):
        while not self._stop.is_set():
            try:
                with open(NPU_LOAD_PATH) as f:
                    val = f.read().strip()
                    if '@' in val:
                        load_str = val.split('@')[0]
                        self.samples.append(int(load_str))
            except:
                pass
            time.sleep(0.05)

    def stats(self):
        if not self.samples:
            return 0, 0, 0
        arr = np.array(self.samples)
        return arr.mean(), arr.min(), arr.max()


def load_dataset():
    data = np.load(DATA_PATH, allow_pickle=True)
    label = np.load(LABEL_PATH, allow_pickle=True)
    all_data = []
    all_label = []
    for i in range(len(data)):
        d = np.array(data[i]).astype(np.float32)  # (N, 8, 200)
        l = np.array(label[i]).astype(np.int64)
        all_data.append(d)
        all_label.append(l)
    X = np.concatenate(all_data, axis=0)  # (3394, 8, 200)
    y = np.concatenate(all_label, axis=0)  # (3394,)
    # Add channel dim: (N, 1, 8, 200)
    X = X[:, np.newaxis, :, :]
    return X, y


def test_cpu_batch(model, X, y):
    model.eval()
    correct = 0
    total = len(X)
    times = []
    t_total = time.perf_counter()
    for i in range(total):
        inp = torch.from_numpy(X[i:i+1]).float()
        t0 = time.perf_counter()
        with torch.no_grad():
            _, out = model(inp)
        times.append(time.perf_counter() - t0)
        pred = out.argmax(dim=1).item()
        label = y[i] + 1  # map [-1,0,1] -> [0,1,2]
        if pred == label:
            correct += 1
    elapsed = time.perf_counter() - t_total
    avg_ms = np.mean(times) * 1000
    return correct, total, avg_ms, elapsed


def test_npu_batch(rknn, X, y):
    correct = 0
    total = len(X)
    times = []

    monitor = NPULoadMonitor()
    monitor.start()

    t_total = time.perf_counter()
    for i in range(total):
        inp = X[i:i+1]
        t0 = time.perf_counter()
        outputs = rknn.inference(inputs=[inp])
        times.append(time.perf_counter() - t0)
        logits = outputs[1]
        pred = logits.argmax(axis=1)[0]
        label = y[i] + 1
        if pred == label:
            correct += 1

    elapsed = time.perf_counter() - t_total
    monitor.stop()
    npu_mean, npu_min, npu_max = monitor.stats()
    avg_ms = np.mean(times) * 1000
    return correct, total, avg_ms, elapsed, npu_mean, npu_min, npu_max


def main():
    print('Loading dataset...')
    X, y = load_dataset()
    print(f'Dataset: {X.shape} | Labels: {y.shape}')
    unique_labels, counts = np.unique(y, return_counts=True)
    label_names = {-1: 'Negative', 0: 'Neutral', 1: 'Positive'}
    for ul, c in zip(unique_labels, counts):
        print(f'  {label_names[ul]}: {c} ({100*c/len(y):.1f}%)')

    # CPU
    print('\n' + '=' * 60)
    print('[1/3] CPU Inference (PyTorch)')
    print('=' * 60)
    model = load_model(WEIGHT_PATH, 'cpu')
    cpu_correct, cpu_total, cpu_avg, cpu_elapsed = test_cpu_batch(model, X, y)
    cpu_acc = 100 * cpu_correct / cpu_total
    print(f'  Accuracy: {cpu_correct}/{cpu_total} = {cpu_acc:.2f}%')
    print(f'  Avg latency: {cpu_avg:.3f} ms/sample')
    print(f'  Total time: {cpu_elapsed:.2f}s | {cpu_total/cpu_elapsed:.1f} fps')

    # NPU
    print('\n' + '=' * 60)
    print('[2/3] NPU Inference (RK3588)')
    print('=' * 60)
    rknn = RKNNLite()
    ret = rknn.load_rknn(RKNN_MODEL)
    if ret != 0:
        print(f'Load RKNN failed: {ret}')
        return
    ret = rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0)
    if ret != 0:
        print(f'Init runtime failed: {ret}')
        return

    npu_correct, npu_total, npu_avg, npu_elapsed, npu_mean_ld, npu_min_ld, npu_max_ld = test_npu_batch(rknn, X, y)
    npu_acc = 100 * npu_correct / npu_total
    print(f'  Accuracy: {npu_correct}/{npu_total} = {npu_acc:.2f}%')
    print(f'  Avg latency: {npu_avg:.3f} ms/sample')
    print(f'  Total time: {npu_elapsed:.2f}s | {npu_total/npu_elapsed:.1f} fps')
    print(f'  NPU Load: mean={npu_mean_ld}%, min={npu_min_ld}%, max={npu_max_ld}%')

    # Compare outputs
    print('\n' + '=' * 60)
    print('[3/3] Output Consistency Check (first 100 samples)')
    print('=' * 60)
    mismatches = 0
    for i in range(min(100, len(X))):
        inp = X[i:i+1]
        with torch.no_grad():
            _, cpu_out = model(torch.from_numpy(inp).float())
        npu_outputs = rknn.inference(inputs=[inp])
        cpu_cls = cpu_out.argmax(dim=1).item()
        npu_cls = npu_outputs[1].argmax(axis=1)[0]
        if cpu_cls != npu_cls:
            mismatches += 1
    print(f'  Mismatches in first 100: {mismatches}')
    print(f'  Agreement: {100-mismatches}%')

    # Summary
    print('\n' + '=' * 60)
    print('SUMMARY')
    print('=' * 60)
    speedup = cpu_elapsed / npu_elapsed
    print(f'  Dataset: 3394 samples (8ch x 200ts), 3 classes')
    print(f'  CPU:  {cpu_acc:.2f}% acc, {cpu_elapsed:.2f}s total, {cpu_avg:.1f} ms/sample')
    print(f'  NPU:  {npu_acc:.2f}% acc, {npu_elapsed:.2f}s total, {npu_avg:.1f} ms/sample')
    print(f'  Speedup: {speedup:.1f}x')
    print(f'  NPU utilization: avg {npu_mean_ld:.0f}%, peak {npu_max_ld}%')

    rknn.release()


if __name__ == '__main__':
    main()
