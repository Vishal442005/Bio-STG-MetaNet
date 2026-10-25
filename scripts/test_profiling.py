import sys, os
sys.path.insert(0, os.path.abspath('.'))
from src.utils.profile_efficiency import profile_model_efficiency

for m in ['biostg', 'eegnet', 'shallow', 'cnnlstm']:
    res = profile_model_efficiency(m, num_channels=61, time_steps=256, device='cuda', n_runs=50)
    print(f"{res['model']:12s} -> Latency: {res['latency_ms_mean']:.2f} +- {res['latency_ms_std']:.2f} ms | Throughput: {res['throughput_samples_per_sec']:.1f} trials/s")
