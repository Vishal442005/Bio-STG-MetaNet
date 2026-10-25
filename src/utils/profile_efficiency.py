"""
Computational Efficiency Profiling Module.
Measures parameters, inference latency (GPU/CPU), throughput, and memory footprint.
"""

import time
import torch
import numpy as np
from src.models import get_model


def profile_model_efficiency(model_name, num_channels=61, time_steps=256, device="cuda", n_runs=100):
    """
    Profiles parameters, inference latency (mean ± std ms), and throughput.
    """
    dev = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
    model = get_model(model_name, num_channels=num_channels, time_steps=time_steps).to(dev)
    model.eval()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Dummy input
    x = torch.randn(1, num_channels, time_steps, device=dev)
    
    # Warmup
    with torch.no_grad():
        for _ in range(15):
            _ = model(x)
            
    # Measure latency
    latencies = []
    if dev.type == "cuda":
        torch.cuda.synchronize()
        
    with torch.no_grad():
        for _ in range(n_runs):
            t0 = time.perf_counter()
            _ = model(x)
            if dev.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0) # in ms
            
    mean_lat = np.mean(latencies)
    std_lat = np.std(latencies)
    throughput = 1000.0 / mean_lat # trials per second
    
    return {
        "model": model_name,
        "device": str(dev),
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "latency_ms_mean": float(mean_lat),
        "latency_ms_std": float(std_lat),
        "throughput_samples_per_sec": float(throughput)
    }
