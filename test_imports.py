"""
Quick import test — run after pip install completes.
"""
import sys
failed = []
for pkg in ['torch', 'sklearn', 'scipy', 'matplotlib', 'pandas', 'numpy', 'seaborn']:
    try:
        __import__(pkg)
        print(f"  ✅ {pkg}")
    except ImportError as e:
        print(f"  ❌ {pkg}: {e}")
        failed.append(pkg)

if failed:
    print(f"\nMissing: {failed}")
    sys.exit(1)
else:
    import torch
    print(f"\nAll imports OK!")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.cuda.is_available()}")
    # Quick model test
    from eeg_master_pipeline import EnhancedMetaModel, count_params
    m = EnhancedMetaModel()
    x = torch.randn(2, 64, 256)
    out = m(x)
    print(f"Model params: {count_params(m):,}")
    print(f"Forward pass: {out.shape} ✅")
