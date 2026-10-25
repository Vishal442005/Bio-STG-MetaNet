import sys, os
sys.path.insert(0, os.path.abspath("."))
from src.data.openneuro_processor import load_openneuro_cross_dataset

X, y, subs = load_openneuro_cross_dataset()
print("OpenNeuro cohort ready:")
print("  Shape:", X.shape)
print("  Labels:", np.bincount(y))
print("  Unique subjects:", len(np.unique(subs)))
