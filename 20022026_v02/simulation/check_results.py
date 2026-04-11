import pickle
import numpy as np
import os

results_dirs = []
for root, dirs, files in os.walk("Results"):
    for f in files:
        if f.endswith(".p"):
            results_dirs.append(os.path.join(root, f))

for path in results_dirs:
    try:
        data = pickle.load(open(path, "rb"))
        arr = np.array(data)
        nan_pct = np.sum(np.isnan(arr)) / arr.size * 100 if np.issubdtype(arr.dtype, np.floating) else 0
        print(f"{path}: shape={arr.shape}, dtype={arr.dtype}, NaN%={nan_pct:.1f}%")
    except Exception as e:
        print(f"{path}: ERROR - {e}")


info = pickle.load(open("Results/Central/info.p", "rb"))
dlmp = pickle.load(open("Results/Central/DLMPs.p", "rb"))

print("info keys:", info.keys() if isinstance(info, dict) else type(info))
print("dlmp shape:", np.array(dlmp).shape)

mc = pickle.load(open("Results/Central/mc.p", "rb"))
print("mc columns:", mc.columns.tolist())
print("mc shape:", mc.shape)
print(mc.head())

# Check what price arrays are available
print("\nFiles in Results/Central:")
for f in os.listdir("Results/Central"):
    print(" ", f)

"""# Export for plotting
np.save("Results/Central/dlmp_array.npy", dlmp)"""

import pickle
import numpy as np

for market in ['Central', 'P2P', 'ToU']:
    print(f"\n=== {market} ===")
    try:
        mc = pickle.load(open(f"Results/{market}/mc.p", "rb"))
        print(f"mc shape: {mc.shape}")
        print(f"mc columns: {mc.columns.tolist()}")
        print(f"mc price range: [{mc['price'].min():.4f}, {mc['price'].max():.4f}]")
        print(f"mc energy sum: {mc['energy'].sum():.4f}")
    except Exception as e:
        print(f"mc error: {e}")