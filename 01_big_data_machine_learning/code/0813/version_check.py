import sys, pandas as pd, numpy as np, locale
print(f"Python: {sys.version}")
print(f"Pandas: {pd.__version__}")  
print(f"NumPy: {np.__version__}")
print(f"Locale: {locale.getlocale()}")
print(f"Platform: {sys.platform}")

# bins의 정확한 값도 확인
bins = np.arange(156, 195, 5)
print(f"bins 타입: {type(bins)}")
print(f"bins 정밀도: {bins.dtype}")
print(f"첫 번째 값: {repr(bins[0])}")