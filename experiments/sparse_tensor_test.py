import time

import numpy as np
import scipy.sparse as sp
import torch

from rlaopt.sparse import SparseCSRTensor

torch_precision = torch.float64
np_precision = np.float32 if torch_precision == torch.float32 else np.float64

torch.set_default_dtype(torch_precision)

X = sp.load_npz("yelp_train.npz").astype(np_precision)
# device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

X_torch = SparseCSRTensor(data=X, device=device)
X_torch_T = X_torch.T

# Test csr row slicing
idx = torch.tensor(np.arange(1024), dtype=torch.int64, device=device)
b = torch.randn(X_torch.shape[1], 10, device=device)
assert torch.allclose(X_torch[idx] @ b, (X_torch @ b)[idx])

# Test csc matrix multiplication
d = torch.randn(X_torch_T.shape[1], device=device)
ts = time.time()
result = X_torch_T @ d
print("Time taken for csc matvec (extension):", time.time() - ts)

d_np = d.cpu().numpy()
ts = time.time()
result_np = X.T @ d_np
print("Time taken for csc matvec (scipy):", time.time() - ts)
result_np = torch.tensor(result_np, device=device)

assert torch.allclose(result, result_np)

# D = torch.randn(X_torch_T.shape[1], 256, device=device)
D = torch.randn(X_torch_T.shape[1], 32, device=device)

ts = time.time()
result = X_torch_T @ D
print("Time taken for csc matmat (extension):", time.time() - ts)

D_np = D.cpu().numpy()
ts = time.time()
result_np = X.T @ D_np
print("Time taken for csc matmat (scipy):", time.time() - ts)
result_np = torch.tensor(result_np, device=device)

assert torch.allclose(result, result_np)
