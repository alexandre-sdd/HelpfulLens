from pathlib import Path
from typing import Any

from scipy import sparse as _sp
import numpy as np


# ---------- Paths helpers ----------

def get_project_root() -> Path:
    """
    Infer project root by walking up until we find a 'data' folder.
    """
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "data").exists():
            return parent
    raise RuntimeError("Could not locate project root (no 'data/' folder found).")


# ---------- Array helpers ----------

def densify_sparse(X):
    """
    Convert a scipy sparse matrix to a dense numpy array when needed.
    This is safe to use in sklearn's FunctionTransformer.
    """
    return X.toarray() if hasattr(X, "toarray") else X


def to_csr_matrix(X: Any):
	"""
	Convert a dense 2D array-like into a CSR sparse matrix.
	If X is already a scipy sparse matrix, return as-is.
	"""
	if hasattr(X, "tocsr"):
		# Already a sparse matrix
		return X.tocsr()
	# Ensure numpy array (2D)
	arr = np.asarray(X)
	if arr.ndim == 1:
		arr = arr.reshape(-1, 1)
	return _sp.csr_matrix(arr)