import numpy as np
import scipy.linalg
from scipy.linalg import sqrtm, svdvals

def fidelity(rho, sigma):
    """Calculate state fidelity between two density matrices."""
    sqrt_rho = scipy.linalg.sqrtm(rho)
    inner = sqrt_rho @ sigma @ sqrt_rho
    sqrt_inner = scipy.linalg.sqrtm(inner)
    return np.real(np.trace(sqrt_inner)) ** 2

def fidelity1(rho, sigma):
    d = rho.shape[0]
    return np.real((d * np.trace(rho @ sigma) + 1) / (d + 1))

def operator_fidelity(rho_ideal, rho_actual):
    """Calculate fidelity between two quantum channels."""
    d = rho_ideal.shape[0]
    fid = np.real((d * np.trace(rho_ideal @ rho_actual) + 1) / (d + 1))

    return fid

def trace_distance(rho: np.ndarray, sigma: np.ndarray, type: str = 'eig') -> float:

    if type == 'eig':
        delta = rho - sigma
        abs_delta = sqrtm(delta.conj().T @ delta)
        return 0.5 * np.trace(abs_delta).real

    elif type == 'svd':
        s = svdvals(rho - sigma)
        return 0.5 * np.sum(s)

    else:
        raise ValueError(f"Unknown method type: {type}")
    