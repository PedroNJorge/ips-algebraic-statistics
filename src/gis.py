import numpy as np
from scipy.special import logsumexp


def compute_t(A: np.ndarray, h: np.ndarray, eta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    log_phi = np.log(h) + A.T @ eta
    log_p = log_phi - logsumexp(log_phi)
    p = np.exp(log_p)

    return A @ p, p


def GIS(A: list[np.ndarray], u: np.ndarray, h: np.ndarray) -> np.ndarray:
    A = A[0]
    n, m = A.shape
    l = 0
    eta = np.zeros(m)
    t_obs = A @ u
    a = np.sum(A[:, 0])
    assert np.allclose(np.sum(A, axis=0), a), "A must have constant column sum"

    def F(eta: np.ndarray, t: np.ndarray) -> np.ndarray:
        return eta + (1.0 / a) * np.log(t_obs / t)

    MAX_STEPS = 100
    CONVERGENCE_TOL = 1e-12
    while l < MAX_STEPS:
        t, p = compute_t(A, h, eta)

        if np.max(np.abs(t_obs - t)) < CONVERGENCE_TOL:
            print(f"Converged in {l} iterations")
            return p, eta
        eta = F(eta, t)
        l += 1
    else:
        print(f"Could not converge in {l} steps")
