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
            return p
        eta = F(eta, t)
        l += 1
    else:
        print(f"Could not converge in {l} steps")


def preprocess_block(A: np.ndarray, b: np.ndarray):
    b = b.flatten()

    # Nonnegative
    xi = min(np.min(A), np.min(b))
    if xi < 0:
        A = A - xi
        b = b - xi

    # Scale
    xi_prime = max(np.max(np.sum(A, axis=0)), np.sum(b))
    if xi_prime > 0:
        A = A / xi_prime
        b = b / xi_prime

    # Add row to make column sums = 1
    col_sums = np.sum(A, axis=0)
    b_sum = np.sum(b)

    if not np.allclose(col_sums, 1) or not np.allclose(b_sum, 1):
        A = np.vstack([A, 1 - col_sums])
        b = np.hstack([b, 1 - b_sum])

    return A, b


def RBI(A: list[np.ndarray], u: np.ndarray, h: np.ndarray) -> np.ndarray:
    p = h / np.sum(h)
    u = u / np.sum(u)
    preprocessed_blocks: list[tuple[np.ndarray]] = []
    for A_k in A:
        preprocessed_blocks.append(preprocess_block(A_k, A_k @ u))

    l = 0
    MAX_STEPS = 100
    CONVERGENCE_TOL = 1e-12
    while l < MAX_STEPS:
        p_old = p.copy()
        for A_k, t_obs in preprocessed_blocks:
            t = A_k @ p
            p = p * np.exp(A_k.T @ np.log(t_obs / t))

        if np.max(np.abs(p - p_old)) < CONVERGENCE_TOL:
            print(f"Converged in {l} iterations")
            return p
        l += 1
    else:
        print(f"Could not converge in {l} steps")
