import numpy as np


# ANSI color codes for printing text
RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
RESET = '\033[0m'


def info_proj_L(p: np.ndarray, A: np.ndarray, u: np.ndarray) -> np.ndarray:
    """
    Provides the information projection of p to the linear family L^i:
        L^i = {p in simplex | A^i p = A^i u}, A^i is the i-th partition

    Args:
        p: Distribution to project.
        A: Partition to use.
        u: Vector of counts of the observed data.

    Returns:
        Projected distribution.
    """
    m = p.shape[0]
    proj = np.zeros(m)

    for j in range(m):
        # Find row of A that has j in it's support
        for row in A:
            if row[j] != 0:
                a_S = row
                break

        proj[j] = p[j] * np.dot(a_S, u) / np.dot(a_S, p)

    return proj


def IPS(A: list[np.ndarray], u: np.ndarray) -> np.ndarray:
    """
    Args:
        A: Multipartition matrix. List of k ndarray matrices of shape [n, m].
            Note: A[i] can have a different shape in comparison to A[j].
        u: Vector of counts of the observed data.

    Returns:
        The MLE, p* distribution, of the statistical model associated to A.
    """
    k = len(A)  # Number of partitions
    m = u.shape[0]  # Distribution p* lives in the m-1 simplex
    l = 0  # Current iteration
    p_prev = np.full(m, 1/m, dtype=float)  # p^0
    p = np.zeros(m)

    MAX_STEPS = 100
    CONVERGENCE_TOL = 1e-12
    while l < MAX_STEPS:
        partition_idx = l % k
        p = info_proj_L(p_prev, A[partition_idx], u)

        if np.max(np.abs(p - p_prev)) < CONVERGENCE_TOL:
            print(f"{GREEN}Converged in {l} steps!{RESET}")
            return p

        print(f"{YELLOW}Step {l} produced {p} using A^{partition_idx + 1}{RESET}")
        p_prev = p.copy()
        l += 1
    else:
        print(f"{RED}Could not converge in {l} steps!{RESET}")
    return p
