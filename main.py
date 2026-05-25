from src.ips import IPS
from src.gis import GIS
import numpy as np


def main():
    # IPS vs GIS: Independence model (Square)
    norm_u = np.array([1/10, 2/10, 3/10, 4/10])

    # IPS
    A = [
        np.array([
            [1, 1, 0, 0],
            [0, 0, 1, 1],
        ]),
        np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
        ])
    ]
    MLE = IPS(A, norm_u)

    # GIS
    A = [
        np.array([
            [1, 1, 0, 0],
            [0, 0, 1, 1],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
        ])
    ]
    h = np.ones(4)
    gMLE = GIS(A, norm_u, h)
    print(f"IPS: MLE = {MLE}")
    print(f"GIS: MLE = {gMLE}")

    # GIS: Staged Tree Model
    A = [
        np.array([
            [1, 1, 1, 0, 0],
            [0, 0, 0, 1, 1],
            [2, 1, 0, 1, 0],
            [0, 1, 2, 0, 1],
        ], dtype=float)
    ]
    h = np.array([1, 2, 1, 1, 1])


if __name__ == "__main__":
    main()
