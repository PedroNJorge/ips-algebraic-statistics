from src.ips import IPS
from src.gis import GIS, RBI
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
    gMLE = RBI(A, norm_u, h)
    print(f"IPS: MLE = {MLE}")
    print(f"GIS: MLE = {gMLE}")

    # GIS: Staged Tree Model
    A = [
        np.array([
            [1, 1, 1, 0, 0],
            [0, 0, 0, 1, 1],
        ]),
        np.array([
            [2, 1, 0, 1, 0],
            [0, 1, 2, 0, 1],
        ])
    ]
    h = np.array([1, 2, 1, 1, 1])
    norm_u = np.array([1, 2, 3, 4, 5])
    gMLE = RBI(A, norm_u, h)
    print(f"GIS: MLE = {gMLE}")


if __name__ == "__main__":
    main()
