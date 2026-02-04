from ips.algorithm import *

def main():
    # Independent model
    norm_u = np.array([1/10, 2/10, 3/10, 4/10])
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
    print(MLE)


if __name__ == "__main__":
    main()
