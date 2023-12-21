import numpy as np


def G2D(G):
    l = G.shape[0]
    D = np.zeros((l * l, l * l))
    for i in range(0, l):
        for j in range(0, l):
            if G[i, j] == 0:
                for m in range(0, l):
                    for n in range(0, l):
                        if G[m, n] == 0:
                            im = np.abs(i - m)
                            jn = np.abs(j - n)
                            if im + jn == 1 or (im == 1 and jn == 1):
                                D[i * l + j, m * l + n] = (im + jn) ** 0.5

    return D
