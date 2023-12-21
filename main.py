import numpy as np
import matplotlib.pyplot as plt
from pylab import mpl
from G2D import G2D
from tqdm import tqdm


mpl.rcParams["font.sans-serif"] = ["SimHei"]

G = np.array(
    [
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    ]
)

MM = G.shape[0]
Tau = np.ones((MM * MM, MM * MM))
Tau = 8.0 * Tau
K = 100
M = 50
S = 1
E = MM * MM
Alpha = 1
Beta = 7
Rho = 0.3
Q = 1
minkl = np.inf
mink = 0
minl = 0
D = G2D(G)
N = D.shape[0]
a = 1

Ex = a * (np.mod(E, MM) - 0.5)
if Ex == -0.5:
    Ex = MM - 0.5
Ey = a * (MM + 0.5 - np.ceil(E / MM))

Eta = np.zeros(N)

for i in range(0, N):
    ix = a * (np.mod(i + 1, MM) - 0.5)
    if ix == -0.5:
        ix = MM - 0.5
    iy = a * (MM + 0.5 - np.ceil((i + 1) / MM))
    if i != E - 1:
        Eta[i] = 1 / ((ix - Ex) ** 2 + (iy - Ey) ** 2) ** 0.5
    else:
        Eta[i] = 100

ROUTES = [[0] * M for i in range(K)]

PL = np.zeros((K, M))

# 启动K轮蚂蚁觅食活动，每轮派出M只蚂蚁
for k in tqdm(range(0, K)):
    for m in range(0, M):
        # 状态初始化
        W = S
        Path = [S]
        PLkm = 0
        TABUkm = np.ones(N)
        TABUkm[S - 1] = 0
        DD = D.copy()
        # 下一步可以前往的节点
        DW = DD[W - 1]
        DW1 = np.argwhere(DW != 0)
        DW1 = np.reshape(DW1, (len(DW1),))
        for j in range(0, len(DW1)):
            if TABUkm[DW1[j]] == 0:
                DW[DW1[j]] = 0
        LJD = np.argwhere(DW != 0)
        LJD = np.reshape(LJD, (len(LJD),))
        Len_LJD = len(LJD)
        # 蚂蚁未遇到食物或者陷入死胡同或者觅食停止
        while W != E and Len_LJD >= 1:
            # 转轮赌法选择下一步怎么走
            PP = np.zeros(Len_LJD)
            for i in range(0, Len_LJD):
                PP[i] = (Tau[W - 1, LJD[i]] ** Alpha) * ((Eta[LJD[i]]) ** Beta)
            sumpp = sum(PP)
            PP = PP / sumpp
            Pcum = np.zeros(Len_LJD)
            Pcum[0] = PP[0]
            for i in range(1, Len_LJD):
                Pcum[i] = Pcum[i - 1] + PP[i]
            Select = np.argwhere(Pcum >= np.random.rand())
            Select = np.reshape(Select, (len(Select),))
            to_visit = LJD[Select[0]]
            # 状态更新和记录
            Path.append(to_visit + 1)
            PLkm = PLkm + DD[W - 1, to_visit]
            W = to_visit + 1
            for kk in range(0, N):
                if TABUkm[kk] == 0:
                    DD[W - 1, kk] = 0
                    DD[kk, W - 1] = 0
            TABUkm[W - 1] = 0
            DW = DD[W - 1]
            DW1 = np.argwhere(DW != 0)
            DW1 = np.reshape(DW1, (len(DW1),))
            for j in range(0, len(DW1)):
                if TABUkm[DW1[j]] == 0:
                    DW[j] = 0
            LJD = np.argwhere(DW != 0)
            LJD = np.reshape(LJD, (len(LJD),))
            Len_LJD = len(LJD)

        # 记下每一代每一只蚂蚁的觅食路线和路线长度
        ROUTES[k][m] = Path
        if Path[-1] == E:
            PL[k][m] = PLkm
            if PLkm < minkl:
                mink = k
                minl = m
                minkl = PLkm
        else:
            PL[k, m] = 0
    # 更新信息素
    Delta_Tau = np.zeros((N, N))
    for m in range(0, M):
        if PL[k][m] != 0:
            ROUT = ROUTES[k][m]
            TS = len(ROUT) - 1
            PL_km = PL[k][m]
            for s in range(0, TS):
                x = ROUT[s]
                y = ROUT[s + 1]
                Delta_Tau[x - 1][y - 1] = Delta_Tau[x - 1][y - 1] + Q / PL_km
                Delta_Tau[y - 1][x - 1] = Delta_Tau[y - 1][x - 1] + Q / PL_km
    Tau = (1 - Rho) * Tau + Delta_Tau

# 绘图

plotif = 1

if plotif == 1:
    minPL = np.zeros(K)
    for i in range(0, K):
        PLK = PL[i]
        Nonzero = np.argwhere(PLK != 0)
        Nonzero = np.reshape(Nonzero, (len(Nonzero),))
        PLKPLK = PLK[Nonzero]
        minPL[i] = np.min(PLKPLK)
    plt.figure(1)
    plt.plot(np.arange(0, K), minPL, "-b")
    plt.grid(True)
    plt.ylim(0, max(minPL) + 5)
    plt.xlim(0, K)
    plt.title("收敛曲线变化趋势")
    plt.xlabel("迭代次数")
    plt.ylabel("最小路径长度")
    plt.figure(2)
    plt.axis([0, MM, 0, MM])
    for i in range(0, MM):
        for j in range(0, MM):
            if G[i][j] == 1:
                x1 = j
                y1 = MM - i - 1
                x2 = j + 1
                y2 = MM - i - 1
                x3 = j + 1
                y3 = MM - i
                x4 = j
                y4 = MM - i
                plt.fill([x1, x2, x3, x4], [y1, y2, y3, y4], "black")
            else:
                x1 = j
                y1 = MM - i - 1
                x2 = j + 1
                y2 = MM - i - 1
                x3 = j + 1
                y3 = MM - i
                x4 = j
                y4 = MM - i
                plt.fill([x1, x2, x3, x4], [y1, y2, y3, y4], "white")
    plt.title("机器人运动轨迹")
    plt.xlabel("坐标x")
    plt.ylabel("坐标y")
    ROUT = ROUTES[mink][minl]
    LENROUT = len(ROUT)
    Rx = ROUT.copy()
    Ry = ROUT.copy()
    for ii in range(0, LENROUT):
        Rx[ii] = a * (np.mod(ROUT[ii], MM) - 0.5)
        if Rx[ii] == -0.5:
            Rx[ii] = MM - 0.5
        Ry[ii] = a * (MM + 0.5 - np.ceil(ROUT[ii] / MM))
    plt.plot(Rx, Ry, "r-")
    plt.show()
plotif2 = 1

if plotif2 == 1:
    plt.figure(3)
    plt.axis(np.array([0, MM, 0, MM]))
    for i in range(0, MM):
        for j in range(0, MM):
            if G[i][j] == 1:
                x1 = j
                y1 = MM - i - 1
                x2 = j + 1
                y2 = MM - i - 1
                x3 = j + 1
                y3 = MM - i
                x4 = j
                y4 = MM - i
                plt.fill([x1, x2, x3, x4], [y1, y2, y3, y4], "black")
            else:
                x1 = j
                y1 = MM - i - 1
                x2 = j + 1
                y2 = MM - i - 1
                x3 = j + 1
                y3 = MM - i
                x4 = j
                y4 = MM - i
                plt.fill([x1, x2, x3, x4], [y1, y2, y3, y4], "white")
    for k in range(0, K):
        PLK = PL[k, :]
        minPLK = np.min(PLK)
        pos = np.argwhere(PLK == minPLK)
        pos = np.reshape(pos, (len(pos),))
        m = pos[0]
        ROUT = ROUTES[k][m]
        LENROUT = len(ROUT)
        Rx = ROUT.copy()
        Ry = ROUT.copy()
        for ii in range(0, LENROUT):
            Rx[ii] = a * (np.mod(ROUT[ii], MM) - 0.5)
            if Rx[ii] == -0.5:
                Rx[ii] = MM - 0.5
            Ry[ii] = a * (MM + 0.5 - np.ceil(ROUT[ii] / MM))
        plt.plot(Rx, Ry, "r-")
    plt.show()
