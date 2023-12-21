import numpy as np
import matplotlib.pyplot as plt


def F(x, y):
    return -(
        x**2
        + 3 * y**4
        - 0.2 * np.cos(3 * np.pi * x)
        - 0.4 * np.cos(4 * np.pi * y)
        + 0.6
    )


Ant = 300  # 蚂蚁数量
Times = 80  # 移动次数
Rou = 0.9  # 荷尔蒙发挥系数
P0 = 0.2  # 转移概率
Lower_1, Upper_1 = -1, 1  # 搜索范围
Lower_2, Upper_2 = -1, 1

X = np.random.uniform(low=Lower_1, high=Upper_1, size=(Ant, 2))
Tau = np.array([F(x[0], x[1]) for x in X])

step = 0.05
x, y = np.meshgrid(
    np.arange(Lower_1, Upper_1 + step, step), np.arange(Lower_2, Upper_2 + step, step)
)
z = F(x, y)
fig = plt.figure(1)
ax1 = fig.add_subplot(121, projection="3d")
ax1.plot_surface(x, y, z, cmap="viridis")
ax1.scatter(X[:, 0], X[:, 1], Tau, c="k", marker="*")
ax1.text(0.1, 0.8, -0.1, "Ants' Initial Positions")
ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.set_zlabel("f(x,y)")

for T in range(1, Times + 1):
    lamda = 1 / T
    Tau_Best, BestIndex = max(Tau), np.argmax(Tau)

    P = (Tau_Best - Tau) / Tau_Best  # 计算转移状态概率

    for i in range(Ant):
        if P[i] < P0:  # 局部搜索
            temp1 = X[i, 0] + (2 * np.random.rand() - 1) * lamda
            temp2 = X[i, 1] + (2 * np.random.rand() - 1) * lamda
        # else:  # 全局搜索
        #     temp1 = X[i, 0] + (Upper_1 - Lower_1) * (np.random.rand() - 0.5)
        #     temp2 = X[i, 1] + (Upper_2 - Lower_2) * (np.random.rand() - 0.5)

        temp1 = np.clip(temp1, Lower_1, Upper_1)  # 越界处理
        temp2 = np.clip(temp2, Lower_2, Upper_2)

        if F(temp1, temp2) > F(X[i, 0], X[i, 1]):  # 更新位置
            X[i, 0] = temp1
            X[i, 1] = temp2

    # Tau = (1 - Rou) * Tau + np.array([F(x[0], x[1]) for x in X])  # 更新荷尔蒙

ax2 = fig.add_subplot(122, projection="3d")
ax2.plot_surface(x, y, z, cmap="viridis")
ax2.scatter(X[:, 0], X[:, 1], [eval(f"{F(x[0], x[1])}") for x in X], c="k", marker="*")
ax2.text(0.1, 0.8, -0.1, "Ants' Final Positions")
ax2.set_xlabel("x")
ax2.set_ylabel("y")
ax2.set_zlabel("f(x,y)")

max_index = np.argmax(Tau)
maxX, maxY = X[max_index, 0], X[max_index, 1]
maxValue = F(maxX, maxY)

plt.show()
