# 2. 次の最適化問題の最大値を，単体法を用いて求めよ．
import numpy as np
from scipy import linalg


def lp_revised_simplex(cs, Ass, bs, MEPS=1.0e-10):
    """
    線型最適化問題を解くための改訂シンプレックス法
    """
    np.seterr(divide="ignore")
    flag = 0
    m, n = Ass.shape
    AI = np.hstack((Ass, np.identity(m)))
    cs0 = np.r_[cs, np.zeros(m)]
    basis = [n + i for i in range(m)]
    nonbasis = [j for j in range(n)]
    while True:
        ys = linalg.solve(AI[:, basis].T, cs0[basis])
        cc = cs0[nonbasis] - np.dot(ys, AI[:, nonbasis])
        # 最適性判定
        if np.all(cc <= MEPS):
            flag = 1
            xs = np.zeros(n + m)
            xs[basis] = linalg.solve(AI[:, basis], bs)
            break
        else:
            s = np.argmax(cc)
        d = linalg.solve(AI[:, basis], AI[:, nonbasis[s]])
        # 非有界性判定
        if np.all(d <= MEPS):
            xs = None
            break
        else:
            bb = linalg.solve(AI[:, basis], bs)
            ratio = bb / d
            ratio[ratio < -MEPS] = np.inf
            r = np.argmin(ratio)
            # 基底と非基底の入れ替え
            nonbasis[s], basis[r] = basis[r], nonbasis[s]
    return flag, basis, cs0, xs


# (1) - 目的関数: z = 5x1 + 4x2
# - 制約条件: {5x1 + 2x2 <= 30, x1 + 2x2 <= 14, x1, x2 >= 0}
cs = np.array([5, 4])
Ass = np.array(
    [
        [5, 2],
        [1, 2],
    ]
)
bs = np.array([30, 14])
print("(1)")
flag, basis, cs0, xs = lp_revised_simplex(cs, Ass, bs)
if flag:
    print("Optimal")
    print(f"最適値: {np.dot(cs0[basis], xs[basis])}")
    for i in range(Ass.shape[0]):
        print(f"x{i} = {xs[i]}")
else:
    print("Unbounded")

# (2) - 目的関数: z = 2.5x1 + 5x2 + 3.4x3
# - 制約条件: {2x1 + 10x2 + 4x3 <= 425, 6x1 + 5x2 + 8x3 <= 400, 7x1 + 10x2 + 8x3 <= 600, x1, x2, x3 >= 0}
cs = np.array([2.5, 5, 3.4])
Ass = np.array(
    [
        [2, 10, 4],
        [6, 5, 8],
        [7, 10, 8],
    ]
)
bs = np.array([425, 400, 600])
flag, basis, cs0, xs = lp_revised_simplex(cs, Ass, bs)
print("(2)")
if flag:
    print("Optimal")
    print(f"最適値: {np.dot(cs0[basis], xs[basis])}")
    for i in range(Ass.shape[0]):
        print(f"x{i} = {xs[i]}")
else:
    print("Unbounded")
