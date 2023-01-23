# 3. 次の制約条件付き非線型最適化問題 (最小化) を解く.
# 目的関数: z = Σ_{i = 0}^{7} wi*di
# 制約条件: di >= √((X - xi)^2 + (Y - yi)^2)
# 最適解: (X, Y) ≒ (72.41, 12.55)
# 必要なモジュールをインポートする
import matplotlib.pyplot as plt
import numpy as np
import picos as pic

# 2次錐最適化問題を設定する
socp = pic.Problem()
# 家の番号i
n = 7
hs = np.array(list(range(n)))
# 家のx座標xi
xs = np.array([44, 64, 67, 83, 36, 70, 88, 58])
# 家のy座標yi
ys = np.array([47, 67, 9, 21, 87, 88, 12, 65])
# 家の座標s(xi, yi)
xys = np.array(list(zip(xs, ys)))
# ゴミの量wi
ws = [1, 2, 2, 1, 2, 5, 4, 1]
# 最適解となる変数(X, Y)
Xs = socp.add_variable("X", 2)
# ゴミ集積所から各家までの距離di
ds = [socp.add_variable(f"ds[{i}]", 1) for i in hs]
# 目的関数
objective = sum(w * d for w, d in zip(ws, ds))
# 目的関数および最小化を問題として設定する
socp.set_objective("min", objective)
# 制約条件を追加する
socp.add_list_of_constraints([abs(xy - Xs) <= d for xy, d in zip(xys, ds)])
# 2次錐最適化問題を解く
res = socp.solve(solver="cvxopt")
# 結果を表示する
print(f"最適解: (X, Y) = ({Xs.value[0]}, {Xs.value[1]})")
# 結果を図示する
plt.title("06-03. Second Order Conic Optimization")
plt.xlabel("x")
plt.ylabel("y")
plt.scatter(xs, ys, marker="+")
plt.scatter(Xs.value[0], Xs.value[1], marker="o")
plt.show()
