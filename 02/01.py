# 1. 以下の線型最適化問題 (最大化) を解く:
# 目的関数: z = 150x_1 + 200x_2 + 300x_3
# 制約条件: 3x_1 + x_2 + x_3 <= 60, x_1 + 3x_2 <= 36, 2x_2 + 4x_3 <= 48 (x_1, x_2, x_3 >= 0)
# 必要なモジュールをインポートする
import numpy as np
import pulp

# 線型最適化問題 (最大化) を定義する
prob = pulp.LpProblem(name="02-01", sense=pulp.LpMaximize)
# 制約条件のパラメータを設定する
Ass = np.array([
    [3, 1, 2],
    [1, 3, 0],
    [0, 2, 4],
])
bs = np.array([60, 36, 48])
cs = np.array([150, 200, 300])
# 係数行列Aの行数mと列数n
m, n = Ass.shape
# 変数を定義する
xs = [pulp.LpVariable(f"x{i+1}", lowBound=0) for i in range(n)]
# 目的関数を設定する
prob += pulp.lpDot(cs, xs)
# 制約条件を設定する
for i, (As, b) in enumerate(zip(Ass, bs)):
    prob += pulp.lpDot(As, xs) <= b
# 問題を解く
prob.solve()
# 結果を表示する
print(prob)
print(pulp.LpStatus[prob.status])
print(f"最適値: z = {pulp.value(prob.objective)}")
print("最適解:")
for var in prob.variables():
    print(f"{var} = {pulp.value(var)}")
