# 1. 制約条件
# {2x1 + x2 = 3, x1 + 2x2 = 3, x1 >= 0, x2 >= 0}
# に対して，目的関数
# x1 + x2
# を最大化せよ．
import numpy as np
import pulp

prob = pulp.LpProblem(name="01-01", sense=pulp.LpMaximize)
Ass = np.array(
    [
        [2, 1],
        [1, 2],
    ]
)
bs = np.array([3, 3])
cs = np.array([1, 1])
_, n = Ass.shape
xs = [pulp.LpVariable(f"x{i+1}", lowBound=0) for i in range(n)]
prob += pulp.lpDot(cs, xs)
for As, b in zip(Ass, bs):
    prob += pulp.lpDot(As, xs) == b
prob.solve()
print(f"最適値: {pulp.value(prob.objective)}")
print("最適解:")
for var in prob.variables():
    print(f"{var} = {pulp.value(var)}")
