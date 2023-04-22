# 1. 次の目的関数f(x)に関する線形最適化問題を解け.
import numpy as np
import pulp

# (1)目的関数: f(x) = x1 + x2の最大化
# 制約条件: {x1 + x2 >= 1, x1 - x2 >= 0, x1, x2 >= 0}
# ※恐らくunbounded?最小化の誤り?
prob = pulp.LpProblem(name="07-01-01", sense=pulp.LpMaximize)
Ass = np.array(
    [
        [1, 1],
        [1, -1],
    ]
)
bs = np.array([1, 0])
cs = np.array([1, 1])
_, n = Ass.shape
xs = [pulp.LpVariable(f"x{i+1}", lowBound=0) for i in range(n)]
prob += pulp.lpDot(cs, xs)
for As, b in zip(Ass, bs):
    prob += pulp.lpDot(As, xs) >= b
prob.solve()
print("(1)")
print(f"最適値: {pulp.value(prob.objective)}")
print("最適解:")
for var in prob.variables():
    print(f"{var} = {pulp.value(var)}")

# (2)目的関数: f(x) = 3x1 - 7x2 + 13x2 - 14x4
# 制約条件: {x1 - x2 + x3 - 2x4 = 1, x2 - 2x3 + 2x4 = 4, x2 - 2x3 - x4 >= -6, xj >= 0 (j = 1, ..., 4)}
prob = pulp.LpProblem(name="07-01-02", sense=pulp.LpMinimize)
Ass = np.array(
    [
        [1, -1, 1, -2],
        [0, 1, -2, 2],
        [0, 1, -2, -1],
    ]
)
bs = np.array([1, 4, -6])
cs = np.array([3, -7, 13, -14])
_, n = Ass.shape
xs = [pulp.LpVariable(f"x{i+1}", lowBound=0) for i in range(n)]
prob += pulp.lpDot(cs, xs)
for i, (As, b) in enumerate(zip(Ass, bs)):
    if i < 2:
        prob += pulp.lpDot(As, xs) == b
    else:
        prob += pulp.lpDot(As, xs) >= b
prob.solve()
print("(2)")
print(f"最適値: {pulp.value(prob.objective)}")
print("最適解:")
for var in prob.variables():
    print(f"{var} = {pulp.value(var)}")
