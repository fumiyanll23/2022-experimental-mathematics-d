# 2. 次の離散最適化問題を解く． (ビンパッキング問題) 以下の表のように, 重さがwj (kg) である品物jがある (j = 1, 2 · · · , 10). これらを容量C = 25の容器に入れて運びたい. 必要な容器の最小数を求めよ.
# 目的関数: f(x) = Σ_{J ∈ \mathcal{N}} xJの最小化
# 制約条件: Σ_{J ∈ \mathcal{N}} a_{i, J} xJ >= 1 (i ∈ N), xJ ∈ {0, 1} (J ∈ \mathcal{N})
# ※ここで,
# - N = {1, 2, ..., 10},
# - \mathcal{N} = {J ⊂ N | Σ_{j ∈ J} w_{j} <= C},
# - a_{i, J} = {
#     1 (i ∈ J のとき),
#     0\ (i ∉ J のとき),
#     }
# である.
# 必要なモジュールをインポートする
from collections import deque

import numpy as np
import pulp


class KnapsackProblem(object):
    """
    ナップザック問題の定義
    """

    def __init__(self, name, capacity, items, costs, weights, zeros=set(), ones=set()):
        self.name = name
        self.capacity = capacity
        self.items = items
        self.costs = costs
        self.weights = weights
        self.zeros = zeros
        self.ones = ones
        self.lb = -100
        self.ub = -100
        ratio = {j: costs[j] / weights[j] for j in items}
        self.sitemList = [
            k for k, _ in sorted(ratio.items(), key=lambda x: x[1], reverse=True)
        ]
        self.xlb = {j: 0 for j in self.items}
        self.xub = {j: 0 for j in self.items}
        self.bi = None

    def getbounds(self):
        """
        上限および下限を求める
        """
        for j in self.zeros:
            self.xlb[j] = self.xub[j] = 0
        for j in self.ones:
            self.xlb[j] = self.xub[j] = 1
        if self.capacity < sum(self.weights[j] for j in self.ones):
            self.lb = self.ub = -100
            return 0
        ritems = self.items - self.zeros - self.ones
        sitems = [j for j in self.sitemList if j in ritems]
        cap = self.capacity - sum(self.weights[j] for j in self.ones)
        for j in sitems:
            if self.weights[j] <= cap:
                cap -= self.weights[j]
                self.xlb[j] = self.xub[j] = 1
            else:
                self.xub[j] = cap / self.weights[j]
                self.bi = j
                break
        self.lb = sum(self.costs[j] * self.xlb[j] for j in self.items)
        self.ub = sum(self.costs[j] * self.xub[j] for j in self.items)

    def __str__(self):
        """
        KnapSackProblemの情報を表示する
        """
        return (
            "Name = " + self.name + ", capacity = " + str(self.capacity) + ",\n"
            "items = "
            + str(self.items)
            + ",\n"
            + "costs = "
            + str(self.costs)
            + ",\n"
            + "weights = "
            + str(self.weights)
            + ",\n"
            + "zeros = "
            + str(self.zeros)
            + ", ones = "
            + str(self.ones)
            + ",\n"
            + "lb = "
            + str(self.lb)
            + ", ub = "
            + str(self.ub)
            + ",\n"
            + "sitemList = "
            + str(self.sitemList)
            + ",\n"
            + "xlb = "
            + str(self.xlb)
            + ",\n"
            + "xub = "
            + str(self.xub)
            + ",\n"
            + "bi = "
            + str(self.bi)
            + "\n"
        )


def KnapsackProblemSolve(capacity, items, costs, weights):
    queue = deque()
    root = KnapsackProblem(
        "KP",
        capacity=capacity,
        items=items,
        costs=costs,
        weights=weights,
        zeros=set(),
        ones=set(),
    )
    root.getbounds()
    best = root
    queue.append(root)
    while queue != deque([]):
        p = queue.popleft()
        p.getbounds()
        # bestを更新する可能性がある
        if p.ub > best.lb:
            # bestを更新する
            if p.lb > best.lb:
                best = p
            # 子問題を作って分枝する
            if p.ub > p.lb:
                k = p.bi
                p1 = KnapsackProblem(
                    p.name + "+" + str(k),
                    capacity=p.capacity,
                    items=p.items,
                    costs=p.costs,
                    weights=p.weights,
                    zeros=p.zeros,
                    ones=p.ones.union({k}),
                )
                queue.append(p1)
                p2 = KnapsackProblem(
                    p.name + "-" + str(k),
                    capacity=p.capacity,
                    items=p.items,
                    costs=p.costs,
                    weights=p.weights,
                    zeros=p.zeros.union({k}),
                    ones=p.ones,
                )
                queue.append(p2)
    return "Optimal", best.lb, best.xlb


def binpacking(
    capacity: int, w: dict[int, int], MEPS: float = 1.0e-8
) -> tuple[bool, list[list[int]]]:
    m = len(w)
    items = set(range(m))
    A = np.identity(m)
    solved = False
    columns = 0
    dual = pulp.LpProblem(name="D(K)", sense=pulp.LpMaximize)
    y = [pulp.LpVariable("y" + str(i), lowBound=0) for i in items]
    # 目的関数の設定
    dual += pulp.lpSum(y[i] for i in items)
    # 制約条件の付加
    for j in range(len(A.T)):
        dual += pulp.lpDot(A.T[j], y) <= 1, "ineq" + str(j)
    while not (solved):
        # 双対問題を解く
        dual.solve()
        costs = {i: y[i].varValue for i in items}
        weights = {i: w[i] for i in items}
        _, val, sol = KnapsackProblemSolve(capacity, items, costs, weights)
        if val >= 1.0 + MEPS:
            a = np.array([int(sol[i]) for i in items])
            dual += pulp.lpDot(a, y) <= 1, "ineq" + str(m + columns)
            A = np.hstack((A, a.reshape((-1, 1))))
            columns += 1
        else:
            solved = True
    m, n = A.shape
    primal = pulp.LpProblem(name="P(K)", sense=pulp.LpMinimize)
    x = [pulp.LpVariable("x" + str(j), lowBound=0, cat="Binary") for j in range(n)]
    # 目的関数の設定
    primal += pulp.lpSum(x[j] for j in range(n))
    for i in range(m):  # 制約条件の付加
        primal += pulp.lpDot(A[i], x) >= 1, "ineq" + str(i)
    primal.solve()
    flag_opt = False
    if pulp.value(primal.objective) - pulp.value(dual.objective) < 1.0 - MEPS:
        flag_opt = True
    K = [j for j in range(n) if x[j].varValue > MEPS]
    results = []
    itms = set(range(m))
    for j in K:
        J = {i for i in range(m) if A[i, j] > MEPS and i in itms}
        r = [w[i] for i in J]
        itms -= J
        results.append(r)

    return (flag_opt, results)


# 容量
capacity = 25
# 品物の個数
n = 10
items = set(range(n))
# 許容誤差
MEPS = 1.0e-8
np.random.seed(1)
# 品物の重量
w = {i: np.random.randint(5, 10) for i in items}
flag, ans = binpacking(capacity, w, MEPS)
# 結果を表示する
print(f"必要な容器の個数: {len(ans)}")
if flag:
    print("最適解: ", end="")
else:
    print("近似解: ", end="")
print(ans)
