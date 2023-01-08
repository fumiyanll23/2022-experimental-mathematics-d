# 3. 次の制約条件なしの非線型最適化問題 (最小化) を準Newton法で解く. ここでx = T(x1, x2)である:
# 目的関数: f(x) = (1.5 - x1 + x1x2)^2 + (2.25 - x1 + x1x2^2)^2 + (2.625 - x1 + x1x2^3)^2
# 最適解: (x1, x2) = (3, 0.5)
# 最適値: 0
# 必要なモジュールをインポートする
import numpy as np
import scipy
from nptyping import Float, NDArray, Shape


def f(xs: NDArray[Shape["3"], Float]) -> float:
    """
    目的関数
    """
    return (
        (1.5 - xs[0] + xs[0] * xs[1]) ** 2
        + (2.25 - xs[0] + xs[0] * (xs[1] ** 2)) ** 2
        + (2.625 - xs[0] + xs[0] * (xs[1] ** 3)) ** 2
    )


# 初期点を設定する
xs0 = np.array([0, 0])
# 準Newton法で大域的最適解を求める
ans = scipy.optimize.minimize(f, xs0, method="BFGS")
# 結果を表示する
print(ans)
print(f"最適解: x = {ans.x}")
print(f"最適値: {ans.fun}")
