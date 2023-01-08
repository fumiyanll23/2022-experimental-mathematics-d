# 2. 次の制約条件なしの非線型最適化問題 (最小化) をNewton法で解く. ここでx = T(x1, x2, x3)である:
# 目的関数: f(x) = 100(x1^2 + x2)^2 + (x1 - 1)^2 + 100(x2^2 - x3^2)^2 + (x2 - 1)^2
# 最適解: (x1, x2, x3) = (1, 1, 1)
# 最適値: 0
# 必要なモジュールをインポートする
import numpy as np
import scipy.linalg
from nptyping import Float, NDArray, Shape


def f(xs: NDArray[Shape["3"], Float]) -> float:
    """
    目的関数
    """
    return sum(100 * (xs[i + 1] - xs[i] ** 2) ** 2 + (xs[i] - 1) ** 2 for i in range(2))


def grad_f(xs: NDArray[Shape["3"], Float]) -> NDArray[Shape["3"], Float]:
    """
    目的関数の勾配ベクトル
    """
    return np.array(
        [
            -400 * xs[0] * (-xs[0] ** 2 + xs[1]) + 2 * xs[0] - 2,
            -200 * xs[0] ** 2 - 400 * xs[1] * (-xs[1] ** 2 + xs[2]) + 202 * xs[1] - 2,
            -200 * xs[1] ** 2 + 200 * xs[2],
        ]
    )


def hessian_f(xs: NDArray[Shape["3"], Float]) -> NDArray[Shape["3, 3"], Float]:
    """
    目的関数のヘッセ行列
    """
    return np.array(
        [
            [1200 * xs[0] ** 2 - 400 * xs[1] + 2, -400 * xs[0], 0],
            [-400 * xs[0], 1200 * xs[1] ** 2 - 400 * xs[2] + 202, -400 * xs[1]],
            [0, -400 * xs[1], 200],
        ]
    )


# 初期点を設定する. ただし, 各成分の型にnp.float64を指定する
xs0 = np.array([10, 10, 10], dtype=np.float64)
# 許容誤差
eps = 1e-6
# 最大反復回数
iter_max = 100
# Newton法で最適解を求める
while scipy.linalg.norm(grad_f(xs0)) > eps:
    d = -np.dot(scipy.linalg.inv(hessian_f(xs0)), grad_f(xs0))
    xs0 += d
# 結果を表示する
print(f"最適解: x = {xs0}")
print(f"最適値: {f(xs0)}")
