# x1^2+x2^2+x3^2を2x1+3x2+x3=6の制約条件のもとで最小化する
# 最適解(x1,x2,x3)=(0.85714285(=6/7), 1.28571429(=9/7), 0.42857143(=3/7))で最適値18/7(=2.5714285714285725)
# 必要なモジュールをインポート
import scipy


# 目的関数を定義
def objective_fnc(x):
    x1 = x[0]
    x2 = x[1]
    x3 = x[2]
    return x1**2 + x2**2 + x3**2


# 等式制約条件
def equality_constraint(x):
    x1 = x[0]
    x2 = x[1]
    x3 = x[2]
    return 2 * x1 + 3 * x2 + x3 - 6


constraint1 = {"type": "eq", "fun": equality_constraint}
constraint = [constraint1]
# 初期点を設定
x0 = [0, 0, 0]
# 逐次二次計画法を実行
result = scipy.optimize.minimize(
    objective_fnc, x0, method="SLSQP", constraints=constraint
)
# 計算結果を表示
print(result)

# 参考文献: Scipyで多変数関数の最小値を求める（逐次二次計画法の利用）(https://qiita.com/toneriver_py/items/f4f46bef9494d6b40b47)
