#x1^2+3/2*x2^2-x1x2-2x1-4x2を2x1+3x2<=6,x1+4x2<=5,x1>=0,x2>=0の制約条件のもとで最小化する
#最適解(x1,x2)=(1.18604651(=51/43), 0.95348837(=41/43))で最適値-4.5465116279069555
#必要なモジュールをインポート

from scipy.optimize import minimize
import numpy

#　目的関数を定義
def objective_fnc(x):
    x1 = x[0]
    x2 = x[1]
    return x1**2 + 3*x2**2/2 -x1*x2 -2*x1 -4*x2

# 不等式制約条件1 2x1+3x2-6>=0
def inequality_constraint(x):
    x1 = x[0]
    x2 = x[1]
    return - 2*x1 - 3*x2 + 6

# 不等式制約条件2 x1+4x2-5>=0
def inequality_constraint2(x):
    x1 = x[0]
    x2 = x[1]
    return - x1 - 4*x2 + 5

#x1,x2の定義域を0~100にすることで制約条件のx1>=0とx2>=0を満たすようにする
bounds_x1 = (0,100)
bounds_x2 = (0,100)
bound = [bounds_x1,bounds_x2]

constraint2 = {"type":"ineq","fun":inequality_constraint,"fun":inequality_constraint2}
constraint = [constraint2]

#初期点を設定
x0=[-1,-1]
#逐次二次計画法を実行
result=minimize(objective_fnc, x0, method="SLSQP", constraints=constraint)

#計算結果を表示
print(result)

#参考文献:Scipyで多変数関数の最小値を求める（逐次二次計画法の利用）(https://qiita.com/toneriver_py/items/f4f46bef9494d6b40b47)
