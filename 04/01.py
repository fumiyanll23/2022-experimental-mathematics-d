#必要なモジュールをインポート
import numpy as np

def f(x0, x1, x2):
    #目的関数の定義
    y = x0**2 + x1**2 - x2*2 + 4*x0*x2 + 4*x1*x2 -3*x0 +2*x1 + x2 - 6
    #勾配ベクトルの定義
    dydxdz = np.array([2*x0 + 4*x2 - 3,2*x1 + 4*x2 +2,-2 + 4*x0 +4*x1+1])
    #ヘッセ行列の定義
    H = np.array([[2, 0, 4],[0, 2, 4],[4, 4, 0]])
    return y, dydxdz, H
#初期座標を設定
x0, x1, x2 = 1, 1, 1
#ステップ数、各種変数の推移を表示
print("i    x1          x2          x3          f(x)")
#ニュートン法を実行
for i in range(100):
    y, dydxdz, H = f(x0, x1, x2)
    print(f"{i:3d} [{x0:10.3e}, {x1:10.3e},{x2:10.3e}], {y:10.3e}")
    d = - np.dot(np.linalg.inv(H), dydxdz)
    x0 += d[0]
    x1 += d[1]
    x2 += d[2]

#https://helve-blog.com/posts/math/newtons-method-python/　参照
