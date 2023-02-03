#f(x)=50*x1+40*x2+10*x3+70*x4+55*x5を7*x1+5*x2+x3+9*x4+6*x5<=15,xj=0or1,j=1~5の条件のもと最大化する(ナップサック問題)。
#品物の種類,容量,品物の重さと価格をリスト表示して設定する。
n = 5
capacity = 15
size  = [7, 5, 1, 9, 6]
price = [50, 40, 10, 70, 55]

#重さと価格の最適解を記録する
max_size = -1
max_price = -1
combination = []

for i in range(2**n - 1) :
    tmp_sumS = 0
    tmp_sumP = 0
    tmp_comb = []
    over_flag = False

    for j in range(n) :
        # シフトして１ビットずつ判断
        is_put = i>>(n-j-1)&1
        tmp_comb.append(is_put)
        tmp_sumS += is_put * size[j]
        tmp_sumP += is_put * price[j]

        # capa を越えたらフラグを立てて break
        if tmp_sumS > capacity :
            over_flag = True
            break

    # over flag が立ってない かつ 暫定 max price より高いときに更新
    if (not over_flag) and tmp_sumP > max_price :
        max_price = tmp_sumP
        max_size = tmp_sumS
        combination = tmp_comb

#結果を表示
print("合計が最大になる組み合わせ")
print(combination)
print("合計価格: ", max_price)
print("合計サイズ: ", max_size)

# 参考文献:  Pythonでナップサック問題を総当たりで解く(https://zeronosu77108.hatenablog.com/entry/2017/12/25/131726)
