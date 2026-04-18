import numpy as np
register=np.array([1,0,1,0,1,1,0,0,1,1,1,0,0,0,0,1]) #配列(1,0,1,0,1,1,0,0,1,1,1,0,0,0,0,1)をregisterに代入
N=8
M=1000
from statistics import mean, median,variance,stdev
a=np.zeros(N) #要素数8の配列を0で初期化したものをaに代入
result=np.zeros(M) #要素数1000の配列を0で初期化したものをresultに代入
for j in range(M):
  for i in range(N):
    SR=(register[10]+register[12]+register[13]+register[15])%2 #register10,12,13,15の値をそれぞれ足し合わせた数値を2で割った余りをSRに代入
    a[i]=register[15] #register15の値をa[i]として外へ出力する
    register=np.roll(register,1) #配列の要素を右へ1つずらした配列をregisterに代入
    register[0]=SR #上記で求めたSRをregister0に代入
  d=0
  p=1
  for i in range(N):
    d+=p*a[i]
    p*=2 
  result[j]=d
print("平均",mean(result))
print("標準偏差 ",stdev(result))
print("中央値",median(result))
print("分散",variance(result))
print(np.sqrt(variance(result)))

print("1が出力される確率",np.sum(a)/N)
