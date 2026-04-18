# -*- coding: utf-8 -*-
import cmath
import math
import numpy
from scipy.stats import norm
import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt

#振幅値 
A = 1.0
#行列Jを指定
J = np.array([[0, 1] , [-1, 0]])
#単位行列Iを指定
Id = np.array([[1.0, 0] , [0 , 1.0]])  

class Qstate:
    """量子ガウス状態を表すクラス"""
    def __init__(self, c_amplitude = A + 0j ):
        """初期値の設定を追加 (sohma Aug 12 2018)"""
        # 複素振幅値
        self.alpha = c_amplitude

        # コヒーレント状態の正規分布の広がり(共分散行列)を表す        
        self.a = np.array([[0.25 , 0] , [0 , 0.25]])

        # 正規受信者の確率        
        self.Pr = 0.0
        
        
    def attenuate(self, att_rate):
        """減衰する
           aの値（共分散）と平均が変換される"""
        #減衰後の複素振幅を計算
        self.alpha = self.alpha* math.sqrt(att_rate)
        #減衰後の共分散行列を計算
        self.a = (att_rate * self.a) + ((1.0-att_rate) * Id / 4.0)


    def measure_homodyne(self, theta_val):
        """ホモダイン測定を行う"""     
        # 量子状態を元の位置まで回転させる
        rev_rot =  cmath.rect(1.0, theta_val)
        beta =  self.alpha * rev_rot
        U = np.array([[math.cos(theta_val) , -math.sin(theta_val)] , [math.sin(theta_val) , math.cos(theta_val)]])
        b = np.dot(U.T , np.dot(self.a , U))
        
        # 共分散行列 b の x軸方向の標準偏差
        dev_x = math.sqrt(b[0, 0])
        
        # 0が出力される確率（測定値が正になる確率）       
        self.Pr =  norm.sf(x = 0.0 , loc = beta.real ,scale = dev_x )
        return self.Pr
    
    def homodyne_measurement(self):
        dev_x=math.sqrt(self.a[0,0])
        res = np.random.normal(loc=np.real(self.alpha), scale = dev_x)
        return res

    def plot_distribution(self,ax,max_x,max_y,min_x,min_y,c_name="blue"):  
      x = np.arange(min_x, max_x, 0.01) # x点として[min_x, max_y]まで0.01刻みでサンプル
      y = np.arange(min_y, max_y, 0.01)  # y点として[min_x, max_y]まで0.01刻みでサンプル
      x, y = np.meshgrid(x, y)  # 上述のサンプリング点(x,y)を使ったメッシュ生成
      dev_x=math.sqrt(self.a[0,0])   #共分散行列を表す属性a[0,0]の平方根をdev_xに代入
      dev_y=math.sqrt(self.a[1,1])   #共分散行列を表す属性a[1,1]の平方根をdev_yに代入
      z = norm.pdf(x,self.alpha.real,dev_x)*norm.pdf(y,self.alpha.imag,dev_y)
      ax.set_zlim(0.0,1.0)
      ax.plot_wireframe(x, y, z, color=c_name,linewidth=0.3) # ワイヤーフレームのプロット。color,linewidthは曲面のメッシュの線の色と太さをそれぞれ表す。

      ax.tick_params(labelbottom="off",bottom="off") # x軸の削除
      ax.tick_params(labelleft="off",left="off") # y軸の削除
      #ax.set_xticklabels([]) 
      #ax.set_yticklabels([])
      ax.set_zticklabels([])

    def homodyne(self,thld):#arrがthld(しきい値)より大きければ1,小さければ0を出力する
      arr=np.random.normal(self.alpha.real,np.sqrt(self.a[0][0]),1) #平均self.alpha.real,標準偏差np.sqrt(self.a[0][0]),乱数を1個出力する正規分布をarrに代入
      #print("arr:",arr,"thld:",thld)
      if arr>thld:
        return 1
      else:
        return 0
         #1か0を出力する
