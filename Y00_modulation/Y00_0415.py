##
# author   : Yuka Matsumae
# created  : 15.04.2026 
##

'''

Y-00 Quantum-Noise Randomized Stream Cipher
Using Intensity Modulation Signals for Physical
Layer Security of Optical Communications

https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9063467&tag=1
'''

import numpy as np
import matplotlib.pyplot as plt

#パラメータ

#minimum power P1
P1 = ??

#maximum power P2M
P2M = ??

#Planck constant h
h = ??

#electric charge e
e = ??

#optical light frequency v0
v0 = 193.4 THz

#optical power of P0
P0 = (P2M + P1)/2

#signal bandwidth B
B = 1.5 GHz

#lambda
lamda = 1550nm

#Number of bases M
M = 2^11

#ONSR optical signal-to-noise ratio
ONSR = 20dB??

#shot noise σ
noise_shot = e * numpy.sqrt(2 * P0 * B /(h v0))

#ΔP basis 隣り合う信号レベル間の距離
delta_P_basis = P2M - P1)/(2M - 1)

#gamma IM shot　ショット雑音限界における信号品質
gamma_IM_shot = (2 * noise_shot) /(delta_P_basis)

#ASE noise
noise_ASE = numpy.sqrt((2 * B * P_0^2)/(R_REF * OSNR))

#gamma_IM
gamma_IM = (2* noise_ASE) / delta_P_basis

