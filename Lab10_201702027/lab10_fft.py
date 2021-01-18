import numpy as np
import scipy.signal as sp
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from scipy.fftpack import dct
import matplotlib.font_manager as fm
fontprop = fm.FontProperties(fname="Jalnan.ttf",size =15)

#정현파 조합
N = 1024 #
T = 1.0 / 44100.0 #1초에 44100개 샘플링
f1 = 697 #헤르츠(Hz)
f2 = 1209
t = np.linspace(0.0, N*T, N) #샘플링 저장배열
y1 = 1.1 * np.sin(2 * np.pi * f1 * t) #샘플링결과(sin함수)
y2 = 0.9 * np.sin(2 * np.pi * f2 * t)
y = y1 + y2

plt.subplot(311)
plt.plot(t, y1)
plt.title(r"$1.1\cdot\sin(2\pi\cdot 697t)$")
plt.subplot(312)
plt.plot(t, y2)
plt.title(r"$0.9\cdot\sin(2\pi\cdot 1209t)$")
plt.subplot(313)
plt.plot(t, y)
plt.title(r"$1.1\cdot\sin(2\pi\cdot 697t) + 0.9\cdot\sin(2\pi\cdot 1209t)$")
plt.tight_layout()
plt.show()

#fft
y2 = np.hstack([y, y, y])

plt.subplot(211)
plt.plot(y2)
plt.axvspan(N, N * 2, alpha=0.3, color='green')
plt.xlim(0, 3 * N)

plt.subplot(212)
plt.plot(y2)
plt.axvspan(N, N * 2, alpha=0.3, color='green')
plt.xlim(900, 1270)

plt.show()

yf = fft(y, N)
xf = np.linspace(0.0, 1.0/(2.0*T), N//2)

plt.stem(xf, 2.0/N * np.abs(yf[0:N//2]))
plt.xlim(0, 3000)

plt.show()

#dct
dct_type = 2
yf2 = dct(y, dct_type, N)

plt.subplot(311)
plt.stem(np.real(yf))
plt.title("DFT 실수부",fontproperties = fontprop)

plt.subplot(312)
plt.stem(np.imag(yf))
plt.title("DFT 허수부",fontproperties = fontprop)

plt.subplot(313)
plt.stem(np.abs(yf2))
plt.title("DCT")

plt.tight_layout()
plt.show()





