import cv2
import numpy as np
from matplotlib import pyplot as plt
import math

# take input
img_input = cv2.imread('.\images\\two_noise.jpeg', 0)
img = img_input.copy()
image_size = img.shape[0] * img.shape[1]


#%%
# fourier transform
ft = np.fft.fft2(img)
ft_shift = np.fft.fftshift(ft)
#ft_shift = ft
magnitude_spectrum_ac = np.abs(ft_shift)
magnitude_spectrum = 20 * np.log(np.abs(ft_shift)+1)
magnitude_spectrum = cv2.normalize(magnitude_spectrum, None,0,255,cv2.NORM_MINMAX,dtype=cv2.CV_8U)
ang = np.angle(ft_shift)
ang_ = cv2.normalize(ang, None,0,255,cv2.NORM_MINMAX,dtype=cv2.CV_8U)
## phase add


notch = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        if (math.sqrt((i - 300) ** 2 + (j - 400) ** 2) <= 5) :
            notch[i, j] = 0

        else:
            notch[i, j] = 1

magnitude_spectrum1 = magnitude_spectrum * notch

final_result = np.multiply(magnitude_spectrum_ac, np.exp(1j*ang))

# inverse fourier
img_back = np.real(np.fft.ifft2(np.fft.ifftshift(final_result)))
img_back_scaled = cv2.normalize(img_back, None, 0,255,cv2.NORM_MINMAX,dtype=cv2.CV_8U)


## plot
cv2.imshow("input", img_input)
cv2.waitKey(0)

f1 = plt.figure(1)
plt.imshow(np.log(np.abs(ft_shift)), 'gray')
plt.show()

f1 = plt.figure(2)
plt.imshow(np.log(np.abs(magnitude_spectrum)), 'gray')
plt.show()

cv2.imshow("Magnitude Spectrum",magnitude_spectrum1)
cv2.waitKey(0)

cv2.imshow("Phase", ang_)
cv2.waitKey(0)

#cv2.imshow("Final_result", norm_fin)

cv2.imshow("Inverse transform",img_back_scaled)

cv2.waitKey(0)
cv2.destroyAllWindows()