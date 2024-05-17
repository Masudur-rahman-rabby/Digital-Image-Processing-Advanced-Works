import matplotlib.pyplot as plt
import cv2
import numpy as np


def histogram_equalization(image):
  hist, bins = np.histogram(image.flatten(), range(len(np.unique(image)) + 1))
  pdf = hist / np.sum(hist)
  cdf = np.cumsum(pdf)
  cdf_normalized = cdf / cdf[-1]
  transformation_function = np.round(cdf_normalized * (len(bins) - 1)).astype(np.uint8)
  equalized_image1 = transformation_function[image.flatten()]
  equalized_image = equalized_image1.reshape(image.shape)
  return equalized_image

image_path = '.\images\\histogram.jpg'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

hist1 = cv2.calcHist([image],[0],None,[256],[0,256])
plt.figure(1)
plt.plot(hist1)
plt.title("Original Image Histogram")
plt.show()

equalized_image = histogram_equalization(image)
hist2 = cv2.calcHist([equalized_image],[0],None,[256],[0,256])
plt.figure(2)
plt.plot(hist2)
plt.title("Equalized Image Histogram")
plt.show()


plt.subplot(121), plt.imshow(image, cmap="gray"), plt.title("Original Image")
plt.subplot(122), plt.imshow(equalized_image, cmap="gray"), plt.title("Equalized Image")
plt.show()
