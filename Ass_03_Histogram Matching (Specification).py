import numpy as np
import matplotlib.pyplot as plt

def double_gaussian(x, mu1, sigma1, mu2, sigma2):
    return (1 / (sigma1 * np.sqrt(2 * np.pi))) * np.exp(-(x - mu1)**2 / (2 * sigma1**2)) + \
           (1 / (sigma2 * np.sqrt(2 * np.pi))) * np.exp(-(x - mu2)**2 / (2 * sigma2**2))


x_values = np.arange(256)
print(x_values)
target_hist = double_gaussian(x_values, 30, 8, 165, 20)
target_hist /= np.sum(target_hist)


plt.figure(figsize=(12, 8))
plt.plot(x_values, target_hist, color='blue')
plt.title('Target Histogram (Double Gaussian)')
plt.xlabel('Pixel Intensity')
plt.ylabel('Probability')
plt.grid(True)
plt.show()



def linear_interpolation(x, xa, ya):
    if np.any(np.diff(xa) < 0):
        raise ValueError("xp array must be increasing.")
    x = np.clip(x, xa[0], xa[-1])
    indices = np.searchsorted(xa, x) - 1

    dx = (x - xa[indices]) / (xa[indices + 1] - xa[indices])
    interpolated_values = ya[indices] + dx * (ya[indices + 1] - ya[indices])

    return interpolated_values


def histogram_matching(image, target_hist):
    hist, _ = np.histogram(image.flatten(), bins=256, range=(0, 255), density=True)
    cdf = np.cumsum(hist)
    cdf_normalized = (cdf - cdf.min()) / (cdf.max() - cdf.min())

    target_cdf = np.cumsum(target_hist)
    target_cdf /= target_cdf[-1]

    mapped_intensity = linear_interpolation(cdf_normalized, target_cdf, np.arange(256))

    matched_image = mapped_intensity[image.astype(int)]

    return matched_image.astype(np.uint8)



input_image = plt.imread('.\images\\histogram.jpg')


if len(input_image.shape) == 3:
    input_gray = np.mean(input_image, axis=2)
else:
    input_gray = input_image

output_image = histogram_matching(input_gray, target_hist)

plt.figure(figsize=(12, 8))
plt.subplot(1, 2, 1)
plt.imshow(input_gray, cmap='gray')
plt.title('Input Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(output_image, cmap='gray')
plt.title('Output Image (Histogram Matched)')
plt.axis('off')
plt.show()


plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.hist(input_gray.ravel(), bins=256, range=(0, 255), density=True, color='blue', alpha=0.7)
plt.plot(x_values, target_hist, color='red')
plt.title('Input Image Histogram and Target Histogram')
plt.xlabel('Pixel Intensity')
plt.ylabel('Probability')
plt.legend(['Target Histogram', 'Input Histogram'])
plt.grid(True)

plt.subplot(2, 2, 2)
plt.plot(x_values, np.histogram(input_gray, bins=256, range=(0, 255), density=True)[0], color='blue')
plt.title('PDF of Input Image')
plt.xlabel('Pixel Intensity')
plt.ylabel('Probability')
plt.grid(True)

plt.subplot(2, 2, 3)
plt.hist(output_image.ravel(), bins=256, range=(0, 255), density=True, color='green', alpha=0.7)
plt.title('Output Image Histogram')
plt.xlabel('Pixel Intensity')
plt.ylabel('Probability')
plt.grid(True)

plt.subplot(2, 2, 4)
plt.plot(x_values, np.histogram(output_image, bins=256, range=(0, 255), density=True)[0], color='green')
plt.title('PDF of Output Image')
plt.xlabel('Pixel Intensity')
plt.ylabel('Probability')
plt.grid(True)

plt.tight_layout()
plt.show()
