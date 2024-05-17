import cv2
import numpy as np
import math

def generate_gaussian_kernel_x(sigma, size=7, center_x=-1, center_y=-1):
    width = int(sigma* size) | 1
    height = int(sigma* size) | 1

    if center_x == -1:
        center_x = width // 2
    if center_y == -1:
        center_y = height // 2

    x = np.arange(width) - center_x
    y = np.arange(height) - center_y
    xx, yy = np.meshgrid(x, y, sparse=True)

    dx, dy = xx ** 2, yy ** 2
    x_part, y_part = dx / (sigma** 2), dy / (sigma** 2)

    kernel = -(xx * (np.exp(-0.5 * (x_part + y_part)))) / (sigma ** 2)
    kernel /= np.min(kernel)

    return kernel
def generate_gaussian_kernel_y(sigma, size=7, center_x=-1, center_y=-1):
    width = int(sigma* size) | 1
    height = int(sigma* size) | 1

    if center_x == -1:
        center_x = width // 2
    if center_y == -1:
        center_y = height // 2

    x = np.arange(width) - center_x
    y = np.arange(height) - center_y
    xx, yy = np.meshgrid(x, y, sparse=True)

    dx, dy = xx ** 2, yy ** 2
    x_part, y_part = dx / (sigma** 2), dy / (sigma** 2)

    kernel = -(yy * (np.exp(-0.5 * (x_part + y_part)))) / (sigma ** 2)
    kernel /= np.min(kernel)

    return kernel

def normalize_image(image):
    cv2.normalize(image, image, 0, 255, cv2.NORM_MINMAX)
    return np.round(image).astype(np.uint8)

def pad_image(image, height, width, center):
    pad_top = center[0]
    pad_bottom = height - center[0] - 1
    pad_left = center[1]
    pad_right = width - center[1] - 1

    padded_image = np.pad(image, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant', constant_values=0)
    return padded_image

def convolution(image, kernel, center=(-1, -1)):
    kernel_height, kernel_width = len(kernel), len(kernel[0])

    if center == (-1, -1):
        center = (kernel_width // 2, kernel_height // 2)

    padded_image = pad_image(image, kernel_height, kernel_width, center)
    output = np.zeros_like(padded_image, dtype='float32')
    padded_height, padded_width = padded_image.shape
    kernel_center_y, kernel_center_x = center

    for y in range(kernel_center_y, padded_height - (kernel_height - kernel_center_y)):
        for x in range(kernel_center_x, padded_width - (kernel_width - kernel_center_x)):
            sum_val = 0
            for ky in range(kernel_height):
                for kx in range(kernel_width):
                    image_x = x - kernel_center_x + kx
                    image_y = y - kernel_center_y + ky
                    sum_val += kernel[ky][kx] * padded_image[image_y][image_x]

            output[y, x] = sum_val

    out_height = padded_height - kernel_height + 1
    out_width = padded_width - kernel_width + 1
    out = output[kernel_center_y:out_height, kernel_center_x:out_width]
    return out




image = cv2.imread('./images/lena.jpg', cv2.IMREAD_GRAYSCALE)

def segmentation(image, threshold_value=110):

    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    segmented_image = np.zeros_like(image)
    segmented_image[image > threshold_value] = 255

    return segmented_image

threshold_value = 110


print("Sigma: ")
sigma =float(input())


kernel_x = generate_gaussian_kernel_x(sigma=sigma,  size= 7, center_x=-1, center_y=-1 )

x_out = convolution(image=image, kernel=kernel_x, center=(-1,-1))
final_out_x = normalize_image(x_out)

kernel_y = generate_gaussian_kernel_x(sigma=sigma,  size= 7, center_x=-1, center_y=-1 )
y_out = convolution(image=image, kernel=kernel_y, center=(-1,-1))
final_out_y = normalize_image(y_out)

final = np.sqrt((final_out_x ** 2)+(final_out_y ** 2))
final_2 = normalize_image(final)

segmented_image = segmentation(final, threshold_value)


cv2.imshow('Original Image', image)
cv2.imshow('X Derivated Image', final_out_x)
cv2.imshow('Y Derivated Image', final_out_y)
cv2.imshow('Merged', final_2)
#cv2.imshow('Segmented Image', segmented_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

