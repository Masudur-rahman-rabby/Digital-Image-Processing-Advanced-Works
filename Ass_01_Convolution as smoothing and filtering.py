import math
import numpy as np
import cv2


def generate_gaussian_kernel(sigma_x, sigma_y, size=7, center_x=-1, center_y=-1):
    width = int(sigma_x * size) | 1
    height = int(sigma_y * size) | 1

    if center_x == -1:
        center_x = width // 2
    if center_y == -1:
        center_y = height // 2

    x = np.arange(width) - center_x
    y = np.arange(height) - center_y
    xx, yy = np.meshgrid(x, y, sparse=True)

    dx, dy = xx ** 2, yy ** 2
    x_part, y_part = dx / (sigma_x ** 2), dy / (sigma_y ** 2)

    kernel = np.exp(-0.5 * (x_part + y_part))
    kernel /= np.min(kernel)

    return kernel


def generate_mean_kernel(rows=3, cols=3):
    kernel = np.ones((rows, cols)) / (rows * cols)
    return kernel


def generate_laplacian_kernel(negative_center=True, size=3):
    other_val = 1 if negative_center else -1
    kernel = other_val * np.ones((size, size))
    center = size // 2
    kernel[center, center] = - other_val * (size ** 2 - 1)
    return kernel

def generate_log_kernel(sigma, size=7):
    n = int(sigma * size)
    n = n | 1
    kernel = np.zeros((n, n))
    center = n // 2
    part1 = -1 / (np.pi * sigma ** 4)

    for x in range(n):
        for y in range(n):
            dx, dy = x - center, y - center
            part2 = (dx ** 2 + dy ** 2) / (2 * sigma ** 2)
            kernel[x][y] = part1 * (1 - part2) * np.exp(-part2)

    min_value = np.min(np.abs(kernel))
    return kernel

def generate_sobel_kernel(horizontal=True):

    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])

    sobel_y = np.array([[1, 2, 1],
                        [0, 0, 0],
                        [-1, -2, -1]])
    return sobel_x if horizontal else sobel_y


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






def find_difference(image1, image2):
    image1 = cv2.resize(image1, (image2.shape[1], image2.shape[0]))
    difference = cv2.absdiff(image1, image2)
    difference = normalize_image(difference)

    return difference


def perform_convolution(imagePath, kernel, kernel_center=(-1, -1)):
    image = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
    out_conv = convolution(image=image, kernel=kernel, center=kernel_center)
    out_nor = normalize_image(out_conv)
    cv2.imshow('Input image', image)
    cv2.imshow('Output image', out_nor)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def convolve_hsv(image, kernel, kernel_center=(-1, -1)):
    channels = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
    channels_conv = [normalize_image(convolution(channel, kernel, kernel_center)) for channel in channels]

    merged_hsv = cv2.merge(channels_conv)

    cv2.imshow("Original(HSV) image", image)
    cv2.imshow("Merged(HSV) image", merged_hsv)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return merged_hsv



def convolve_rgb(image, kernel, kernel_center=(-1, -1)):
    channels = cv2.split(image)
    channels_conv = [normalize_image(convolution(channel, kernel, kernel_center)) for channel in channels]

    merged = cv2.merge(channels_conv)

    for i, channel_name in enumerate(["Red", "Green", "Blue"]):
        cv2.imshow(f"Extracted {channel_name}", channels_conv[i])

    cv2.imshow("Original image", image)
    cv2.imshow("Merged image", merged)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return merged



def perform_sobel(imagePath, kernel_center=(-1, -1)):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    sobel_kernel_horiz = generate_sobel_kernel()
    sobel_image_horiz = convolution(image=image, kernel=sobel_kernel_horiz, center=kernel_center)

    sobel_kernel_vert = generate_sobel_kernel(horizontal=False)
    sobel_image_vert = convolution(image=image, kernel=sobel_kernel_vert, center=kernel_center)

    out = np.sqrt(sobel_image_horiz ** 2 + sobel_image_vert ** 2)
    out = normalize_image(out)

    cv2.imshow('Horizontal Sobel', sobel_image_horiz)
    cv2.imshow('Vertical Sobel', sobel_image_vert)
    cv2.imshow('Sobel Output', out)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def output_function(kernel,cx=-1,cy=-1):
    perform_convolution(imagePath=image_path, kernel=kernel, kernel_center=(cx, cy))

    image1 = convolve_rgb(image=image, kernel=kernel, kernel_center=(cx, cy))
    image2 = convolve_hsv(image=image, kernel=kernel, kernel_center=(cx, cy))
    dif = find_difference(image1=image1, image2=image2)

    cv2.imshow("Convolved_RGB", image1)
    cv2.imshow("Convolved_HSV", image2)
    cv2.imshow("Difference", dif)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



image_path = '.\images\\lena.jpg'
image = cv2.imread( image_path )

sigma_x, sigma_y, cx, cy , sigma, size = 0, 0, 0, 0 , 0 , 0

#Outputs:


def kernel_type(no):
    if (no == 1): #Gaussian filter used

        print("<------Gaussian Filter------>")
        print("Sigma x: ")
        sigma_x = float(input())
        print("Sigma y: ")
        sigma_y = float(input())
        print("Center x: ")
        cx = int(input())
        print("Center y: ")
        cy = int(input())
        kernel = generate_gaussian_kernel(sigma_x=sigma_x, sigma_y=sigma_y, size=7, center_x=cx, center_y=cy)
        print("\n<------Kernel value------>\n")
        print(kernel)
        output_function(kernel=kernel,cx=cx,cy=cy)

    elif (no == 2): #Mean filter used

        print("<------Mean Filter------>")
        print("Center x: ")
        cx = int(input())
        print("Center y: ")
        cy = int(input())
        kernel =generate_mean_kernel(rows=3, cols=3)
        print("\n<------Kernel value------>\n")
        print(kernel)
        output_function(kernel=kernel,cx=cx,cy=cy)

    elif (no == 3): #Laplacian filter used

        print("<------Laplacian Filter------>")
        print("Center x: ")
        cx = int(input())
        print("Center y: ")
        cy = int(input())
        print("Size: (3 is preferable) ")
        size = int(input())
        kernel = generate_laplacian_kernel(negative_center=True, size= size)
        print("\n<------Kernel value------>\n")
        print(kernel)
        output_function(kernel=kernel,cx=cx,cy=cy)

    elif (no == 4): #log Filter used

        print("<------Laplacian of Gaussian Filter------>")
        print("Sigma: ")
        sigma = float(input())
        print("Center x: ")
        cx = int(input())
        print("Center y: ")
        cy = int(input())
        print("Size: (7 is preferable) ")
        size = int(input())
        kernel = generate_log_kernel(sigma, size=size)
        print("\n<------Kernel value------>\n")
        print(kernel)
        output_function(kernel=kernel,cx=cx,cy=cy)

    elif (no == 5): #Sobel Filter used

        print("<------Sobel Filter------>")
        print("Center x: ")
        cx = int(input())
        print("Center y: ")
        cy = int(input())
        kernel_Horiz = generate_sobel_kernel(horizontal=True)
        print("\n<------Horizontal Kernel value------>\n")
        print(kernel_Horiz)
        kernel_vertical = generate_sobel_kernel(horizontal=False)
        print("\n<------Vertical Kernel value------>\n")
        print(kernel_vertical)
        perform_sobel(imagePath=image_path, kernel_center=(cx, cy))

    else:
        print("Give input of numbers from 1 to 4")

print("\n<------Enter Your Kernel Type------>")
no= int(input())
kernel_type(no)





