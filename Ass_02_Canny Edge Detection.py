import math
import numpy as np
import cv2


#Kernel generation

def generate_gaussian_kernel(sigmaX, sigmaY, MUL=7):
    w = int(sigmaX * MUL) | 1
    h = int(sigmaY * MUL) | 1

    # print(w,h)

    cx = w // 2
    cy = h // 2

    kernel = np.zeros((w, h))
    c = 1 / (2 * 3.1416 * sigmaX * sigmaY)

    for x in range(w):
        for y in range(h):
            dx = x - cx
            dy = y - cy

            x_part = (dx * dx) / (sigmaX * sigmaX)
            y_part = (dy * dy) / (sigmaY * sigmaY)

            kernel[x][y] = c * math.exp(- 0.5 * (x_part + y_part))

    formatted_kernel = kernel / np.min(kernel)
    formatted_kernel = formatted_kernel.astype(int)

    return (kernel, formatted_kernel)



#Convolution



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


def get_kernel(sigma=0.7, MUL=7):
    kernel, _ = generate_gaussian_kernel(sigmaX=sigma, sigmaY=sigma, MUL=MUL)
    h = len(kernel)

    kernel_x = np.zeros((h, h))
    kernel_y = np.zeros((h, h))

    mn1, mn2 = 100, 100
    cx = h // 2

    for x in range(h):
        for y in range(h):
            act_x = (x - cx)
            act_y = (y - cx)

            c1, c2 = -act_x / (sigma * 2), -act_y / (sigma * 2)

            kernel_x[x, y] = c1 * kernel[x, y]
            kernel_y[x, y] = c2 * kernel[x, y]

            mn1 = min(abs(kernel_x[x, y]), mn1) if kernel_x[x, y] != 0 else mn1
            mn2 = min(abs(kernel_y[x, y]), mn2) if kernel_y[x, y] != 0 else mn2


    return kernel_y, kernel_x


def merge(image_horiz, image_vert):
    height, width = image_horiz.shape
    out = np.zeros_like(image_horiz, dtype='float32')

    for x in range(0, height):
        for y in range(0, width):
            dx = image_horiz[x, y]
            dy = image_vert[x, y]

            res = math.sqrt(dx ** 2 + dy ** 2)
            out[x, y] = res

    return out


import numpy as np


def find_avg(image, t=-1):
    mask = image > t
    mu1 = np.mean(image[~mask])
    mu2 = np.mean(image[mask])
    return (mu1 + mu2) / 2


def find_threeshold(image):
    total = np.sum(image)
    t = total / np.size(image)

    while True:
        dif = find_avg(image, t)
        if abs(dif - t) < 0.1**4:
            return dif
        t = dif


def make_binary(t, image, low=0, high=255):
    out = image.copy()
    h, w = image.shape
    for x in range(h):
        for y in range(w):
            v = image[x, y]

            out[x, y] = high if v > t else low

    return out


def perform_threshold(image, lowThresholdRatio=0.05, highThresholdRatio=0.09):
    highThreshold = image.max() * highThresholdRatio;
    lowThreshold = highThreshold * lowThresholdRatio;

    M, N = image.shape
    res = np.zeros((M, N), dtype=np.int32)

    weak = np.int32(25)
    strong = np.int32(255)

    strong_i, strong_j = np.where(image >= highThreshold)

    weak_i, weak_j = np.where((image <= highThreshold) & (image >= lowThreshold))

    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak

    return (res, weak, strong)


def perform_hysteresis(image, weak, strong=255):
    M, N = image.shape
    out = image.copy()

    for i in range(1, M - 1):
        for j in range(1, N - 1):
            if (image[i, j] == weak):
                if (
                        (image[i + 1, j - 1] == strong) or (image[i + 1, j] == strong) or
                        (image[i + 1, j + 1] == strong) or (image[i, j - 1] == strong) or
                        (image[i, j + 1] == strong) or (image[i - 1, j - 1] == strong) or
                        (image[i - 1, j] == strong) or (image[i - 1, j + 1] == strong)
                ):
                    out[i, j] = strong
                else:
                    out[i, j] = 0
    return out



def perform_edge_detection(image):
    kernel_x, kernel_y = get_kernel()

    conv_x = convolution(image=image, kernel=kernel_x)
    conv_y = convolution(image=image, kernel=kernel_y)

    kernel, _ = generate_gaussian_kernel(sigmaX=1, sigmaY=1, MUL=5)
    conv_x = convolution(image=conv_x, kernel=kernel)
    conv_y = convolution(image=conv_y, kernel=kernel)

    merged_image = merge(conv_x, conv_y)
    theta = np.arctan2(conv_y.copy(), conv_x.copy())

    merged_image_nor = normalize_image(merged_image)

    t = find_threeshold(image=merged_image_nor)
    print(f"Threshold {t}")
    final_out = make_binary(t=t, image=merged_image_nor, low=0, high=100)

    cv2.imshow("X derivative", normalize_image(conv_x))
    cv2.imshow("Y derivative", normalize_image(conv_y))
    cv2.imshow("Merged image", merged_image_nor)
    cv2.imshow("After Thresholding", final_out)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return merged_image, theta

def perform_non_maximum_suppression(image, theta):
    image = image.copy()

    image = image / image.max() * 255

    M, N = image.shape
    out = np.zeros((M, N), dtype=np.uint8)

    angle = theta * 180. / np.pi  # max -> 180, min -> -180

    for i in range(1, M - 1):
        for j in range(1, N - 1):
            q = 0
            r = 0

            ang = angle[i, j]

            if (-22.5 <= ang < 22.5) or (157.5 <= ang <= 180) or (-180 <= ang <= -157.5):
                r = image[i, j - 1]
                q = image[i, j + 1]

            elif (-67.5 <= ang <= -22.5) or (112.5 <= ang <= 157.5):
                r = image[i - 1, j + 1]
                q = image[i + 1, j - 1]

            elif (67.5 <= ang <= 112.5) or (-112.5 <= ang <= -67.5):
                r = image[i - 1, j]
                q = image[i + 1, j]

            elif (22.5 <= ang < 67.5) or (-167.5 <= ang <= -112.5):
                r = image[i + 1, j + 1]
                q = image[i - 1, j - 1]

            if (image[i, j] >= q) and (image[i, j] >= r):
                out[i, j] = image[i, j]
            else:
                out[i, j] = 0
    return out


def perform_canny(image_path, sigma):
    # Gray Scale Coversion
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Perform Gaussian Blurr
    kernel, _ = generate_gaussian_kernel(sigmaX=sigma, sigmaY=sigma, MUL=7)
    image = convolution(image=image, kernel=kernel)

    cv2.imshow("Blurred Input Image", normalize_image(image))
    cv2.waitKey(0)

    # Gradient Calculation
    image_sobel, theta = perform_edge_detection(image)

    # Non Maximum Suppression
    suppressed = perform_non_maximum_suppression(image=image_sobel, theta=theta)

    # Threesholding and hysteresis
    threes, weak, strong = perform_threshold(image=suppressed)
    final_output = perform_hysteresis(image=threes, weak=weak, strong=strong)

    cv2.imshow("After sobel", normalize_image(image_sobel))
    cv2.imshow("Non maximum suppression", normalize_image(suppressed))
    cv2.imshow("After Thresholding", normalize_image(threes))
    cv2.imshow("Final Edge detected output", normalize_image(final_output))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return final_output

print("<----------Input the value of sigma---------->")
sigma = float(input())

image_path = '.\images\\leaves.png'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
image1 = perform_canny(image_path=image_path, sigma=sigma)
#image2 = cv2.Canny(image, 50, 150)


#def find_difference(image1, image2):
#    image1 = cv2.resize(image1, (image2.shape[1], image2.shape[0]))
#    difference = cv2.absdiff(image1, image2)
#    difference = normalize_image(difference)

    #return difference
#difference = find_difference(image1 = image1,image2 = image2)

#cv2.imshow("My Experiment",image1)
#cv2.imshow("Library Function used",image2)
#cv2.imshow("Difference",difference)

