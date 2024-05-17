import cv2
import numpy as np
from tabulate import tabulate

def find_boundary(binary_image):
    kernel = np.ones((3,3), np.uint8)
    eroded = cv2.erode(binary_image, kernel, iterations=1)
    border = binary_image - eroded
    return border

def calculate_perimeter(binary_image):
    return np.count_nonzero(binary_image)

def calculate_area(binary_image):
    return np.count_nonzero(binary_image)

def calculate_max_diameter(binary_image):

    kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(binary_image, kernel, iterations=1)
    border = binary_image - eroded

    contours, _ = cv2.findContours(border, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return 0
    max_diameter = 0
    for cnt in contours:
        _, _, w, h = cv2.boundingRect(cnt)
        diameter = max(w, h)
        if diameter > max_diameter:
            max_diameter = diameter
    return max_diameter

def calculate_compactness(area, perimeter):
    return (perimeter ** 2) / (4 * np.pi * area)

def calculate_roundness(area, perimeter):
    return (4 * np.pi * area) / (perimeter ** 2)

def calculate_form_factor(area, perimeter, max_diameter):
    return (4 * np.pi * area) / (perimeter ** 2)

def calculate_descriptors(image):
    binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)[1]
    boundary = find_boundary(binary_image)
    perimeter = calculate_perimeter(boundary)
    area = calculate_area(binary_image)
    max_diameter = calculate_max_diameter(boundary)
    form_factor = calculate_form_factor(area, perimeter)
    roundness = calculate_roundness(area, perimeter)
    compactness = calculate_compactness(area, perimeter, max_diameter)
    return form_factor, roundness, compactness


train_image_paths = ["./images/t1.jpg", "./images/c1.jpg", "./images/p1.png"]
test_image_paths = ["./images/t2.jpg", "./images/c2.jpg", "./images/p2.png", "./images/st.jpg"]

train_descriptors = []
test_descriptors = []

for path in train_image_paths:
    train_img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    train_descriptors.append(calculate_descriptors(train_img))

for path in test_image_paths:
    test_img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    test_descriptors.append(calculate_descriptors(test_img))

distances_matrix = np.zeros((len(test_descriptors), len(train_descriptors)))
for i, test_descriptor in enumerate(test_descriptors):
    for j, train_descriptor in enumerate(train_descriptors):
        distances_matrix[i][j] = np.sqrt(np.sum((np.array(test_descriptor) - np.array(train_descriptor)) ** 2))

row_headers = [f'Test {i + 1}' for i in range(len(test_image_paths))]
col_headers = [f'Train {i + 1}' for i in range(len(train_image_paths))]
table = tabulate(distances_matrix, headers=col_headers, showindex=row_headers, tablefmt='grid')


with open("results.txt", "w") as file:
    file.write("Distance Matrix:\n")
    file.write(table)